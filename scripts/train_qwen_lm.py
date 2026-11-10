"""Smoke de loss de linguagem no Qwen3-30B-A3B com offload.

Treina o LSTM e a LoRA dos experts. O store e o residente ficam
congelados. Não é o experimento do λ, não usa o tokenizador GPT-2 e
não sobe shard. A GTX 1060 não executa bf16: o compute é float32 na CPU.

Saída em ``outputs/qwen-lm-smoke/``, separado dos checkpoints antigos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.lm_train import (
    LAUNCH_CEILING_BYTES,
    OffloadLanguageModel,
    estimate_bytes,
    fingerprint,
    lm_step,
    rss_bytes,
    ru_maxrss_bytes,
    load_trainable,
    refuse_cuda_bf16,
    save_checkpoint,
)
from moe_memory.offload.qwen import load_resident, load_spec
from moe_memory.offload.store import ExpertStore
from moe_memory.qwen_corpus import iter_arrow_text, iter_token_windows, wikitext_arrow_files
from moe_memory.qwen_tokenizer import load_qwen_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke LM do router LSTM e da LoRA, com o store congelado.")
    parser.add_argument("--store-dir", type=Path, default=Path("models/qwen3-30b-a3b-store"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen3-30b-a3b"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen-lm-smoke"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--k1", type=int, default=4)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--b2", type=int, default=5)
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--l2", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lstm-embed", type=int, default=64)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--check-tokenizer", action="store_true")
    parser.add_argument("--memory-log", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seq_len < 2:
        parser.error("steps >= 1 e seq-len >= 2")
    if not args.check_tokenizer and args.device != "cpu":
        parser.error("este smoke é CPU. A 1060 não executa o residente bf16.")
    return args


def _load_tokenizer(model_dir: Path, expected_vocab: int | None = None):
    try:
        tokenizer = load_qwen_tokenizer(model_dir)
    except ValueError as exc:
        raise SystemExit(f"tokenizador recusado: {exc}") from exc
    config_vocab = expected_vocab
    config_path = model_dir / "config.json"
    if config_vocab is None and config_path.is_file():
        config_vocab = _config_vocab(config_path)
    if config_vocab is not None and tokenizer.vocab_size > config_vocab:
        raise SystemExit(
            f"maior id do tokenizador é {tokenizer.vocab_size - 1} e o vocab do config é {config_vocab}. "
            "Não remapeio ids."
        )
    return tokenizer, config_vocab


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_tokenizer:
        tokenizer, config_vocab = _load_tokenizer(args.model_dir)
        print(json.dumps(_tokenizer_report(tokenizer, config_vocab), ensure_ascii=False, indent=2))
        return 0
    store_path = args.store_dir / "experts.moestore"
    resident_path = args.store_dir / "resident.bin"
    config_path = args.model_dir / "config.json"
    for path in (store_path, resident_path, config_path):
        if not path.is_file():
            raise SystemExit(f"arquivo ausente: {path}. Não vou baixar.")
    store = ExpertStore.open(store_path, direct=True)
    try:
        spec = load_spec(config_path, store.header.dtype_code)
        if spec.vocab_size != _config_vocab(config_path):
            raise SystemExit("config.json e o loader discordam do vocab")
        if (spec.n_layers, spec.n_experts, spec.hidden_size, spec.moe_intermediate) != (
            store.header.n_layers,
            store.header.n_experts,
            store.header.hidden,
            store.header.intermediate,
        ):
            raise SystemExit("config.json não fecha com o cabeçalho do store")
        resident_bytes = resident_path.stat().st_size
        estimate = estimate_bytes(
            resident_bytes=resident_bytes,
            vocab=spec.vocab_size,
            hidden=spec.hidden_size,
            intermediate=spec.moe_intermediate,
            n_layers=spec.n_layers,
            n_experts=spec.n_experts,
            rank=args.lora_rank,
            seq_len=args.seq_len,
            k1=args.k1,
            lstm_embed=args.lstm_embed,
            lstm_hidden=args.lstm_hidden,
            n_heads=spec.n_heads,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
        )
        cgroup = estimate["cgroup_bytes"]
        fits = estimate["total_bytes"] <= LAUNCH_CEILING_BYTES and (cgroup is None or estimate["total_bytes"] <= cgroup)
        summary = {
            "kind": "qwen-lm-lora-lstm-smoke",
            "device": "cpu",
            "compute_dtype": "float32",
            "estimate": estimate,
            "fits_launch_ceiling": fits,
            "gpu": "não usada; residente bf16 e capability 6.1 não executam bf16",
        }
        print(json.dumps(summary, indent=2))
        if args.estimate_only:
            return 0
        if not summary["fits_launch_ceiling"]:
            raise SystemExit(
                f"estimativa {estimate['total_bytes']} bytes passa de {LAUNCH_CEILING_BYTES}. "
                "Não inicio o passo. O teto do cgroup continua 9663676416."
            )
        before = {
            "store": fingerprint(store_path),
            "resident": fingerprint(resident_path),
        }
        torch.manual_seed(args.seed)
        refuse_cuda_bf16(torch.device(args.device), spec.dtype)
        tokenizer, _config_vocab_size = _load_tokenizer(args.model_dir, spec.vocab_size)
        weights = load_resident(resident_path, spec, torch.device("cpu"))
        router = TwoHorizonLstmRouter(
            vocab_size=spec.vocab_size,
            n_layers=spec.n_layers,
            n_experts=spec.n_experts,
            l1=args.l1,
            l2=args.l2,
            k1=args.k1,
            k2=args.k2,
            b2=args.b2,
            embed_dim=args.lstm_embed,
            hidden_size=args.lstm_hidden,
        )
        model = OffloadLanguageModel(weights, router, store, rank=args.lora_rank, alpha=args.lora_alpha)
        model.memory_log = args.memory_log
        if args.memory_log:
            print(
                json.dumps({"phase": "resident-loaded", "rss_bytes": rss_bytes(), "ru_maxrss_bytes": ru_maxrss_bytes()}),
                flush=True,
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=0.0,
        )
        train_files = wikitext_arrow_files(args.cache_dir, "train")
        valid_files = wikitext_arrow_files(args.cache_dir, "validation")
        windows = list(
            iter_token_windows(
                iter_arrow_text(train_files),
                tokenizer,
                seq_len=args.seq_len,
                limit=args.steps,
            )
        )
        if len(windows) < args.steps:
            raise SystemExit(f"o treino só rendeu {len(windows)} janelas")
        probe = tokenizer.encode("Hello")
        rows = []
        first = None
        for step, window in enumerate(windows, start=1):
            tokens = torch.tensor(window, dtype=torch.long).view(1, -1)
            if first is None:
                first = tokens.clone()
            row = lm_step(model, optimizer, tokens)
            row["step"] = step
            rows.append(row)
            print(json.dumps(row), flush=True)
        valid = next(
            iter_token_windows(
                iter_arrow_text(valid_files),
                tokenizer,
                seq_len=args.seq_len,
                limit=1,
            )
        )
        with torch.no_grad():
            valid_loss_t, _plan = model.forward_window(torch.tensor(valid, dtype=torch.long).view(1, -1))
            valid_loss = float(valid_loss_t)
        args.out.mkdir(parents=True, exist_ok=True)
        checkpoint = args.out / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            step=args.steps,
            model=model,
            optimizer=optimizer,
            extra={"windows": windows, "seed": args.seed, "seq_len": args.seq_len},
        )
        routed = model.router.plan(first)[0].hot_index.detach().clone()
        head = model.router.heads[0].weight.detach().clone()
        with torch.no_grad():
            model.router.heads[0].weight.add_(1)
        if torch.equal(model.router.heads[0].weight, head):
            raise SystemExit("a perturbação do LSTM não alterou o peso")
        blob = torch.load(checkpoint, weights_only=False)
        load_trainable(model, blob)
        optimizer.load_state_dict(blob["optimizer"])
        if blob.get("step") != args.steps or not torch.equal(model.router.heads[0].weight, head):
            raise SystemExit("o arquivo do checkpoint não devolveu o peso do LSTM")
        restored = model.router.plan(first)[0].hot_index.detach().clone()
        if not torch.equal(restored, routed):
            raise SystemExit("o arquivo do checkpoint não devolveu a rota do passo treinado")
        after = {
            "store": fingerprint(store_path),
            "resident": fingerprint(resident_path),
        }
        if before != after:
            raise SystemExit("o store ou o residente mudou durante o smoke")
        payload = {
            **summary,
            "steps": rows,
            "valid_loss": valid_loss,
            "tokenizer_probe_hello": probe,
            "eos_id": tokenizer.eos_id,
            "checkpoint": str(checkpoint),
            "store_fingerprint": after["store"],
            "resident_fingerprint": after["resident"],
            "note": (
                "Smoke curto de LM + LSTM + LoRA. Não é qualidade, não é tok/s "
                "e não mediu a 1060. A entropia do router é diagnóstico, não loss."
            ),
        }
        (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"checkpoint": str(checkpoint), "valid_loss": valid_loss, "store_unchanged": True}))
        return 0
    finally:
        store.close()


def _tokenizer_report(tokenizer, config_vocab: int | None) -> dict:
    probes = {
        "Hello": tokenizer.encode("Hello"),
        "pt": tokenizer.encode("Olá, o João não vai à praça."),
        "nfd_nao": tokenizer.encode("Na\u0303o"),
        "space": tokenizer.encode(" "),
        "blank_line": tokenizer.encode("\n\n"),
        "eos_piece": tokenizer.encode("<|im_end|>"),
    }
    return {
        "vocab_size": tokenizer.vocab_size,
        "max_id": tokenizer.vocab_size - 1,
        "eos_id": tokenizer.eos_id,
        "config_vocab": config_vocab,
        "fits_config": config_vocab is None or tokenizer.vocab_size <= config_vocab,
        "probes": probes,
        "note": "ids do BPE e dos added tokens. Não carrega store nem residente.",
    }


def _config_vocab(path: Path) -> int:
    return int(json.loads(path.read_text(encoding="utf-8"))["vocab_size"])


if __name__ == "__main__":
    raise SystemExit(main())
