"""Smoke de loss de linguagem no Qwen3-30B-A3B com offload.

Treina o LSTM e a LoRA dos experts. O store e o residente ficam
congelados. Não é o experimento do λ, não usa o tokenizador GPT-2 e
não sobe shard. A GTX 1060 não executa bf16: o compute é float32 na CPU.

Saída em ``outputs/qwen-lm-smoke/``, separado dos checkpoints antigos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MALLOC_ARENA_MAX", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.lm_gpu import QWEN_VRAM_FIT_SLACK, fit_vram_cache, host_compute_threads
from moe_memory.offload.lm_train import (
    LAUNCH_CEILING_BYTES,
    OffloadLanguageModel,
    estimate_bytes,
    fingerprint,
    launch_allowed,
    load_checkpoint,
    lm_step,
    rss_bytes,
    ru_maxrss_bytes,
    make_optimizer,
    evaluate_windows,
    persist_checkpoint,
    replace_checkpoint,
    scheduled_lr,
    set_learning_rate,
    profile_lm_step,
    refuse_cuda_bf16,
    release_step_memory,
)
from moe_memory.offload.qwen import load_resident, load_spec
from moe_memory.offload.qwen_gate import QwenGateRouter
from moe_memory.offload.store import ExpertStore
from moe_memory.qwen_corpus import HOLDOUT_SPLIT, holdout_note, iter_arrow_text, iter_token_windows, wikitext_arrow_files
from moe_memory.qwen_tokenizer import load_qwen_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke LM do router LSTM e da LoRA, com o store congelado.")
    parser.add_argument("--store-dir", type=Path, default=Path("models/qwen3-30b-a3b-store"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen3-30b-a3b"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen-lm-smoke"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--router", choices=("lstm", "qwen"), default="lstm")
    parser.add_argument(
        "--qwen-topk",
        type=int,
        default=None,
        help="Top-k do gate original. Omitido usa o num_experts_per_tok (8). K=k1 iguala o LSTM.",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact-log", action="store_true")
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
    parser.add_argument("--device-compute", default="cpu")
    parser.add_argument("--vram-budget-gb", type=float, default=5.0)
    parser.add_argument("--vram-expert-cache-gb", type=float, default=0.0)
    parser.add_argument("--vram-cache-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--lm-head-vram",
        action="store_true",
        help="Sobe a lm_head uma vez em bf16 e reaproveita max/log Z no backward.",
    )
    parser.add_argument(
        "--prefetch-vram-aware",
        action="store_true",
        help="Não relê do store um expert que já está no cache de VRAM. Lê num anel pinned.",
    )
    parser.add_argument(
        "--vram-cache-sim",
        default="",
        help="Capacidades, em experts, do LRU que prevê o hit da VRAM. Vazio desliga.",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--check-tokenizer", action="store_true")
    parser.add_argument("--memory-log", action="store_true")
    parser.add_argument("--time-log", action="store_true")
    parser.add_argument("--profile-step", type=int, default=None)
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--profile-trace", action="store_true")
    parser.add_argument("--expert-cache", type=int, default=0)
    parser.add_argument("--expert-cache-bytes", type=int, default=0)
    parser.add_argument("--expert-prefetch-layers", type=int, default=0)
    parser.add_argument("--expert-prefetch-bytes", type=int, default=0)
    parser.add_argument("--expert-cache-sim", default="32,64,128")
    parser.add_argument("--valid-forward", action="store_true")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Não grava weights/, optim/ nem checkpoint.pt. O summary.json continua.",
    )
    parser.add_argument(
        "--sparse-adamw",
        action="store_true",
        help="AdamW só nas fatias LoRA tocadas. O padrão continua o AdamW denso.",
    )
    args = parser.parse_args(argv)
    if args.seq_len < 2:
        parser.error("seq-len >= 2")
    if args.steps < 0:
        parser.error("steps precisa ser >= 0")
    if args.steps == 0 and (args.router != "qwen" or args.eval_batches < 1):
        parser.error("steps 0 é a avaliação do Qwen original: --router qwen e --eval-batches >= 1")
    if args.qwen_topk is not None and (args.router != "qwen" or args.qwen_topk < 1):
        parser.error("--qwen-topk pede --router qwen e K >= 1")
    if args.eval_every < 0 or args.eval_batches < 1:
        parser.error("--eval-every >= 0 e --eval-batches >= 1")
    if args.warmup_steps < 0 or args.lr_min < 0 or args.checkpoint_every < 0:
        parser.error("warmup, lr-min e checkpoint-every precisam ser >= 0")
    if not args.check_tokenizer and args.device != "cpu":
        parser.error("este smoke é CPU. A 1060 não executa o residente bf16.")
    if args.profile_step is None and (args.profile_stack or args.profile_trace):
        parser.error("--profile-stack e --profile-trace pedem --profile-step")
    if args.profile_step is not None and not 1 <= args.profile_step <= args.steps:
        parser.error("--profile-step precisa ser um passo entre 1 e --steps")
    if args.expert_cache < 0 or args.expert_cache_bytes < 0:
        parser.error("--expert-cache e --expert-cache-bytes precisam ser >= 0")
    if args.expert_prefetch_layers < 0 or args.expert_prefetch_bytes < 0:
        parser.error("--expert-prefetch-layers e --expert-prefetch-bytes precisam ser >= 0")
    if args.device_compute not in ("cpu", "cuda"):
        parser.error("--device-compute é cpu ou cuda")
    if args.vram_budget_gb <= 0:
        parser.error("--vram-budget-gb precisa ser > 0")
    if args.vram_expert_cache_gb < 0:
        parser.error("--vram-expert-cache-gb precisa ser >= 0")
    if any(item.strip() and not item.strip().isdigit() for item in args.vram_cache_sim.split(",")):
        parser.error("--vram-cache-sim é uma lista de capacidades inteiras")
    return args


def _cuda_compute_is_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8 and bool(torch.cuda.is_bf16_supported())


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
    model = None
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
        qwen_topk = spec.experts_per_tok if args.qwen_topk is None else args.qwen_topk
        if args.router == "qwen" and qwen_topk > spec.n_experts:
            raise SystemExit(f"qwen-topk {qwen_topk} passa de {spec.n_experts} experts")
        vram_k = qwen_topk if args.router == "qwen" else None
        requested_cache = int(args.vram_expert_cache_gb * 1024**3)
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
            expert_cache_slots=args.expert_cache,
            expert_cache_max_bytes=args.expert_cache_bytes,
            expert_prefetch_layers=args.expert_prefetch_layers,
            expert_prefetch_bytes=args.expert_prefetch_bytes,
            k2=args.k2,
            compute_cuda=args.device_compute == "cuda",
            compute_elem=2 if _cuda_compute_is_bf16() else 4,
            vram_expert_cache_bytes=requested_cache,
            lm_head_vram=args.lm_head_vram,
            vram_cache_elem=2 if args.vram_cache_dtype == "bf16" else 4,
            sparse_adam_steps=args.steps if args.sparse_adamw and args.steps > 0 else 0,
            trainable_lora=args.steps > 0,
            trainable_lstm=args.router == "lstm" and args.steps > 0,
            vram_k=vram_k,
        )
        cache_bytes = requested_cache
        vram_slack = 0
        if args.router == "qwen" and args.device_compute == "cuda":
            vram_slack = QWEN_VRAM_FIT_SLACK
            cache_bytes = fit_vram_cache(int(estimate["vram_bytes"]), requested_cache, int(args.vram_budget_gb * 1024**3), vram_slack)
            if cache_bytes != requested_cache:
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
                    expert_cache_slots=args.expert_cache,
                    expert_cache_max_bytes=args.expert_cache_bytes,
                    expert_prefetch_layers=args.expert_prefetch_layers,
                    expert_prefetch_bytes=args.expert_prefetch_bytes,
                    k2=args.k2,
                    compute_cuda=True,
                    compute_elem=2 if _cuda_compute_is_bf16() else 4,
                    vram_expert_cache_bytes=cache_bytes,
                    lm_head_vram=args.lm_head_vram,
                    vram_cache_elem=2 if args.vram_cache_dtype == "bf16" else 4,
                    sparse_adam_steps=args.steps if args.sparse_adamw and args.steps > 0 else 0,
                    trainable_lora=args.steps > 0,
                    trainable_lstm=False,
                    vram_k=vram_k,
                )
        checkpoint_bytes = int(estimate["lora_fp32_bytes"] + estimate["adam_lora_bytes"] + estimate["lstm_with_adam_bytes"] - estimate["lstm_grad_bytes"])
        cgroup = estimate["cgroup_bytes"]
        fits_cgroup = cgroup is None or estimate["cgroup_total_bytes"] <= cgroup
        fits = launch_allowed(estimate)
        vram_budget = int(args.vram_budget_gb * 1024**3)
        fits_vram = args.device_compute == "cpu" or int(estimate["vram_bytes"]) + vram_slack <= vram_budget
        summary = {
            "kind": "qwen-lm-lora-lstm-smoke",
            "device": "cpu",
            "compute_device": args.device_compute,
            "compute_dtype": "bfloat16" if args.device_compute == "cuda" and _cuda_compute_is_bf16() else "float32",
            "estimate": estimate,
            "fits_launch_ceiling": fits,
            "fits_cgroup_estimate": fits_cgroup,
            "fits_vram": fits_vram,
            "vram_budget_bytes": vram_budget,
            "vram_expert_cache_gb": args.vram_expert_cache_gb,
            "vram_expert_cache_bytes_used": cache_bytes,
            "qwen_topk": qwen_topk if args.router == "qwen" else None,
            "qwen_vram_slack_bytes": vram_slack,
            "vram_cache_dtype": args.vram_cache_dtype,
            "lm_head_vram": args.lm_head_vram,
            "prefetch_vram_aware": args.prefetch_vram_aware,
            "sparse_adamw": args.sparse_adamw,
            "router": args.router,
            "checkpoint_bytes": checkpoint_bytes,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "holdout": holdout_note(),
            "lr_schedule": args.lr_schedule,
            "warmup_steps": args.warmup_steps,
            "checkpoint_every": args.checkpoint_every,
            "epochs": "uma passagem do split train; o loader não repete",
            "vram_cache_sim": args.vram_cache_sim,
            "expert_cache_slots": args.expert_cache,
            "expert_cache_max_bytes": args.expert_cache_bytes,
            "expert_prefetch_layers": args.expert_prefetch_layers,
            "expert_prefetch_bytes": args.expert_prefetch_bytes,
            "gpu": (
                "uma camada por vez; Pascal fica em fp32"
                if args.device_compute == "cuda"
                else "não usada; residente bf16 e capability 6.1 não executam bf16"
            ),
        }
        print(json.dumps(summary, indent=2))
        if args.estimate_only:
            return 0
        if not summary["fits_launch_ceiling"]:
            raise SystemExit(
                f"estimativa {estimate['total_bytes']} bytes passa de {LAUNCH_CEILING_BYTES}. "
                "Não inicio o passo. O teto do cgroup continua 9663676416."
            )
        if not fits_vram:
            raise SystemExit(
                f"VRAM estimada {estimate['vram_bytes']} bytes passa de {vram_budget}. "
                "Não carrego o modelo inteiro na GPU."
            )
        before = {
            "store": fingerprint(store_path),
            "resident": fingerprint(resident_path),
        }
        torch.manual_seed(args.seed)
        refuse_cuda_bf16(torch.device(args.device), spec.dtype)
        tokenizer, _config_vocab_size = _load_tokenizer(args.model_dir, spec.vocab_size)
        resident_started = time.perf_counter()
        weights = load_resident(resident_path, spec, torch.device("cpu"), with_gate=args.router == "qwen")
        resident_load_s = time.perf_counter() - resident_started
        if args.router == "qwen":
            router = QwenGateRouter(
                vocab_size=spec.vocab_size,
                n_layers=spec.n_layers,
                n_experts=spec.n_experts,
                experts_per_tok=qwen_topk,
            )
        else:
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
        model = OffloadLanguageModel(
            weights,
            router,
            store,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            use_lora=args.steps > 0,
        )
        model.set_expert_cache(args.expert_cache, args.expert_cache_bytes)
        model.set_prefetch_vram_aware(args.prefetch_vram_aware)
        model.set_expert_prefetch(args.expert_prefetch_layers, args.expert_prefetch_bytes)
        model.set_vram_expert_cache(cache_bytes, args.vram_cache_dtype)
        if args.device_compute == "cuda":
            torch.set_num_threads(host_compute_threads(os.cpu_count() or 1, True))
            model.set_compute_device(torch.device("cuda"), vram_budget)
        if args.lm_head_vram:
            model.enable_lm_head_vram()
        model.memory_log = args.memory_log
        model.time_log = args.time_log
        if args.time_log:
            model.cache_sim_sizes = tuple(int(item) for item in args.expert_cache_sim.split(",") if item)
            model.vram_cache_sim_sizes = tuple(int(item) for item in args.vram_cache_sim.split(",") if item)
            print(
                json.dumps(
                    {
                        "phase": "resident-load",
                        "resident_load_s": resident_load_s,
                        "torch_threads": torch.get_num_threads(),
                    }
                ),
                flush=True,
            )
        if args.memory_log:
            print(
                json.dumps({"phase": "resident-loaded", "rss_bytes": rss_bytes(), "ru_maxrss_bytes": ru_maxrss_bytes()}),
                flush=True,
            )
        optimizer = None if args.steps == 0 else make_optimizer(model, args.lr, sparse=args.sparse_adamw)
        checkpoint = args.out / "checkpoint"
        start_step = 0
        if args.resume:
            if optimizer is None or not checkpoint.is_dir():
                raise SystemExit("retomar pede um checkpoint e steps > 0")
            resumed = load_checkpoint(checkpoint, model, optimizer)
            start_step = int(resumed["step"])
            if start_step >= args.steps:
                raise SystemExit(f"o checkpoint já está no passo {start_step}")
        holdout = []
        if args.steps == 0 or args.eval_every > 0:
            holdout = list(
                iter_token_windows(
                    iter_arrow_text(wikitext_arrow_files(args.cache_dir, HOLDOUT_SPLIT)),
                    tokenizer,
                    seq_len=args.seq_len,
                    limit=args.eval_batches,
                )
            )
            if len(holdout) < args.eval_batches:
                raise SystemExit(f"a validação só rendeu {len(holdout)} janelas")
        windows = []
        if args.steps > 0:
            train_files = wikitext_arrow_files(args.cache_dir, "train")
            windows = list(
                iter_token_windows(
                    iter_arrow_text(train_files),
                    tokenizer,
                    seq_len=args.seq_len,
                    skip=start_step,
                    limit=args.steps - start_step,
                )
            )
            if len(windows) < args.steps - start_step:
                raise SystemExit(
                    f"o treino só rendeu {len(windows)} janelas novas. "
                    "O loader não repete o split train."
                )
        probe = tokenizer.encode("Hello")
        rows = []
        evals = []

        def _eval(step: int) -> None:
            if not holdout:
                return
            started_eval = time.perf_counter()
            metrics = evaluate_windows(model, holdout)
            metrics["step"] = step
            metrics["eval_s"] = time.perf_counter() - started_eval
            evals.append(metrics)
            print(json.dumps({"phase": "eval", **metrics}), flush=True)

        if args.eval_every > 0 or args.steps == 0:
            _eval(start_step)
        for offset, window in enumerate(windows, start=1):
            step = start_step + offset
            tokens = torch.tensor(window, dtype=torch.long).view(1, -1)
            if args.lr_schedule != "constant" or args.warmup_steps:
                set_learning_rate(
                    optimizer,
                    scheduled_lr(
                        step,
                        args.steps,
                        args.lr,
                        warmup=args.warmup_steps,
                        schedule=args.lr_schedule,
                        lr_min=args.lr_min,
                    ),
                )
            step_started = time.perf_counter()
            if args.profile_step == step:
                with profile_lm_step(
                    args.out / f"profile-step-{step}",
                    with_stack=args.profile_stack,
                    with_trace=args.profile_trace,
                ):
                    row = lm_step(model, optimizer, tokens)
            else:
                row = lm_step(model, optimizer, tokens)
            row["step"] = step
            row["lr"] = float(optimizer.param_groups[0]["lr"])
            row["step_s"] = time.perf_counter() - step_started
            rows.append(row)
            if args.compact_log:
                print(
                    f"passo {step}: loss={row['loss']:.4f} lr={row['lr']:.3e} "
                    f"s={row['step_s']:.2f} leituras={row['store_reads']}",
                    flush=True,
                )
            else:
                print(json.dumps(row), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                args.out.mkdir(parents=True, exist_ok=True)
                replace_checkpoint(
                    checkpoint,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    extra={"seed": args.seed, "seq_len": args.seq_len, "router": args.router},
                )
            if args.eval_every and step % args.eval_every == 0:
                _eval(step)
        if optimizer is not None:
            release_step_memory(optimizer)
        valid_loss = None
        if args.valid_forward:
            valid_files = wikitext_arrow_files(args.cache_dir, "validation")
            valid = next(
                iter_token_windows(
                    iter_arrow_text(valid_files),
                    tokenizer,
                    seq_len=args.seq_len,
                    limit=1,
                )
            )
            with torch.no_grad():
                valid_loss_t, valid_plan = model.forward_window(torch.tensor(valid, dtype=torch.long).view(1, -1))
                valid_loss = float(valid_loss_t)
                del valid_loss_t, valid_plan
            if optimizer is not None:
                release_step_memory(optimizer)
            if args.memory_log:
                print(
                    json.dumps({"phase": "valid-forward", "rss_bytes": rss_bytes(), "ru_maxrss_bytes": ru_maxrss_bytes()}),
                    flush=True,
                )
        model.release_expert_cache()
        args.out.mkdir(parents=True, exist_ok=True)
        checkpoint_saved = False
        if optimizer is not None and not args.no_checkpoint:
            checkpoint_started = time.perf_counter()
            extra = {"windows": windows, "seed": args.seed, "seq_len": args.seq_len, "router": args.router}
            if args.checkpoint_every:
                replace_checkpoint(checkpoint, step=args.steps, model=model, optimizer=optimizer, extra=extra)
                checkpoint_saved = True
            else:
                checkpoint_saved = persist_checkpoint(
                    True,
                    checkpoint,
                    step=args.steps,
                    model=model,
                    optimizer=optimizer,
                    extra=extra,
                )
            checkpoint_s = time.perf_counter() - checkpoint_started
            if args.memory_log or args.time_log:
                print(
                    json.dumps(
                        {
                            "phase": "checkpoint",
                            "checkpoint_save_s": checkpoint_s,
                            "rss_bytes": rss_bytes(),
                            "ru_maxrss_bytes": ru_maxrss_bytes(),
                        }
                    ),
                    flush=True,
                )
        after = {
            "store": fingerprint(store_path),
            "resident": fingerprint(resident_path),
        }
        if before != after:
            raise SystemExit("o store ou o residente mudou durante o smoke")
        payload = {
            **summary,
            "steps": rows,
            "evals": evals,
            "holdout_split": HOLDOUT_SPLIT if evals else None,
            "valid_loss": valid_loss,
            "tokenizer_probe_hello": probe,
            "eos_id": tokenizer.eos_id,
            "lm_head_h2d_bytes": model.lm_head_h2d_bytes,
            "lm_head_cpu_released": model.lm_head_cpu_released,
            "checkpoint_saved": checkpoint_saved,
            "checkpoint": str(checkpoint) if checkpoint_saved else None,
            "store_fingerprint": after["store"],
            "resident_fingerprint": after["resident"],
            "note": (
                "Smoke curto de LM + LSTM + LoRA. Não é qualidade, não é tok/s "
                "e não mediu a 1060. A entropia do router é diagnóstico, não loss."
            ),
        }
        (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"checkpoint_saved": checkpoint_saved, "checkpoint": str(checkpoint) if checkpoint_saved else None, "valid_loss": valid_loss, "store_unchanged": True}))
        return 0
    finally:
        if model is not None:
            model.release_expert_cache()
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
