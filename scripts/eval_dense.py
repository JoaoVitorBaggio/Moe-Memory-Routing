"""Qwen3 denso pequeno, nas janelas já cortadas pelo tokenizer local.

Os pesos vão para ``models/baselines/``, nunca para ``models/qwen3-30b-a3b*``.
Qwen3-1.7B e Qwen3-4B usam o mesmo vocabulário de 151936 ids do A3B.
O 1.7B cabe na 1060 em bf16. O 4B não: a maior parte fica na CPU e a
GPU recebe no máximo 5 GiB. O cgroup de 9 GiB continua o teto da RAM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Parâmetros publicados vezes 2 bytes (bf16), mais tokenizer e índice.
DENSE_MODELS = {
    "Qwen/Qwen3-1.7B": {"params": 1_700_000_000, "disk_bytes": 3_800_000_000, "gpu_gib": 5},
    "Qwen/Qwen3-4B": {"params": 4_000_000_000, "disk_bytes": 8_500_000_000, "gpu_gib": 5},
}
EXPECTED_VOCAB = 151_936

DENSE_INSTALL_HINT = (
    "Pacote transformers ausente. Instale com: "
    "pip install 'moe-memory-routing[dense]' "
    "(ou: uv pip install 'moe-memory-routing[dense]')."
)


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(DENSE_INSTALL_HINT) from exc
    return AutoModelForCausalLM, AutoTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perplexidade e tok/s de um Qwen3 denso pequeno.")
    parser.add_argument("--model", choices=tuple(DENSE_MODELS), required=True)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("models/qwen3-30b-a3b"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dest is None:
        args.dest = Path("models/baselines") / args.model.split("/")[-1]
    if "qwen3-30b-a3b" in args.dest.as_posix():
        parser.error("o denso não grava em models/qwen3-30b-a3b*")
    return args


def plan(args: argparse.Namespace) -> dict:
    spec = DENSE_MODELS[args.model]
    return {
        "model": args.model,
        "dest": str(args.dest),
        "disk_bytes": spec["disk_bytes"],
        "params": spec["params"],
        "expected_vocab": EXPECTED_VOCAB,
        "gpu_gib": spec["gpu_gib"],
        "cpu_gib": 8,
        "windows": str(args.windows),
        "note": (
            "Janelas vêm do tokenizer local do A3B. "
            "Qwen3-1.7B e Qwen3-4B publicam o mesmo vocab de 151936. "
            "O 4B não cabe inteiro na 1060; o restante fica na CPU."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan(args)
    if args.dry_run:
        report["dry_run"] = True
        _write(args.out, report)
        print(json.dumps(report, indent=2), flush=True)
        return 0
    payload = json.loads(args.windows.read_text(encoding="utf-8"))
    windows = payload["windows"] if isinstance(payload, dict) else payload
    digest = payload.get("windows_sha256") if isinstance(payload, dict) else None
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from moe_memory.qwen_corpus import target_tokens, windows_sha256
    from moe_memory.qwen_tokenizer import load_qwen_tokenizer

    local = load_qwen_tokenizer(args.tokenizer_dir)
    probes = {text: local.encode(text) for text in ("Hello", " ", "\n\n")}
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=None)
    args.dest.mkdir(parents=True, exist_ok=True)
    match = tokenizer.vocab_size == EXPECTED_VOCAB and all(
        tokenizer.encode(text, add_special_tokens=False) == ids for text, ids in probes.items()
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        max_memory={0: f"{DENSE_MODELS[args.model]['gpu_gib']}GiB", "cpu": "8GiB"},
    )
    model.eval()
    import torch

    loss_sum = 0.0
    seen = 0
    started = time.perf_counter()
    with torch.no_grad():
        for window in windows:
            ids = torch.tensor(window, dtype=torch.long).view(1, -1)
            out = model(ids, labels=ids)
            count = ids.shape[1] - 1
            loss_sum += float(out.loss.detach()) * count
            seen += count
    mean = loss_sum / max(seen, 1)
    prompt = torch.tensor(windows[0][:8], dtype=torch.long).view(1, -1)
    gen_started = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=args.decode_tokens, do_sample=False)
    gen_s = time.perf_counter() - gen_started
    new_tokens = int(generated.shape[1] - prompt.shape[1])
    report.update(
        {
            "dry_run": False,
            "tokenizer_match": match,
            "local_probes": probes,
            "windows_sha256": digest or windows_sha256(windows),
            "eval_tokens": target_tokens(windows),
            "hf_loss": mean,
            "eval_perplexity": float(torch.exp(torch.tensor(mean))),
            "eval_wall_s": time.perf_counter() - started,
            "decode_tokens": new_tokens,
            "decode_s": gen_s,
            "decode_tokens_per_s": new_tokens / gen_s if gen_s else None,
            "seen_loss_tokens": seen,
        }
    )
    _write(args.out, report)
    print(json.dumps({key: report[key] for key in report if key != "local_probes"}, indent=2), flush=True)
    return 0


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
