"""Perplexidade do Qwen com offload, nas mesmas janelas para todo modelo.

Lê o split pedido do WikiText-2 raw, corta janelas contíguas e grava o
SHA-256 dessa lista. Um checkpoint treinável entra por cima do mesmo
store. Sem checkpoint e com ``--router qwen``, avalia o modelo base,
sem LoRA. Se o contexto não cabe na VRAM, grava ``skipped`` e sai 0.
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
from moe_memory.offload.lm_gpu import QWEN_VRAM_FIT_SLACK, fit_vram_cache
from moe_memory.offload.lm_train import (
    OffloadLanguageModel,
    estimate_bytes,
    evaluate_windows,
    load_checkpoint,
    make_optimizer,
)
from moe_memory.offload.qwen import load_resident, load_spec
from moe_memory.offload.qwen_gate import QwenGateRouter
from moe_memory.offload.store import ExpertStore
from moe_memory.qwen_corpus import (
    iter_arrow_text,
    paper_windows,
    target_tokens,
    windows_sha256,
    wikitext_arrow_files,
)
from moe_memory.qwen_tokenizer import load_qwen_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perplexidade comparável no WikiText-2.")
    parser.add_argument("--store-dir", type=Path, default=Path("models/qwen3-30b-a3b-store"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen3-30b-a3b"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--router", choices=("lstm", "qwen"), default="qwen")
    parser.add_argument("--qwen-topk", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sparse-adamw", action="store_true")
    parser.add_argument("--windows", type=Path, default=None)
    parser.add_argument("--windows-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k1", type=int, default=4)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--b2", type=int, default=5)
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--l2", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lstm-embed", type=int, default=64)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-compute", default="cpu")
    parser.add_argument("--vram-budget-gb", type=float, default=5.0)
    parser.add_argument("--vram-expert-cache-gb", type=float, default=0.0)
    parser.add_argument("--vram-cache-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--lm-head-vram", action="store_true")
    parser.add_argument("--prefetch-vram-aware", action="store_true")
    parser.add_argument("--expert-cache", type=int, default=0)
    parser.add_argument("--expert-prefetch-layers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.context_len < 2 or args.eval_tokens < 1:
        parser.error("context-len >= 2 e eval-tokens >= 1")
    if args.qwen_topk is not None and (args.router != "qwen" or args.qwen_topk < 1):
        parser.error("--qwen-topk pede --router qwen e K >= 1")
    if args.device != "cpu":
        parser.error("o residente fica na CPU")
    if args.device_compute not in ("cpu", "cuda"):
        parser.error("--device-compute é cpu ou cuda")
    return args


def load_or_make_windows(args, tokenizer) -> list[list[int]]:
    if args.windows is not None:
        payload = json.loads(args.windows.read_text(encoding="utf-8"))
        windows = payload["windows"] if isinstance(payload, dict) else payload
        return windows
    files = wikitext_arrow_files(args.cache_dir, args.split)
    return paper_windows(
        iter_arrow_text(files),
        tokenizer,
        context_len=args.context_len,
        eval_tokens=args.eval_tokens,
    )


def _estimate(args, spec, resident_bytes: int, cache_bytes: int, topk: int | None):
    return estimate_bytes(
        resident_bytes=resident_bytes,
        vocab=spec.vocab_size,
        hidden=spec.hidden_size,
        intermediate=spec.moe_intermediate,
        n_layers=spec.n_layers,
        n_experts=spec.n_experts,
        rank=args.lora_rank,
        seq_len=args.context_len,
        k1=args.k1,
        lstm_embed=args.lstm_embed,
        lstm_hidden=args.lstm_hidden,
        n_heads=spec.n_heads,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        expert_cache_slots=args.expert_cache,
        expert_prefetch_layers=args.expert_prefetch_layers,
        k2=args.k2,
        compute_cuda=args.device_compute == "cuda",
        compute_elem=4,
        vram_expert_cache_bytes=cache_bytes,
        lm_head_vram=args.lm_head_vram,
        vram_cache_elem=2 if args.vram_cache_dtype == "bf16" else 4,
        trainable_lora=args.checkpoint is not None,
        trainable_lstm=args.router == "lstm" and args.checkpoint is not None,
        vram_k=topk,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = load_qwen_tokenizer(args.model_dir)
    windows = load_or_make_windows(args, tokenizer)
    digest = windows_sha256(windows)
    if args.windows_out is not None:
        args.windows_out.parent.mkdir(parents=True, exist_ok=True)
        args.windows_out.write_text(
            json.dumps({"split": args.split, "context_len": args.context_len, "windows_sha256": digest, "windows": windows}),
            encoding="utf-8",
        )
    store_path = args.store_dir / "experts.moestore"
    resident_path = args.store_dir / "resident.bin"
    config_path = args.model_dir / "config.json"
    for path in (store_path, resident_path, config_path):
        if not path.is_file():
            raise SystemExit(f"arquivo ausente: {path}")
    store = ExpertStore.open(store_path, direct=True)
    model = None
    try:
        spec = load_spec(config_path, store.header.dtype_code)
        topk = spec.experts_per_tok if args.qwen_topk is None else args.qwen_topk
        if args.router == "qwen" and topk > spec.n_experts:
            raise SystemExit(f"qwen-topk {topk} passa de {spec.n_experts}")
        vram_k = topk if args.router == "qwen" else None
        requested = int(args.vram_expert_cache_gb * 1024**3)
        budget = int(args.vram_budget_gb * 1024**3)
        estimate = _estimate(args, spec, resident_path.stat().st_size, requested, vram_k)
        cache_bytes = requested
        slack = 0
        if args.router == "qwen" and args.device_compute == "cuda":
            slack = QWEN_VRAM_FIT_SLACK
            cache_bytes = fit_vram_cache(int(estimate["vram_bytes"]), requested, budget, slack)
            estimate = _estimate(args, spec, resident_path.stat().st_size, cache_bytes, vram_k)
        report = {
            "split": args.split,
            "context_len": args.context_len,
            "eval_tokens_requested": args.eval_tokens,
            "eval_tokens": target_tokens(windows),
            "windows": len(windows),
            "windows_sha256": digest,
            "router": args.router,
            "qwen_topk": topk if args.router == "qwen" else None,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "vram_expert_cache_bytes_used": cache_bytes,
            "vram_bytes": int(estimate["vram_bytes"]),
            "vram_slack_bytes": slack,
        }
        if args.device_compute == "cuda" and int(estimate["vram_bytes"]) + slack > budget:
            report["skipped"] = True
            report["reason"] = "o contexto não cabe na VRAM com a folga do modo qwen"
            _write(args.out, report)
            print(json.dumps(report), flush=True)
            return 0
        torch.manual_seed(args.seed)
        weights = load_resident(resident_path, spec, torch.device("cpu"), with_gate=args.router == "qwen")
        use_lora = args.checkpoint is not None
        if args.router == "qwen":
            router = QwenGateRouter(
                vocab_size=spec.vocab_size,
                n_layers=spec.n_layers,
                n_experts=spec.n_experts,
                experts_per_tok=topk,
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
            weights, router, store, rank=args.lora_rank, alpha=args.lora_alpha, use_lora=use_lora
        )
        model.set_expert_cache(args.expert_cache)
        model.set_prefetch_vram_aware(args.prefetch_vram_aware)
        model.set_expert_prefetch(args.expert_prefetch_layers)
        model.set_vram_expert_cache(cache_bytes, args.vram_cache_dtype)
        if args.device_compute == "cuda":
            from moe_memory.offload.lm_gpu import host_compute_threads

            torch.set_num_threads(host_compute_threads(os.cpu_count() or 1, True))
            model.set_compute_device(torch.device("cuda"), budget)
        if args.lm_head_vram:
            model.enable_lm_head_vram()
        if args.checkpoint is not None:
            optimizer = make_optimizer(model, 1e-4, sparse=args.sparse_adamw)
            load_checkpoint(args.checkpoint, model, optimizer)
        started = time.perf_counter()
        metrics = evaluate_windows(model, windows)
        report.update(metrics)
        report["eval_wall_s"] = time.perf_counter() - started
        report["skipped"] = False
        _write(args.out, report)
        print(json.dumps({key: report[key] for key in report if key != "windows"}), flush=True)
        return 0
    finally:
        if model is not None:
            model.release_expert_cache()
        store.close()


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
