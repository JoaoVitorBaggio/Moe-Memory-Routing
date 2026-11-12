"""Lote noturno. Cada etapa roda uma vez. Falha registra e segue.

O launcher é um serviço transitório do systemd do usuário, com teto de
9 GiB, swap zero e I/O idle. Não apaga checkpoint antigo e não grava em
``models/qwen3-30b-a3b*``. O denso vai para ``models/baselines/``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY_MAX = "9663676416"
LSTM_CHECKPOINT = "outputs/qwen-lm-lstm-1000-L1-20261112-134051/checkpoint"
STORE = "models/qwen3-30b-a3b-store"
MODEL = "models/qwen3-30b-a3b"
NIGHT = "outputs/night-paper"
BASELINE_OUT = f"{NIGHT}/qwen-topk4-1000"

# Estimativas, não medidas. O passo LSTM de 1000 foi 3,29 s.
ESTIMATES = [
    {"id": "a32", "what": "Qwen original top-8, ctx 32, validation e test", "hours": 0.4, "ram_gib": 3.3, "vram_gib": 4.8, "disk_gib": 0},
    {"id": "a128", "what": "Qwen original top-8, ctx 128, se couber", "hours": 0.4, "ram_gib": 3.4, "vram_gib": 4.9, "disk_gib": 0},
    {"id": "b", "what": "Checkpoint LSTM L1, o mesmo protocolo", "hours": 0.6, "ram_gib": 5.8, "vram_gib": 4.2, "disk_gib": 0},
    {"id": "c", "what": "Treino gate Qwen top-4, 1000 passos", "hours": 1.3, "ram_gib": 5.6, "vram_gib": 4.8, "disk_gib": 2.3},
    {"id": "d", "what": "Eval do checkpoint top-4", "hours": 0.6, "ram_gib": 5.6, "vram_gib": 4.8, "disk_gib": 0},
    {"id": "e", "what": "Qwen3-1.7B e Qwen3-4B, download e eval", "hours": 0.8, "ram_gib": 8.0, "vram_gib": 5.0, "disk_gib": 12.3},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda a comparação do paper uma vez por etapa.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(NIGHT))
    return parser.parse_args(argv)


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _eval(out: str, split: str, context: str, router: str, windows: str, extra: list[str]) -> list[str]:
    return _py(
        "scripts/eval_ppl.py",
        "--store-dir", STORE,
        "--model-dir", MODEL,
        "--split", split,
        "--context-len", context,
        "--eval-tokens", "8192",
        "--router", router,
        "--device", "cpu",
        "--device-compute", "cuda",
        "--vram-budget-gb", "5",
        "--vram-expert-cache-gb", "3.0",
        "--vram-cache-dtype", "bf16",
        "--lm-head-vram",
        "--prefetch-vram-aware",
        "--expert-cache", "0",
        "--expert-prefetch-layers", "1",
        "--windows-out", windows,
        "--out", out,
        *extra,
    )


def build_steps() -> list[dict]:
    """Comandos Python, sem o systemd. O dry-run mostra os dois."""

    windows32 = f"{NIGHT}/windows-c32.json"
    windows128 = f"{NIGHT}/windows-c128.json"
    shared = [
        "--k1", "4", "--k2", "6", "--b2", "5", "--l1", "1", "--l2", "3",
        "--lora-rank", "4", "--lora-alpha", "8", "--seed", "0",
    ]
    steps = [
        {
            "id": "a-val-32",
            "cmd": _eval(f"{NIGHT}/a-val-32.json", "validation", "32", "qwen", windows32, ["--qwen-topk", "8"]),
        },
        {
            "id": "a-test-32",
            "cmd": _eval(f"{NIGHT}/a-test-32.json", "test", "32", "qwen", f"{NIGHT}/windows-test-c32.json", ["--qwen-topk", "8"]),
        },
        {
            "id": "a-val-128",
            "cmd": _eval(f"{NIGHT}/a-val-128.json", "validation", "128", "qwen", windows128, ["--qwen-topk", "8"]),
        },
        {
            "id": "a-test-128",
            "cmd": _eval(f"{NIGHT}/a-test-128.json", "test", "128", "qwen", f"{NIGHT}/windows-test-c128.json", ["--qwen-topk", "8"]),
        },
        {
            "id": "b-val-32",
            "cmd": _eval(
                f"{NIGHT}/b-val-32.json", "validation", "32", "lstm", windows32,
                ["--windows", windows32, "--checkpoint", LSTM_CHECKPOINT, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-test-32",
            "cmd": _eval(
                f"{NIGHT}/b-test-32.json", "test", "32", "lstm", f"{NIGHT}/windows-test-c32.json",
                ["--windows", f"{NIGHT}/windows-test-c32.json", "--checkpoint", LSTM_CHECKPOINT, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-val-128",
            "cmd": _eval(
                f"{NIGHT}/b-val-128.json", "validation", "128", "lstm", windows128,
                ["--windows", windows128, "--checkpoint", LSTM_CHECKPOINT, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-test-128",
            "cmd": _eval(
                f"{NIGHT}/b-test-128.json", "test", "128", "lstm", f"{NIGHT}/windows-test-c128.json",
                ["--windows", f"{NIGHT}/windows-test-c128.json", "--checkpoint", LSTM_CHECKPOINT, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "c-train-topk4",
            "cmd": _py(
                "scripts/train_qwen_lm.py",
                "--steps", "1000", "--seq-len", "32", "--router", "qwen", "--qwen-topk", "4", "--seed", "0",
                "--device", "cpu", "--device-compute", "cuda",
                "--vram-budget-gb", "5", "--vram-expert-cache-gb", "3.0", "--vram-cache-dtype", "bf16",
                "--lm-head-vram", "--prefetch-vram-aware",
                "--expert-cache", "0", "--expert-prefetch-layers", "1", "--sparse-adamw",
                "--checkpoint-every", "100", "--compact-log",
                "--lora-rank", "4", "--lora-alpha", "8", "--lr", "1e-4",
                "--store-dir", STORE, "--model-dir", MODEL,
                "--out", BASELINE_OUT,
            ),
        },
        {
            "id": "d-val-32",
            "cmd": _eval(
                f"{NIGHT}/d-val-32.json", "validation", "32", "qwen", windows32,
                ["--windows", windows32, "--qwen-topk", "4", "--checkpoint", f"{BASELINE_OUT}/checkpoint", "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "d-test-32",
            "cmd": _eval(
                f"{NIGHT}/d-test-32.json", "test", "32", "qwen", f"{NIGHT}/windows-test-c32.json",
                ["--windows", f"{NIGHT}/windows-test-c32.json", "--qwen-topk", "4", "--checkpoint", f"{BASELINE_OUT}/checkpoint", "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "e-1.7b",
            "cmd": _py(
                "scripts/eval_dense.py",
                "--model", "Qwen/Qwen3-1.7B",
                "--dest", "models/baselines/Qwen3-1.7B",
                "--windows", windows32,
                "--tokenizer-dir", MODEL,
                "--out", f"{NIGHT}/e-1.7b.json",
            ),
        },
        {
            "id": "e-4b",
            "cmd": _py(
                "scripts/eval_dense.py",
                "--model", "Qwen/Qwen3-4B",
                "--dest", "models/baselines/Qwen3-4B",
                "--windows", windows32,
                "--tokenizer-dir", MODEL,
                "--out", f"{NIGHT}/e-4b.json",
            ),
        },
    ]
    return steps


def launch(cmd: list[str]) -> list[str]:
    return [
        "systemd-run", "--user", "--collect", "--wait",
        "-p", f"MemoryMax={MEMORY_MAX}",
        "-p", "MemorySwapMax=0",
        "-p", "OOMPolicy=stop",
        "-p", "Nice=10",
        "-p", "IOSchedulingClass=idle",
        "--",
        "env", "MALLOC_ARENA_MAX=2",
        *cmd,
    ]


def _protected(cmd: list[str]) -> bool:
    text = " ".join(cmd)
    if "rm " in text or "rm\n" in text:
        return True
    for flag in ("--out", "--dest", "--windows-out"):
        if flag in cmd:
            value = cmd[cmd.index(flag) + 1]
            if "models/qwen3-30b-a3b" in value:
                return True
    return False


def render(steps: list[dict]) -> str:
    lines = ["# Lote noturno", "", "Cada etapa uma vez. Falha registra e segue. Sem retry.", ""]
    for item in ESTIMATES:
        lines.append(
            f"- {item['id']}: {item['what']}. Estimativa {item['hours']} h, "
            f"RAM {item['ram_gib']} GiB, VRAM {item['vram_gib']} GiB, disco {item['disk_gib']} GiB."
        )
    lines.append(f"- total estimado: {sum(item['hours'] for item in ESTIMATES):.1f} h, dentro de 8 h.")
    lines.append("")
    for step in steps:
        lines.append(f"## {step['id']}")
        lines.append(shlex.join(launch(step["cmd"])))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = build_steps()
    if any(_protected(step["cmd"]) for step in steps):
        raise SystemExit("o lote escreveria em models/qwen3-30b-a3b* ou apagaria arquivo")
    text = render(steps)
    if args.dry_run:
        print(text, flush=True)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for step in steps:
        command = launch(step["cmd"])
        completed = subprocess.run(command, cwd=ROOT, check=False)
        results.append({"id": step["id"], "returncode": completed.returncode, "command": command})
    summary = {"estimates": ESTIMATES, "results": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Resumo do lote", ""]
    for result in results:
        lines.append(f"- {result['id']}: código {result['returncode']}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"failures": [item["id"] for item in results if item["returncode"] != 0]}), flush=True)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
