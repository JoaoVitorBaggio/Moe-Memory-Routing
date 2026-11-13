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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY_MAX = "9663676416"
LSTM_CHECKPOINT = "outputs/qwen-lm-lstm-1000-L1-20261112-134051/checkpoint"
STORE = "models/qwen3-30b-a3b-store"
MODEL = "models/qwen3-30b-a3b"

ONLY_ALIASES = {
    "etapa1": "a32",
    "etapa2": "a128",
    "etapa3": "b",
    "etapa4": "c",
    "etapa5": "d",
    "etapa6": "e",
}

# Estimativas, não medidas. O passo LSTM de 1000 foi 3,29 s.
ESTIMATES = [
    {"id": "a32", "what": "Qwen original top-8, ctx 32, validation e test", "hours": 0.4, "ram_gib": 3.3, "vram_gib": 4.8, "disk_gib": 0},
    {"id": "a128", "what": "Qwen original top-8, ctx 128, se couber", "hours": 0.4, "ram_gib": 3.4, "vram_gib": 4.9, "disk_gib": 0},
    {"id": "b", "what": "Checkpoint LSTM L1, o mesmo protocolo", "hours": 0.6, "ram_gib": 5.8, "vram_gib": 4.2, "disk_gib": 0},
    {"id": "c", "what": "Treino gate Qwen top-4, 1000 passos", "hours": 1.3, "ram_gib": 5.6, "vram_gib": 4.8, "disk_gib": 2.3},
    {"id": "d", "what": "Eval do checkpoint top-4", "hours": 0.6, "ram_gib": 5.6, "vram_gib": 4.8, "disk_gib": 0},
    {"id": "e", "what": "Qwen3-1.7B e Qwen3-4B, download e eval", "hours": 0.8, "ram_gib": 8.0, "vram_gib": 5.0, "disk_gib": 12.3},
]

REQUIRED_SCRIPTS = (
    "scripts/eval_ppl.py",
    "scripts/train_qwen_lm.py",
    "scripts/eval_dense.py",
)


def default_out_dir(root: Path = ROOT, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return (root / f"outputs/night-paper-{stamp}").resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda a comparação do paper uma vez por etapa.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta de saída (padrão: outputs/night-paper-YYYYMMDD-HHMMSS).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Etapas separadas por vírgula (a32, a128, b, c, d, e ou etapa1..etapa6).",
    )
    return parser.parse_args(argv)


def resolve_out_dir(args: argparse.Namespace, root: Path = ROOT) -> Path:
    if args.out_dir is None:
        return default_out_dir(root)
    return (root / args.out_dir if not args.out_dir.is_absolute() else args.out_dir).resolve()


def _abs(root: Path, rel: str) -> str:
    return str((root / rel).resolve())


def _py_script(root: Path, script: str, *args: str) -> list[str]:
    return [sys.executable, _abs(root, script), *args]


def _eval(
    root: Path,
    out: str,
    split: str,
    context: str,
    router: str,
    windows: str,
    extra: list[str],
) -> list[str]:
    return _py_script(
        root,
        "scripts/eval_ppl.py",
        "--store-dir", _abs(root, STORE),
        "--model-dir", _abs(root, MODEL),
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


def stage_for(step_id: str) -> str:
    if step_id.startswith("a-"):
        return "a128" if "128" in step_id else "a32"
    if step_id.startswith("b-"):
        return "b"
    if step_id.startswith("c-"):
        return "c"
    if step_id.startswith("d-"):
        return "d"
    if step_id.startswith("e-"):
        return "e"
    raise ValueError(f"etapa desconhecida: {step_id}")


def build_steps(out_dir: Path, root: Path = ROOT) -> list[dict]:
    """Comandos Python, sem o systemd. O dry-run mostra os dois."""

    night = str(out_dir.resolve())
    baseline_out = f"{night}/qwen-topk4-1000"
    windows32 = f"{night}/windows-c32.json"
    windows128 = f"{night}/windows-c128.json"
    lstm_ckpt = _abs(root, LSTM_CHECKPOINT)
    shared = [
        "--k1", "4", "--k2", "6", "--b2", "5", "--l1", "1", "--l2", "3",
        "--lora-rank", "4", "--lora-alpha", "8", "--seed", "0",
    ]
    steps = [
        {
            "id": "a-val-32",
            "stage": "a32",
            "cmd": _eval(root, f"{night}/a-val-32.json", "validation", "32", "qwen", windows32, ["--qwen-topk", "8"]),
        },
        {
            "id": "a-test-32",
            "stage": "a32",
            "cmd": _eval(root, f"{night}/a-test-32.json", "test", "32", "qwen", f"{night}/windows-test-c32.json", ["--qwen-topk", "8"]),
        },
        {
            "id": "a-val-128",
            "stage": "a128",
            "cmd": _eval(root, f"{night}/a-val-128.json", "validation", "128", "qwen", windows128, ["--qwen-topk", "8"]),
        },
        {
            "id": "a-test-128",
            "stage": "a128",
            "cmd": _eval(root, f"{night}/a-test-128.json", "test", "128", "qwen", f"{night}/windows-test-c128.json", ["--qwen-topk", "8"]),
        },
        {
            "id": "b-val-32",
            "stage": "b",
            "cmd": _eval(
                root, f"{night}/b-val-32.json", "validation", "32", "lstm", windows32,
                ["--windows", windows32, "--checkpoint", lstm_ckpt, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-test-32",
            "stage": "b",
            "cmd": _eval(
                root, f"{night}/b-test-32.json", "test", "32", "lstm", f"{night}/windows-test-c32.json",
                ["--windows", f"{night}/windows-test-c32.json", "--checkpoint", lstm_ckpt, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-val-128",
            "stage": "b",
            "cmd": _eval(
                root, f"{night}/b-val-128.json", "validation", "128", "lstm", windows128,
                ["--windows", windows128, "--checkpoint", lstm_ckpt, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "b-test-128",
            "stage": "b",
            "cmd": _eval(
                root, f"{night}/b-test-128.json", "test", "128", "lstm", f"{night}/windows-test-c128.json",
                ["--windows", f"{night}/windows-test-c128.json", "--checkpoint", lstm_ckpt, "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "c-train-topk4",
            "stage": "c",
            "cmd": _py_script(
                root,
                "scripts/train_qwen_lm.py",
                "--steps", "1000", "--seq-len", "32", "--router", "qwen", "--qwen-topk", "4", "--seed", "0",
                "--device", "cpu", "--device-compute", "cuda",
                "--vram-budget-gb", "5", "--vram-expert-cache-gb", "3.0", "--vram-cache-dtype", "bf16",
                "--lm-head-vram", "--prefetch-vram-aware",
                "--expert-cache", "0", "--expert-prefetch-layers", "1", "--sparse-adamw",
                "--checkpoint-every", "100", "--compact-log",
                "--lora-rank", "4", "--lora-alpha", "8", "--lr", "1e-4",
                "--store-dir", _abs(root, STORE), "--model-dir", _abs(root, MODEL),
                "--out", baseline_out,
            ),
        },
        {
            "id": "d-val-32",
            "stage": "d",
            "cmd": _eval(
                root, f"{night}/d-val-32.json", "validation", "32", "qwen", windows32,
                ["--windows", windows32, "--qwen-topk", "4", "--checkpoint", f"{baseline_out}/checkpoint", "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "d-test-32",
            "stage": "d",
            "cmd": _eval(
                root, f"{night}/d-test-32.json", "test", "32", "qwen", f"{night}/windows-test-c32.json",
                ["--windows", f"{night}/windows-test-c32.json", "--qwen-topk", "4", "--checkpoint", f"{baseline_out}/checkpoint", "--sparse-adamw", *shared],
            ),
        },
        {
            "id": "e-1.7b",
            "stage": "e",
            "cmd": _py_script(
                root,
                "scripts/eval_dense.py",
                "--model", "Qwen/Qwen3-1.7B",
                "--dest", _abs(root, "models/baselines/Qwen3-1.7B"),
                "--windows", windows32,
                "--tokenizer-dir", _abs(root, MODEL),
                "--out", f"{night}/e-1.7b.json",
            ),
        },
        {
            "id": "e-4b",
            "stage": "e",
            "cmd": _py_script(
                root,
                "scripts/eval_dense.py",
                "--model", "Qwen/Qwen3-4B",
                "--dest", _abs(root, "models/baselines/Qwen3-4B"),
                "--windows", windows32,
                "--tokenizer-dir", _abs(root, MODEL),
                "--out", f"{night}/e-4b.json",
            ),
        },
    ]
    return steps


def parse_only(only: str) -> set[str]:
    wanted: set[str] = set()
    for part in only.split(","):
        token = part.strip()
        if not token:
            continue
        wanted.add(ONLY_ALIASES.get(token, token))
    return wanted


def select_steps(steps: list[dict], only: str | None) -> list[dict]:
    if only is None:
        return steps
    wanted = parse_only(only)
    return [step for step in steps if step["stage"] in wanted or step["id"] in wanted]


def warn_only_dependencies(selected: list[dict]) -> None:
    stages = {step["stage"] for step in selected}
    if "b" in stages and "a32" not in stages and "a128" not in stages:
        print(
            "aviso: etapa b costuma depender das janelas geradas na etapa a (a32/a128); "
            "confira windows-c32.json e windows-c128.json.",
            file=sys.stderr,
        )
    if "d" in stages and "c" not in stages:
        print(
            "aviso: etapa d costuma depender do checkpoint treinado na etapa c (qwen-topk4-1000/checkpoint).",
            file=sys.stderr,
        )
    if "e" in stages and "a32" not in stages:
        print(
            "aviso: etapa e costuma depender de windows-c32.json da etapa a32.",
            file=sys.stderr,
        )


def preflight(root: Path) -> None:
    if not root.is_dir():
        raise SystemExit(f"diretório do projeto ausente: {root}")
    for rel in REQUIRED_SCRIPTS:
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"script ausente: {path}")
    ckpt = root / LSTM_CHECKPOINT
    if not ckpt.exists():
        raise SystemExit(f"checkpoint L1 ausente: {ckpt}")


def launch(cmd: list[str], root: Path = ROOT) -> list[str]:
    wd = str(root.resolve())
    return [
        "systemd-run", "--user", "--collect", "--wait",
        f"--working-directory={wd}",
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


def render(steps: list[dict], root: Path = ROOT) -> str:
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
        lines.append(shlex.join(launch(step["cmd"], root)))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = resolve_out_dir(args, ROOT)
    steps = select_steps(build_steps(out_dir, ROOT), args.only)
    if args.only:
        warn_only_dependencies(steps)
    if any(_protected(step["cmd"]) for step in steps):
        raise SystemExit("o lote escreveria em models/qwen3-30b-a3b* ou apagaria arquivo")
    text = render(steps, ROOT)
    if args.dry_run:
        print(text, flush=True)
        return 0
    if out_dir.exists():
        raise SystemExit(f"pasta de saída já existe; recuso sobrescrever: {out_dir}")
    preflight(ROOT)
    out_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for step in steps:
        command = launch(step["cmd"], ROOT)
        completed = subprocess.run(command, check=False)
        results.append({"id": step["id"], "returncode": completed.returncode, "command": command})
    summary = {"estimates": ESTIMATES, "results": results, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Resumo do lote", ""]
    for result in results:
        lines.append(f"- {result['id']}: código {result['returncode']}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"failures": [item["id"] for item in results if item["returncode"] != 0]}), flush=True)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
