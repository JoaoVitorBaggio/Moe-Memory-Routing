"""WikiText-2 cru já em cache, tokenizado com o vocabulário do Qwen.

Não chama o Hub e não usa o GPT-2. O split de teste não entra no treino.
Linha em branco fecha o artigo, como no loader antigo, e cada artigo
termina no eos do Qwen para não colar dois textos.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from moe_memory.corpus import documents_from_texts
from moe_memory.qwen_tokenizer import QwenTokenizer


def wikitext_arrow_files(cache_dir: str | Path, split: str) -> list[Path]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split do WikiText-2 é train, validation ou test")
    root = Path(cache_dir) / "Salesforce___wikitext" / "wikitext-2-raw-v1"
    if not root.is_dir():
        raise FileNotFoundError(f"não achei {root}. Não vou baixar o WikiText.")
    named = [path for path in sorted(root.rglob("*.arrow")) if _name_has_split(path, split)]
    if not named:
        named = sorted(
            path
            for path in root.rglob("*.arrow")
            if path.parent.name.lower() == split or f"/{split}/" in path.as_posix().lower()
        )
    if not named:
        raise FileNotFoundError(f"não achei arrow do split {split} em {root}")
    return named


def _name_has_split(path: Path, split: str) -> bool:
    name = path.name.lower()
    if split == "train":
        return "train" in name and "validation" not in name and "test" not in name
    if split == "validation":
        return "validation" in name or "valid" in name
    return "test" in name and "validation" not in name


def iter_arrow_text(paths: list[Path]) -> Iterator[str]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_stream(source)
                batches = list(reader)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
                batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
        for batch in batches:
            if "text" not in batch.schema.names:
                raise ValueError(f"{path} não tem coluna text")
            for value in batch.column("text").to_pylist():
                yield "" if value is None else str(value)


def iter_token_windows(
    texts: Iterator[str],
    tokenizer: QwenTokenizer,
    *,
    seq_len: int,
    limit: int,
    skip: int = 0,
) -> Iterator[list[int]]:
    """Janelas contíguas. ``skip`` descarta as primeiras, para retomar o passo."""

    if seq_len < 2:
        raise ValueError("seq_len precisa ser >= 2")
    if limit < 1:
        raise ValueError("limit precisa ser >= 1")
    if skip < 0:
        raise ValueError("skip precisa ser >= 0")
    pending: list[int] = []
    produced = 0
    skipped = 0

    def emit() -> list[int] | None:
        nonlocal produced, skipped
        while len(pending) >= seq_len:
            window = pending[:seq_len]
            del pending[:seq_len]
            if skipped < skip:
                skipped += 1
                continue
            produced += 1
            return window
        return None

    for document in documents_from_texts("WikiText-2", texts):
        pending.extend(tokenizer.encode(document))
        pending.append(tokenizer.eos_id)
        while True:
            window = emit()
            if window is None:
                break
            yield window
            if produced >= limit:
                return
    window = emit()
    while window is not None and produced <= limit:
        yield window
        if produced >= limit:
            return
        window = emit()
