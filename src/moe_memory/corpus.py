"""TinyStories (dev) e WikiText-2 (entrega) para a CE do próximo token.

Os dois passam pelo BPE do GPT-2. Cada documento termina em
``<|endoftext|>`` e os ids são empilhados em blocos de ``seq_len``. O
alvo não é um tensor à parte: ``task_cross_entropy`` lê
``input_ids[:, 1:]``.

WikiText-2 é o texto cru ``Salesforce/wikitext``, config
``wikitext-2-raw-v1``. O id antigo ``wikitext`` sem namespace quebra no
Hub atual. A variante já tokenizada troca o vocabulário e não serve
para este tokenizer.
Linha em branco fecha o artigo. TinyStories: cada registro é uma
história. Não há split de teste; o protocolo usa o valid.

A ordem é a do dataset, sem embaralhar. A semente da corrida mexe no
peso. O shuffle fica para o laço overnight, fora deste loader.

``corpus.active`` escolhe ``dev`` ou ``report``. ``corpus.paths`` vazio
baixa pelo Hugging Face em ``corpus.cache_dir``. TinyStories entra um
parquet por vez (download síncrono); WikiText-2 cabe inteiro.
Um diretório local tem ``train.txt``, ``valid.txt`` e, no WikiText-2,
``test.txt`` — um registro por linha.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from moe_memory.tokenizer import load_gpt2_encoding

KNOWN_CORPORA = frozenset({"TinyStories", "WikiText-2"})

_SPLIT_FILES = {
    "train": ("train.txt",),
    "valid": ("valid.txt", "validation.txt"),
    "test": ("test.txt",),
}


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    hf_path: str
    hf_config: str | None
    splits: dict[str, str]


SPECS: dict[str, CorpusSpec] = {
    "TinyStories": CorpusSpec(
        name="TinyStories",
        hf_path="roneneldan/TinyStories",
        hf_config=None,
        splits={"train": "train", "valid": "validation"},
    ),
    "WikiText-2": CorpusSpec(
        name="WikiText-2",
        hf_path="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
        splits={"train": "train", "valid": "validation", "test": "test"},
    ),
}


def splits_of(name: str) -> tuple[str, ...]:
    return tuple(_spec(name).splits)


def source_label(config, name: str) -> str:
    local = resolve_local_dir(config, name)
    if local is not None:
        return f"local:{local}"
    spec = _spec(name)
    if spec.hf_config:
        return f"huggingface:{spec.hf_path}/{spec.hf_config}"
    return f"huggingface:{spec.hf_path}"


def resolve_cache_dir(config) -> Path:
    path = Path(config.corpus.cache_dir)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def resolve_local_dir(config, name: str) -> Path | None:
    raw = config.corpus.paths.get(name, "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.is_dir():
        raise FileNotFoundError(f"corpus.paths.{name} não existe: {path}")
    return path


def iter_documents(config, split: str, *, name: str | None = None, streaming: bool | None = None) -> Iterator[str]:
    corpus_name = config.corpus.resolve(name)
    spec = _spec(corpus_name)
    if split not in spec.splits:
        available = ", ".join(spec.splits)
        raise ValueError(
            f"{corpus_name} não tem split {split!r} (disponível: {available}). "
            "TinyStories usa o valid; o test é do WikiText-2."
        )
    local = resolve_local_dir(config, corpus_name)
    if local is not None:
        rows = _local_rows(local, split)
    else:
        if streaming is None:
            streaming = corpus_name == "TinyStories"
        rows = _hf_rows(spec, split, resolve_cache_dir(config), streaming)
    return documents_from_texts(corpus_name, rows)


def documents_from_texts(name: str, texts: Iterator[str] | list[str]) -> Iterator[str]:
    if name == "WikiText-2":
        yield from _wikitext_articles(texts)
        return
    for text in texts:
        if text is None:
            continue
        cleaned = str(text).strip()
        if cleaned:
            yield cleaned


def iter_token_ids(documents: Iterator[str] | list[str], encoding) -> Iterator[int]:
    eot = encoding.eot_token
    for document in documents:
        piece = encoding.encode(document, disallowed_special=())
        if not piece:
            continue
        yield from piece
        yield eot


def iter_batches(
    tokens: Iterator[int] | list[int],
    *,
    seq_len: int,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterator[torch.Tensor]:
    """Blocos ``[batch, seq_len]`` int64. O resto que não fecha um bloco é descartado.

    O último yield pode ter menos linhas que ``batch_size`` se os tokens
    acabarem no meio de um batch. ``max_batches`` corta antes disso.
    """

    if seq_len < 2:
        raise ValueError("seq_len precisa ser >= 2 para a CE do próximo token")
    if batch_size < 1:
        raise ValueError("batch_size precisa ser >= 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches precisa ser >= 1")
    buffer: deque[int] = deque()
    batch: list[list[int]] = []
    produced = 0
    for token in tokens:
        buffer.append(int(token))
        while len(buffer) >= seq_len:
            batch.append([buffer.popleft() for _ in range(seq_len)])
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch.clear()
                produced += 1
                if max_batches is not None and produced >= max_batches:
                    return
    if batch and (max_batches is None or produced < max_batches):
        yield torch.tensor(batch, dtype=torch.long)


def iter_split(
    config,
    split: str,
    *,
    name: str | None = None,
    seq_len: int,
    batch_size: int,
    max_batches: int | None = None,
    streaming: bool | None = None,
) -> Iterator[torch.Tensor]:
    documents = iter_documents(config, split, name=name, streaming=streaming)
    encoding = load_gpt2_encoding()
    return iter_batches(
        iter_token_ids(documents, encoding),
        seq_len=seq_len,
        batch_size=batch_size,
        max_batches=max_batches,
    )


def _spec(name: str) -> CorpusSpec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"corpus {name!r} não está implementado. Conhecidos: {', '.join(sorted(KNOWN_CORPORA))}"
        ) from exc


def _wikitext_articles(lines) -> Iterator[str]:
    chunk: list[str] = []
    for line in lines:
        text = "" if line is None else str(line).rstrip("\r\n")
        if not text.strip():
            if chunk:
                yield "\n".join(chunk)
                chunk = []
            continue
        chunk.append(text)
    if chunk:
        yield "\n".join(chunk)


def _local_rows(directory: Path, split: str) -> Iterator[str]:
    names = _SPLIT_FILES.get(split)
    if names is None:
        raise ValueError(f"split local desconhecido: {split!r}")
    for name in names:
        path = directory / name
        if path.is_file():
            return _read_lines(path)
    expected = ", ".join(names)
    raise FileNotFoundError(f"não achei split {split!r} em {directory} (esperado: {expected})")


def _read_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def _hf_rows(spec: CorpusSpec, split: str, cache_dir: Path, streaming: bool) -> Iterator[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # TinyStories é grande. Um parquet por vez, download síncrono: o streaming
    # do `datasets` deixa uma thread HTTP viva e o processo aborta no shutdown
    # se o passo curto fecha o iterador no meio do arquivo.
    if spec.name == "TinyStories":
        yield from _parquet_shard_rows(spec, split, cache_dir)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "O loader Hugging Face precisa do pacote 'datasets'. "
            "Instale com: python3 -m pip install -r requirements.txt"
        ) from exc
    dataset = _load_hf(load_dataset, spec, spec.splits[split], cache_dir, streaming)
    for row in dataset:
        yield row["text"]


def _parquet_shard_rows(spec: CorpusSpec, split: str, cache_dir: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise ImportError(
            "TinyStories no Hub precisa de 'huggingface_hub' e 'pyarrow' "
            "(entram com o pacote datasets). "
            "Instale com: python3 -m pip install -r requirements.txt"
        ) from exc
    hf_split = spec.splits[split]
    files = sorted(
        path
        for path in list_repo_files(spec.hf_path, repo_type="dataset")
        if Path(path).name.startswith(hf_split + "-") and path.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"não achei parquet de {hf_split} em {spec.hf_path}")
    for filename in files:
        local = hf_hub_download(
            spec.hf_path,
            filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=64, columns=["text"]):
            for text in batch.column(0).to_pylist():
                yield text


def _load_hf(load_dataset, spec: CorpusSpec, hf_split: str, cache_dir: Path, streaming: bool):
    kwargs = {"split": hf_split, "streaming": streaming, "cache_dir": str(cache_dir)}
    try:
        if spec.hf_config is None:
            return load_dataset(spec.hf_path, **kwargs)
        return load_dataset(spec.hf_path, spec.hf_config, **kwargs)
    except TypeError:
        # datasets 3 tirou cache_dir de alguns caminhos; o diretório segue via env.
        import os

        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
        kwargs.pop("cache_dir", None)
        if spec.hf_config is None:
            return load_dataset(spec.hf_path, **kwargs)
        return load_dataset(spec.hf_path, spec.hf_config, **kwargs)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
