"""Loaders locais de TinyStories e WikiText-2, sem baixar o Hugging Face."""

from pathlib import Path

import pytest
import torch

from moe_memory.config import load_config
from moe_memory.corpus import (
    SPECS,
    documents_from_texts,
    iter_split,
    iter_token_ids,
    source_label,
    splits_of,
)
from moe_memory.tokenizer import load_gpt2_encoding

ROOT = Path(__file__).resolve().parents[1]
STORY = "Once upon a time a small cat walked to the moon and came back. "


def _write_corpus(root: Path) -> tuple[Path, Path]:
    stories = root / "stories"
    wiki = root / "wiki"
    stories.mkdir()
    wiki.mkdir()
    (stories / "train.txt").write_text((STORY + "\n") * 30, encoding="utf-8")
    (stories / "valid.txt").write_text((STORY + "\n") * 10, encoding="utf-8")
    wiki_text = "\n".join(
        [
            " = Title = ",
            "",
            "The cat sat on the mat and looked at the moon for a long while.",
            "Then the dog ran past the gate.",
            "",
            " = Other = ",
            "",
            "Another article has several words so the block can close.",
            "",
        ]
    )
    for name in ("train.txt", "valid.txt", "test.txt"):
        (wiki / name).write_text((wiki_text + "\n") * 20, encoding="utf-8")
    return stories, wiki


def _config_with_local(tmp_path: Path, stories: Path, wiki: Path):
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace('TinyStories: ""', f'TinyStories: "{stories}"')
    text = text.replace('WikiText-2: ""', f'WikiText-2: "{wiki}"')
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_hub_ids_use_namespace_and_the_raw_wikitext_split():
    assert SPECS["WikiText-2"].hf_path == "Salesforce/wikitext"
    assert SPECS["WikiText-2"].hf_config == "wikitext-2-raw-v1"
    assert SPECS["TinyStories"].hf_path == "roneneldan/TinyStories"
    assert SPECS["TinyStories"].hf_config is None


def test_gpt2_packing_puts_eot_between_documents():
    encoding = load_gpt2_encoding()
    ids = list(iter_token_ids(["alpha", "beta"], encoding))
    alpha = encoding.encode("alpha")
    assert ids[len(alpha)] == encoding.eot_token
    assert ids.count(encoding.eot_token) == 2
    assert encoding.eot_token == 50256
    assert max(ids) < 50257


def test_wikitext_groups_articles_and_tinystories_keeps_rows():
    articles = list(
        documents_from_texts(
            "WikiText-2",
            [" = Title = ", "", "One cat.", "Two dogs.", "", "Other article."],
        )
    )
    assert articles == [" = Title = ", "One cat.\nTwo dogs.", "Other article."]
    stories = list(documents_from_texts("TinyStories", ["Story one.", "", "Story two."]))
    assert stories == ["Story one.", "Story two."]
    assert splits_of("TinyStories") == ("train", "valid")
    assert splits_of("WikiText-2") == ("train", "valid", "test")


def test_local_splits_yield_next_token_batches(tmp_path: Path):
    stories, wiki = _write_corpus(tmp_path)
    config = _config_with_local(tmp_path, stories, wiki)
    assert source_label(config, "TinyStories").startswith("local:")
    train = next(
        iter_split(
            config,
            "train",
            name="TinyStories",
            seq_len=8,
            batch_size=2,
            max_batches=1,
        )
    )
    valid = next(
        iter_split(config, "valid", name="TinyStories", seq_len=8, batch_size=1, max_batches=1)
    )
    assert train.shape == (2, 8)
    assert train.dtype == torch.long
    assert valid.shape[1] == 8
    assert int(train.min()) >= 0
    assert int(train.max()) < 50257
    with pytest.raises(ValueError, match="TinyStories"):
        next(iter_split(config, "test", name="TinyStories", seq_len=8, batch_size=1))

    for split in ("train", "valid", "test"):
        batch = next(
            iter_split(
                config,
                split,
                name="WikiText-2",
                seq_len=8,
                batch_size=2,
                max_batches=1,
            )
        )
        assert batch.shape[1] == 8
        assert int(batch.max()) < 50257


def test_active_report_selects_wikitext_when_the_name_is_omitted(tmp_path: Path):
    stories, wiki = _write_corpus(tmp_path)
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace("active: dev", "active: report")
    text = text.replace('TinyStories: ""', f'TinyStories: "{stories}"')
    text = text.replace('WikiText-2: ""', f'WikiText-2: "{wiki}"')
    path = tmp_path / "report.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path)
    batch = next(iter_split(config, "test", seq_len=8, batch_size=1, max_batches=1))
    assert batch.shape == (1, 8)
