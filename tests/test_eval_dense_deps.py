"""Dependências do eval denso."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_dense():
    spec = importlib.util.spec_from_file_location("eval_dense", ROOT / "scripts" / "eval_dense.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_dense_import_hint_when_transformers_missing(monkeypatch):
    module = _load_eval_dense()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        module._import_transformers()
        raise AssertionError("esperava SystemExit")
    except SystemExit as exc:
        assert "moe-memory-routing[dense]" in str(exc)
        assert "transformers" in module.DENSE_INSTALL_HINT
