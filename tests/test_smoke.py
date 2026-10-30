"""Um passo real a partir de configs/default.yaml, em CPU, sem OOM."""

import math
from pathlib import Path

import numpy as np

from moe_memory.config import load_config
from moe_memory.smoke import run_eval_pass, run_train_step


def test_train_step_and_eval_pass_from_the_default_config(tmp_path: Path):
    config = load_config()
    trained = run_train_step(config, work_dir=tmp_path / "train")
    # 47_939_328 = embedding GPT-2 amarrada (~12,9 M) + 4 camadas MoE com
    # d_ff=512 (~33,7 M) + atenção. Sem o laço do lm_head passa de 60 M.
    assert trained["parameters"] == 47_939_328
    assert 10_000_000 <= trained["parameters"] <= 50_000_000
    assert math.isfinite(trained["total"])
    assert math.isfinite(trained["task"])
    assert math.isfinite(trained["c_mem"])
    assert math.isfinite(trained["balance"])
    assert trained["grad_norm"] > 0
    assert trained["lambda"] == 0.5
    assert trained["beta"] == config.loss.beta
    assert trained["hot"] + trained["warm"] + trained["cold"] == config.model.n_layers * config.model.E
    assert trained["hot"] == config.memory.hot_slots(config.model.E) * config.model.n_layers
    assert trained["cold_files"] == trained["cold"]
    assert trained["warm_copies"] == trained["warm"]
    assert _cold_file_is_finite(tmp_path / "train")

    evaluated = run_eval_pass(config, work_dir=tmp_path / "eval")
    assert evaluated["parameters"] == trained["parameters"]
    assert math.isfinite(evaluated["perplexity_random_init"])
    assert evaluated["aggregation"] == "sum"
    assert evaluated["perplexity_random_init"] > 1.0
    # fp32: o passo de treino devolveu tensores finitos e o eval também
    # Mesmo batch e mesma init: a soma dos top-2 (inferência) é estritamente
    # maior que a média ponderada (treino) enquanto c(ℓ) > 0.
    assert evaluated["c_mem"] > trained["c_mem"]
    assert math.isfinite(evaluated["c_mem"])


def _cold_file_is_finite(directory: Path) -> bool:
    files = list(directory.glob("*.f32"))
    assert files, "o home frio deveria ter gravado ao menos um expert"
    # Um arquivo basta; abrir os ~60 MB todos só atrasaria o CI.
    sample = np.memmap(files[0], dtype=np.float32, mode="r")
    return bool(np.isfinite(sample).all())
