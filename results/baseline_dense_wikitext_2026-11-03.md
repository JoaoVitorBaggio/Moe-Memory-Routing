# Baseline denso no WikiText-2 — sementes 1 e 2 (2026-11-03)

**Status:** loss de treino das sementes 1 e 2 copiada em 2026-11-03. A semente 0 já estava em [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md).

Fonte local (gitignored, fora deste commit): `outputs/baseline/wikitext/seed-1/done.json` e `outputs/baseline/wikitext/seed-2/done.json`, o layout de `experiments/run_baseline.py`. Os valores abaixo foram copiados desses arquivos. Este commit não reexecuta o treino. O `done.json` não traz host, device nem data. Esses campos não foram preenchidos por analogia com a grade MoE.

## O que treinou

LM denso (modelo 3 do protocolo): sem roteador, sem caminho de memória, sem `C_mem`. O MoE com λ = 0 é a outra corrida, em [lambda_grid_seeds_wikitext_2026-11-02.md](lambda_grid_seeds_wikitext_2026-11-02.md).

Uma época (`epochs_completed` 1) nas duas. `finite` verdadeiro. `seq_len` 256. Tokens: 2391808 nas duas, o mesmo total da semente 0. Parâmetros do denso: 16090880. `d_ff` 1024. `status` `done`. O arquivo não traz código de saída.

A semente 1 fechou depois de uma recuperação. A escrita do checkpoint tinha saído com código 5; a corrida retomada desceu o batch e o `done.json` que ficou registra batch 2 e 4522 passos. O código 5 não é um campo deste arquivo: é a tentativa que a recuperação substituiu. A semente 2 ficou no batch 4, 2336 passos, o mesmo comprimento de época da semente 0.

`last_grad_norm` no último passo: semente 1, 2.8969056606292725; semente 2, 1.5181970596313477.

## Loss de treino no último passo

No denso, `L_task` e `L_total` coincidem no último passo. Ponto decimal como no JSON. Isso é loss de treino, não perplexidade de `valid` ou `test`.

| semente | passos | batch | tokens | last_task | last_total | parâmetros | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4522 | 2 | 2391808 | 5.725013732910156 | 5.725013732910156 | 16090880 | done |
| 2 | 2336 | 4 | 2391808 | 5.2047600746154785 | 5.2047600746154785 | 16090880 | done |

A semente 0, na nota de 31/10, tem `last_task` = `last_total` = 5.889871597290039, 2336 passos, batch 4, saída 0.

## Ainda sem número

- PPL de hold-out (`valid`, `test`) do denso nas sementes 1 e 2. Parede, `swaps` e `bytes_moved` desse hold-out também. O hold-out medido em 2026-11-03 é só a grade MoE: [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). Na curva de `B`, o denso entra como `n/a`: [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). O hold-out do denso na semente 0 está em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md) (`test` 339.05; `swaps` e `bytes_moved` null).
