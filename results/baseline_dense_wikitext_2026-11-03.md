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

## Medido depois, no mesmo dia (~22:47–22:49)

Hold-out do denso em `valid` e `test`, na mesma máquina (Fedora, GTX 1060 6GB, `device` cuda, `wall_report_grade` verdadeiro). PPL em 2 casas: semente 1, `valid` 317.19 e `test` 318.16; semente 2, `valid` 327.86 e `test` 327.09. `swaps` e `bytes_moved` são null. A dose continua `n/a`. A parede p50/p95 e o batch do forward estão na seção LM denso de [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). A semente 1 mede o forward em batch 2; a semente 2, em batch 4. Esses ms não se comparam. A semente 0 continua em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md) (`test` 339.05; `swaps` e `bytes_moved` null). Na curva de `B`, o denso segue `n/a`: [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md).
