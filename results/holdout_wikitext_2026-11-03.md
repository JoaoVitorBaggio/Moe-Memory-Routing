# Hold-out no WikiText-2 — sementes 1 e 2 (2026-11-03)

**Status:** medido nas sementes 1 e 2, em 2026-11-03. Host: Fedora, GTX 1060 6GB, America/Sao_Paulo. `device` cuda. `wall_report_grade` verdadeiro nos seis checkpoints MoE. O LM denso dessas duas sementes não entrou nesta medição.

Fonte local (gitignored, fora deste commit): resumo JSON da corrida pedida (`scope` = `request`) de cada semente. Os checkpoints citados nele são `outputs/wikitext/seed-1/lambda-*/checkpoint.pt` e `outputs/wikitext/seed-2/lambda-*/checkpoint.pt`. Os valores abaixo foram copiados desse resumo. Este commit não reexecuta a avaliação.

A semente 0 continua em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md). O script de descoberta lê a semente 0; esta nota é a corrida pedida das outras duas.

## O que foi medido

Perplexidade de token (`exp` da task cross-entropy) em `valid` e `test`, no `B` de referência (8 vagas quentes). Parede p50/p95, `swaps` e `bytes_moved` são do mesmo forward. Isto não é loss de treino. A grade de λ dessas sementes já estava em [lambda_grid_seeds_wikitext_2026-11-02.md](lambda_grid_seeds_wikitext_2026-11-02.md). O checkpoint λ = 0 da semente 2 tem 2583 passos de treino; os outros cinco têm 2336. Tokens de treino: 2391808 nos seis.

PPL e ms com 2 casas. `swaps` e `bytes_moved` inteiros, como no JSON.

### Semente 1

| checkpoint | tipo | λ | split | PPL | p50 ms | p95 ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda-0 | moe | 0.0 | valid | 350.29 | 90.20 | 95.23 | 12693 | 17455253504 |
| lambda-0 | moe | 0.0 | test | 351.04 | 90.37 | 94.43 | 14124 | 19291430912 |
| lambda-0.5 | moe | 0.5 | valid | 333.81 | 48.73 | 53.73 | 4735 | 4991121408 |
| lambda-0.5 | moe | 0.5 | test | 334.21 | 46.78 | 52.44 | 5298 | 5586354176 |
| lambda-1 | moe | 1.0 | valid | 335.98 | 45.84 | 50.87 | 4734 | 4993224704 |
| lambda-1 | moe | 1.0 | test | 335.11 | 45.43 | 51.13 | 5196 | 5476982784 |

### Semente 2

| checkpoint | tipo | λ | split | PPL | p50 ms | p95 ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda-0 | moe | 0.0 | valid | 352.80 | 89.36 | 93.26 | 13750 | 18927560704 |
| lambda-0 | moe | 0.0 | test | 350.26 | 89.48 | 92.75 | 15854 | 21935273984 |
| lambda-0.5 | moe | 0.5 | valid | 329.47 | 60.46 | 67.84 | 8662 | 9119891456 |
| lambda-0.5 | moe | 0.5 | test | 328.68 | 59.63 | 66.39 | 9675 | 10184159232 |
| lambda-1 | moe | 1.0 | valid | 335.99 | 44.55 | 50.21 | 4845 | 5108905984 |
| lambda-1 | moe | 1.0 | test | 336.05 | 44.46 | 51.15 | 5694 | 6000703488 |

## Teste, sementes 0–2

PPL de `test` no `B` de referência, já em 2 casas. A semente 0 é a da nota de 2026-11-02 (351.88, 337.37, 330.18). A média é aritmética. O desvio é o amostral (divisor n−1, n = 3). O denso fica de fora: nas sementes 1 e 2 ele não foi medido, e na semente 0 a PPL de `test` é 339.05, outra corrida.

| λ | semente 0 | semente 1 | semente 2 | média | dp amostral |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 351.88 | 351.04 | 350.26 | 351.06 | 0.81 |
| 0.5 | 337.37 | 334.21 | 328.68 | 333.42 | 4.40 |
| 1.0 | 330.18 | 335.11 | 336.05 | 333.78 | 3.15 |

A curva 0,5× / 1× / 2× das sementes 1 e 2 está em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). 1×B é este mesmo `B`. A loss de treino do LM denso dessas sementes está em [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md). A da semente 0 continua em [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md).

## Ainda sem número

- PPL de hold-out do LM denso nas sementes 1 e 2. Esta corrida só tem a grade MoE. `swaps` e `bytes_moved` do denso continuam sem medição nessas sementes; na semente 0 o JSON traz null, não zero.
