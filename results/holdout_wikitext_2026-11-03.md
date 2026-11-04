# Hold-out no WikiText-2 — sementes 1 e 2 (2026-11-03)

**Status:** medido nas sementes 1 e 2, em 2026-11-03. Host: Fedora, GTX 1060 6GB, America/Sao_Paulo. `device` cuda. `wall_report_grade` verdadeiro nos seis checkpoints MoE e nos dois do LM denso. A grade MoE é a corrida descrita abaixo. O denso é outra corrida, no fim do dia (~22:47–22:49), na seção [LM denso](#lm-denso).

Fonte local (gitignored, fora deste commit): resumo JSON da corrida pedida (`scope` = `request`) de cada semente. Os checkpoints citados nele são `outputs/wikitext/seed-1/lambda-*/checkpoint.pt` e `outputs/wikitext/seed-2/lambda-*/checkpoint.pt`. Os valores abaixo foram copiados desse resumo. Este commit não reexecuta a avaliação.

A semente 0 continua em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md). O script de descoberta lê a semente 0. A grade MoE abaixo é a corrida pedida das outras duas. O LM denso está na seção própria.

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

PPL de `test` no `B` de referência, já em 2 casas. A semente 0 é a da nota de 2026-11-02 (351.88, 337.37, 330.18). A média é aritmética. O desvio é o amostral (divisor n−1, n = 3). O denso fica de fora desta média: é outra corrida. A PPL de `test` dele está na seção [LM denso](#lm-denso). Na semente 0, a da nota de 02/11 é 339.05.

| λ | semente 0 | semente 1 | semente 2 | média | dp amostral |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 351.88 | 351.04 | 350.26 | 351.06 | 0.81 |
| 0.5 | 337.37 | 334.21 | 328.68 | 333.42 | 4.40 |
| 1.0 | 330.18 | 335.11 | 336.05 | 333.78 | 3.15 |

A curva 0,5× / 1× / 2× das sementes 1 e 2 está em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). 1×B é este mesmo `B`. A loss de treino do LM denso dessas sementes está em [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md). A da semente 0 continua em [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md). O hold-out do denso está na seção seguinte.

## LM denso

**Status:** medido nas sementes 1 e 2, em 2026-11-03, por volta de 22:47–22:49 (America/Sao_Paulo). Mesma máquina: Fedora, GTX 1060 6GB. `device` cuda. `wall_report_grade` verdadeiro nos dois checkpoints. Esta corrida não é a grade MoE acima.

Fonte local (gitignored, fora deste commit): `outputs/baseline/wikitext/seed-1/holdout.json` e `outputs/baseline/wikitext/seed-2/holdout.json`, ao lado de cada checkpoint denso. Os valores abaixo foram copiados desses arquivos. O `--summary` da mesma corrida (`scope` request) repete PPL, parede, `swaps`, `bytes_moved` e a dose. Este commit não reexecuta a avaliação.

Perplexidade de token (`exp` da task cross-entropy) em `valid` e `test`. O LM denso não tem teto quente: `swaps` e `bytes_moved` ficam null, não zero. O bloco `dose_response` traz `status` `n/a` e `budgets` null. O motivo no JSON: não há residência nem eixo de orçamento `B`; `set_hot_budget` não se aplica. Isto não é loss de treino.

PPL e ms com 2 casas. `bits_per_byte` com 6 casas. O JSON traz o float completo; o corte não junta células vizinhas. `swaps` e `bytes_moved` como no JSON (null).

A semente 1 é o checkpoint da recuperação: 4522 passos, `checkpoint_batch_size` 2. O forward do hold-out usa esse batch (`wall_forward_batch` 2). A semente 2 tem 2336 passos e batch 4, e o forward também é 4. As duas com 2391808 tokens de treino, `seq_len` 256, `context_matches_checkpoint` verdadeiro e 16090880 parâmetros. `protocol_complete` verdadeiro, `interrupted` falso. `wall_samples` é o número de batches lógicos do split: semente 1, 483 em `valid` e 553 em `test`; semente 2, 242 em `valid` e 277 em `test`.

A parede da semente 1 fica de fora da comparação com as outras duas. O forward usa o batch do checkpoint, 2 contra 4. A semente 0 também é batch 4: o checkpoint na nota de 31/10, e `wall_forward_batch` 4 na comparação dos braços. O p50 de `test` dela, na nota de 02/11, é 12.16. A semente 2, batch 4, tem p50 de `test` 12.55. A semente 1, batch 2, tem 6.69.

`wall_clock` cuda. `wall_gpu` NVIDIA GeForce GTX 1060 6GB.

| semente | split | PPL | p50 ms | p95 ms | wall_forward_batch | swaps | bytes_moved | bits_per_byte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | valid | 317.19 | 6.67 | 7.28 | 2 | null | null | 1.790441 |
| 1 | test | 318.16 | 6.69 | 6.93 | 2 | null | null | 1.823776 |
| 2 | valid | 327.86 | 12.54 | 13.19 | 4 | null | null | 1.800730 |
| 2 | test | 327.09 | 12.55 | 12.62 | 4 | null | null | 1.832529 |

### Teste, sementes 0–2

PPL de `test` já em 2 casas. A semente 0 é a da nota de 2026-11-02 (339.05). As três PPL ficam lado a lado, sem média e sem desvio amostral. Na grade MoE acima, as três sementes entram no mesmo batch de treino e a média usa o desvio amostral (divisor n−1, n = 3). Aqui a semente 1 é a recuperação (batch 2, 4522 passos). As sementes 0 e 2 fecharam em batch 4 e 2336 passos. Uma média misturaria esses regimes.

| semente | PPL de test | passos | batch do checkpoint | wall_forward_batch |
| --- | --- | --- | --- | --- |
| 0 | 339.05 | 2336 | 4 | 4 |
| 1 | 318.16 | 4522 | 2 | 2 |
| 2 | 327.09 | 2336 | 4 | 4 |

A curva de `B` continua `n/a` em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). Esta medição não preenche esse eixo.
