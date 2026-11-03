# Dose–resposta de B no WikiText-2 — semente 0 (2026-11-02)

**Status:** medida na semente 0, em 2026-11-02, sem retreino. Host: Fedora, GTX 1060, America/Sao_Paulo. Parede de relatório: `wall_report_grade` (GTX 1060).

Fonte local (gitignored, fora deste commit): `outputs/holdout/dose_response.json`. O `dose_response.md` irmão continua só com perplexidade; parede, `swaps` e `bytes_moved` desta nota saem do JSON. PPL e ms em 2 casas. O JSON traz o float completo; o arredondamento não junta células vizinhas. `swaps` e `bytes_moved` são os inteiros do split. Este commit não reexecuta a curva.

## Como a curva foi feita

Perplexidade de hold-out (exp da task cross-entropy) no checkpoint já treinado. Cada split recomeça o LRU no `B` do YAML (reset) e o multiplicador entra por `set_hot_budget`, que reatribui pelo LRU fresco e não apaga o carimbo de uso. Não há reset entre `set_hot_budget` e o forward. O bloco `splits` do hold-out continua sendo a medição de referência, nesse mesmo `B`, feita antes da curva. 1×B, quando está em `eval.B_mult`, repete esse `B` por `set_hot_budget`. O LM denso não tem teto quente: `n/a` em todo o eixo de `B`. Cada split da curva é o `score_split` do hold-out, então o JSON traz `swaps`, `bytes_moved` e a parede desse forward (`wall_p50_ms`, `wall_p95_ms`). Em CPU os ms ficam null. O teto não tem outro caminho: `prepare_split_budget`, sem atribuir `hot_slots` à parte.

`B` de referência (YAML): 8 vagas quentes. 0,5×B = 4 vagas, 1×B = 8, 2×B = 16. Checkpoint denso: `n/a`. Checkpoint ausente: ausente. Nesta corrida os quatro checkpoints da semente 0 estavam presentes.

| checkpoint | tipo | λ | split | B_mult | vagas | PPL | wall_p50_ms | wall_p95_ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | dense | — | valid | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| baseline | dense | — | test | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lambda-0 | moe | 0.0 | valid | 0.5 | 4 | 349.27 | 87.66 | 89.53 | 9651 | 11717462016 |
| lambda-0 | moe | 0.0 | test | 0.5 | 4 | 351.89 | 88.00 | 89.99 | 10662 | 12931063808 |
| lambda-0 | moe | 0.0 | valid | 1.0 | 8 | 349.22 | 82.89 | 84.31 | 12591 | 16561352704 |
| lambda-0 | moe | 0.0 | test | 1.0 | 8 | 351.88 | 82.69 | 84.96 | 13890 | 18180890624 |
| lambda-0 | moe | 0.0 | valid | 2.0 | 16 | 349.19 | 72.41 | 74.10 | 15931 | 23104706560 |
| lambda-0 | moe | 0.0 | test | 2.0 | 16 | 351.87 | 71.80 | 73.55 | 17441 | 24934574080 |
| lambda-0.5 | moe | 0.5 | valid | 0.5 | 4 | 334.15 | 58.79 | 67.19 | 4744 | 4997431296 |
| lambda-0.5 | moe | 0.5 | test | 0.5 | 4 | 337.55 | 55.19 | 59.81 | 5545 | 5838749696 |
| lambda-0.5 | moe | 0.5 | valid | 1.0 | 8 | 333.99 | 51.82 | 56.31 | 5749 | 6057492480 |
| lambda-0.5 | moe | 0.5 | test | 1.0 | 8 | 337.37 | 50.93 | 55.42 | 6510 | 6860951552 |
| lambda-0.5 | moe | 0.5 | valid | 2.0 | 16 | 334.01 | 48.34 | 53.02 | 7744 | 9477451776 |
| lambda-0.5 | moe | 0.5 | test | 2.0 | 16 | 337.41 | 47.93 | 50.87 | 8707 | 10619541504 |
| lambda-1 | moe | 1.0 | valid | 0.5 | 4 | 326.88 | 53.63 | 57.73 | 4971 | 5230897152 |
| lambda-1 | moe | 1.0 | test | 0.5 | 4 | 330.46 | 53.36 | 57.61 | 5671 | 5969154048 |
| lambda-1 | moe | 1.0 | valid | 1.0 | 8 | 326.59 | 46.57 | 50.93 | 6042 | 6366676992 |
| lambda-1 | moe | 1.0 | test | 1.0 | 8 | 330.18 | 44.24 | 48.79 | 6370 | 6711617536 |
| lambda-1 | moe | 1.0 | valid | 2.0 | 16 | 326.61 | 41.49 | 44.29 | 5061 | 6082732032 |
| lambda-1 | moe | 1.0 | test | 2.0 | 16 | 330.20 | 40.97 | 44.03 | 5370 | 6440292352 |

## Leitura desta corrida

A PPL fica quase plana de 0,5×B a 2×B, em `valid` e em `test`, nos três λ. `wall_p50_ms` cai conforme `B` cresce (mais vagas quentes, forward mais rápido na GTX 1060), nos três checkpoints MoE e nos dois splits. `wall_p95_ms` acompanha essa queda nesta corrida.

`swaps` e `bytes_moved` sobem com `B` em λ = 0 e em λ = 0.5, nos dois splits. Em λ = 1, 2×B tem menos `swaps` do que 1×B: 5061 contra 6042 em `valid`, 5370 contra 6370 em `test`. Os bytes em λ = 1 também ficam abaixo de 1×B nesse 2×B (6082732032 contra 6366676992 em `valid`; 6440292352 contra 6711617536 em `test`) e acima de 0,5×B. São os números desta corrida. A nota registra a contagem e não atribui um mecanismo à queda em λ = 1.

A mesma curva nas sementes 1 e 2 está em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md).

A PPL em 1×B, com 2 casas, coincide com o hold-out de referência em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md). A parede em 1×B desta curva pode diferir um pouco daquela tabela: a dose mede de novo o mesmo teto por `set_hot_budget`, e o hold-out é o forward de referência feito antes da curva. Nesta corrida o `wall_p50_ms` de 1×B fica, dose contra hold-out: λ = 0, 82.89 / 82.69 contra 82.31 / 82.43; λ = 0.5, 51.82 / 50.93 contra 51.72 / 51.26; λ = 1, 46.57 / 44.24 contra 46.79 / 45.20 (`valid` / `test`). `swaps` e `bytes_moved` em 1×B coincidem com os inteiros do hold-out. No `comparison.json` as células MoE de 0,5× e 2× estão `filled` (ver [model_comparison_wikitext_2026-11-02.md](model_comparison_wikitext_2026-11-02.md)).
