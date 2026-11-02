# Dose–resposta de B no WikiText-2 — semente 0 (2026-11-02)

**Status:** medida na semente 0, em 2026-11-02, sem retreino. Host: Fedora, GTX 1060, America/Sao_Paulo. Parede de relatório: `wall_report_grade` (GTX 1060).

Fonte local (gitignored, fora deste commit): `outputs/holdout/dose_response.md` e `dose_response.json`. A tabela abaixo é a desse markdown, com a PPL em 2 casas. O arquivo de origem traz 4 casas; o arredondamento não junta células vizinhas. Este commit não reexecuta a curva.

## Como a curva foi feita

Perplexidade de hold-out (exp da task cross-entropy) no checkpoint já treinado. Cada split recomeça o LRU no `B` do YAML (reset) e o multiplicador entra por `set_hot_budget`, que reatribui pelo LRU fresco e não apaga o carimbo de uso. Não há reset entre `set_hot_budget` e o forward. O bloco `splits` do hold-out continua sendo a medição de referência, nesse mesmo `B`, feita antes da curva. 1×B, quando está em `eval.B_mult`, repete esse `B` por `set_hot_budget`. O LM denso não tem teto quente: `n/a`. Cada split da curva é o `score_split` do hold-out, então o JSON traz `swaps`, `bytes_moved` e a parede desse forward. Em CPU os ms ficam null. O markdown de origem é só perplexidade. O teto não tem outro caminho: `prepare_split_budget`, sem atribuir `hot_slots` à parte.

`B` de referência (YAML): 8 vagas quentes. Checkpoint denso: `n/a`. Checkpoint ausente: ausente. Nesta corrida os quatro checkpoints da semente 0 estavam presentes.

| checkpoint | tipo | λ | split | 0.5×B (4) | 1×B (8) | 2×B (16) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | dense | — | valid | n/a | n/a | n/a |
| baseline | dense | — | test | n/a | n/a | n/a |
| lambda-0 | moe | 0 | valid | 349.27 | 349.22 | 349.19 |
| lambda-0 | moe | 0 | test | 351.89 | 351.88 | 351.87 |
| lambda-0.5 | moe | 0.5 | valid | 334.15 | 333.99 | 334.01 |
| lambda-0.5 | moe | 0.5 | test | 337.55 | 337.37 | 337.41 |
| lambda-1 | moe | 1 | valid | 326.88 | 326.59 | 326.61 |
| lambda-1 | moe | 1 | test | 330.46 | 330.18 | 330.20 |

1×B, com 2 casas, coincide com a PPL do hold-out de referência em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md). Parede, `swaps` e `bytes_moved` de 0,5× e 2× não entram nesta tabela; no `comparison.json` essas células estão `filled` (ver [model_comparison_wikitext_2026-11-02.md](model_comparison_wikitext_2026-11-02.md)).
