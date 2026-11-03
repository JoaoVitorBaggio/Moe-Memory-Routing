# Hold-out no WikiText-2 — semente 0 (2026-11-02)

**Status:** medido na semente 0, em 2026-11-02. Host: Fedora, GTX 1060, America/Sao_Paulo. `device` cuda. `wall_report_grade` verdadeiro nos quatro checkpoints.

Fonte local (gitignored, fora deste commit): `holdout.json` ao lado de cada checkpoint da semente 0 e `outputs/holdout/summary.json`. Os valores abaixo foram copiados desse resumo. Este commit não reexecuta a avaliação.

## O que foi medido

Perplexidade de token (`exp` da task cross-entropy) em `valid` e `test`, no `B` de referência (8 vagas quentes). Parede p50/p95, `swaps` e `bytes_moved` são do mesmo forward. O denso não tem teto quente: `swaps` e `bytes_moved` ficam null. Isto não é loss de treino.

PPL e ms com 2 casas. `swaps` e `bytes_moved` inteiros, como no JSON.

| checkpoint | tipo | λ | split | PPL | p50 ms | p95 ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | dense | — | valid | 333.48 | 12.16 | 12.83 | null | null |
| baseline | dense | — | test | 339.05 | 12.16 | 12.21 | null | null |
| lambda-0 | moe | 0.0 | valid | 349.22 | 82.31 | 83.91 | 12591 | 16561352704 |
| lambda-0 | moe | 0.0 | test | 351.88 | 82.43 | 83.91 | 13890 | 18180890624 |
| lambda-0.5 | moe | 0.5 | valid | 333.99 | 51.72 | 56.37 | 5749 | 6057492480 |
| lambda-0.5 | moe | 0.5 | test | 337.37 | 51.26 | 54.99 | 6510 | 6860951552 |
| lambda-1 | moe | 1.0 | valid | 326.59 | 46.79 | 50.23 | 6042 | 6366676992 |
| lambda-1 | moe | 1.0 | test | 330.18 | 45.20 | 49.14 | 6370 | 6711617536 |

A curva 0,5× / 1× / 2× está em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md). 1×B é este mesmo `B`. A loss de treino da semente 0 continua em [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md) e [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md).

## Ainda sem número

- PPL de hold-out do LM denso nas sementes 1 e 2. O hold-out MoE dessas sementes está em [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). O script de descoberta continua lendo a semente 0.
- `swaps` e `bytes_moved` do denso. O JSON traz null, não zero.
