# Dose–resposta de B no WikiText-2 — sementes 1 e 2 (2026-11-03)

**Status:** medida nas sementes 1 e 2, em 2026-11-03, sem retreino. Host: Fedora, GTX 1060 6GB, America/Sao_Paulo. Parede de relatório: `wall_report_grade` (GTX 1060).

Fonte local (gitignored, fora deste commit): `dose_response.json` da mesma corrida pedida do hold-out de cada semente. O `dose_response.md` irmão continua só com perplexidade (4 casas); parede, `swaps` e `bytes_moved` desta nota saem do JSON. PPL e ms em 2 casas. O JSON traz o float completo; o arredondamento não junta células vizinhas. `swaps` e `bytes_moved` são os inteiros do split. Este commit não reexecuta a curva.

A semente 0 está em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md).

## Como a curva foi feita

Perplexidade de hold-out (exp da task cross-entropy) no checkpoint já treinado. Cada split recomeça o LRU no `B` do YAML (reset) e o multiplicador entra por `set_hot_budget`, que reatribui pelo LRU fresco e não apaga o carimbo de uso. Não há reset entre `set_hot_budget` e o forward. O bloco `splits` do hold-out continua sendo a medição de referência, nesse mesmo `B`, feita antes da curva. 1×B, quando está em `eval.B_mult`, repete esse `B` por `set_hot_budget`. O LM denso não tem teto quente: `n/a` em todo o eixo de `B`. Nesta corrida o JSON só lista os três MoE de cada semente. A linha `n/a` não é um forward desta curva: o denso não entra em `set_hot_budget`. O hold-out do denso nessas sementes foi medido à parte e o `dose_response.status` continua `n/a`: seção LM denso de [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). Cada split da curva é o `score_split` do hold-out, então o JSON traz `swaps`, `bytes_moved` e a parede desse forward (`wall_p50_ms`, `wall_p95_ms`). Em CPU os ms ficam null. O teto não tem outro caminho: `prepare_split_budget`, sem atribuir `hot_slots` à parte.

`B` de referência (YAML): 8 vagas quentes. 0,5×B = 4 vagas, 1×B = 8, 2×B = 16.

### Semente 1

| checkpoint | tipo | λ | split | B_mult | vagas | PPL | wall_p50_ms | wall_p95_ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | dense | — | valid | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| baseline | dense | — | test | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lambda-0 | moe | 0.0 | valid | 0.5 | 4 | 350.35 | 96.17 | 100.55 | 9960 | 12712321024 |
| lambda-0 | moe | 0.0 | test | 0.5 | 4 | 351.01 | 96.30 | 99.79 | 11187 | 14140459008 |
| lambda-0 | moe | 0.0 | valid | 1.0 | 8 | 350.29 | 90.50 | 94.47 | 12693 | 17455253504 |
| lambda-0 | moe | 0.0 | test | 1.0 | 8 | 351.04 | 90.69 | 94.01 | 14124 | 19291430912 |
| lambda-0 | moe | 0.0 | valid | 2.0 | 16 | 350.14 | 79.26 | 83.04 | 14442 | 20189538304 |
| lambda-0 | moe | 0.0 | test | 2.0 | 16 | 350.79 | 79.26 | 83.01 | 16027 | 22318073856 |
| lambda-0.5 | moe | 0.5 | valid | 0.5 | 4 | 333.68 | 57.36 | 63.81 | 3930 | 4143493120 |
| lambda-0.5 | moe | 0.5 | test | 0.5 | 4 | 334.19 | 56.51 | 62.85 | 4485 | 4726106112 |
| lambda-0.5 | moe | 0.5 | valid | 1.0 | 8 | 333.81 | 48.24 | 52.56 | 4735 | 4991121408 |
| lambda-0.5 | moe | 0.5 | test | 1.0 | 8 | 334.21 | 46.91 | 51.91 | 5298 | 5586354176 |
| lambda-0.5 | moe | 0.5 | valid | 2.0 | 16 | 333.44 | 50.07 | 54.42 | 6863 | 8331155456 |
| lambda-0.5 | moe | 0.5 | test | 2.0 | 16 | 333.85 | 49.34 | 54.50 | 7903 | 9561583616 |
| lambda-1 | moe | 1.0 | valid | 0.5 | 4 | 336.15 | 60.19 | 66.93 | 3756 | 3956299776 |
| lambda-1 | moe | 1.0 | test | 0.5 | 4 | 335.17 | 60.10 | 66.32 | 4255 | 4482123776 |
| lambda-1 | moe | 1.0 | valid | 1.0 | 8 | 335.98 | 46.14 | 51.37 | 4734 | 4993224704 |
| lambda-1 | moe | 1.0 | test | 1.0 | 8 | 335.11 | 44.76 | 50.60 | 5196 | 5476982784 |
| lambda-1 | moe | 1.0 | valid | 2.0 | 16 | 336.03 | 42.68 | 47.28 | 3436 | 3910027264 |
| lambda-1 | moe | 1.0 | test | 2.0 | 16 | 335.13 | 42.22 | 46.53 | 3763 | 4294930432 |

### Semente 2

| checkpoint | tipo | λ | split | B_mult | vagas | PPL | wall_p50_ms | wall_p95_ms | swaps | bytes_moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | dense | — | valid | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| baseline | dense | — | test | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lambda-0 | moe | 0.0 | valid | 0.5 | 4 | 352.87 | 95.04 | 98.90 | 11058 | 13814448128 |
| lambda-0 | moe | 0.0 | test | 0.5 | 4 | 350.22 | 95.23 | 98.75 | 12611 | 15753687040 |
| lambda-0 | moe | 0.0 | valid | 1.0 | 8 | 352.80 | 89.48 | 93.17 | 13750 | 18927560704 |
| lambda-0 | moe | 0.0 | test | 1.0 | 8 | 350.26 | 89.17 | 92.94 | 15854 | 21935273984 |
| lambda-0 | moe | 0.0 | valid | 2.0 | 16 | 352.75 | 77.78 | 81.59 | 15809 | 23641047040 |
| lambda-0 | moe | 0.0 | test | 2.0 | 16 | 350.12 | 77.29 | 81.30 | 18166 | 27265026048 |
| lambda-0.5 | moe | 0.5 | valid | 0.5 | 4 | 329.55 | 60.92 | 67.34 | 7292 | 7674927104 |
| lambda-0.5 | moe | 0.5 | test | 0.5 | 4 | 328.84 | 60.50 | 67.51 | 7833 | 8244920320 |
| lambda-0.5 | moe | 0.5 | valid | 1.0 | 8 | 329.47 | 60.72 | 68.64 | 8662 | 9119891456 |
| lambda-0.5 | moe | 0.5 | test | 1.0 | 8 | 328.68 | 59.28 | 66.17 | 9675 | 10184159232 |
| lambda-0.5 | moe | 0.5 | valid | 2.0 | 16 | 329.47 | 55.80 | 61.28 | 10526 | 13631461376 |
| lambda-0.5 | moe | 0.5 | test | 2.0 | 16 | 328.69 | 53.58 | 60.12 | 11712 | 15074322432 |
| lambda-1 | moe | 1.0 | valid | 0.5 | 4 | 336.11 | 65.82 | 73.25 | 4124 | 4341202944 |
| lambda-1 | moe | 1.0 | test | 0.5 | 4 | 336.20 | 61.51 | 72.47 | 4581 | 4822857728 |
| lambda-1 | moe | 1.0 | valid | 1.0 | 8 | 335.99 | 43.90 | 49.62 | 4845 | 5108905984 |
| lambda-1 | moe | 1.0 | test | 1.0 | 8 | 336.05 | 44.56 | 50.75 | 5694 | 6000703488 |
| lambda-1 | moe | 1.0 | valid | 2.0 | 16 | 336.07 | 48.06 | 52.47 | 4598 | 5098389504 |
| lambda-1 | moe | 1.0 | test | 2.0 | 16 | 336.10 | 47.93 | 52.77 | 5227 | 5834543104 |

## Leitura desta corrida

A PPL fica quase plana de 0,5×B a 2×B, em `valid` e em `test`, nos três λ, nas duas sementes. É o mesmo achado da semente 0. A maior folga nesta tabela, em 2 casas, é 0,37 na semente 1, λ = 0.5, `valid` (333.81 em 1×B e 333.44 em 2×B). Na semente 2, λ = 0.5, `valid`, 1×B e 2×B caem os dois em 329.47.

`wall_p50_ms` cai de 0,5×B a 2×B em λ = 0, nas duas sementes e nos dois splits. `wall_p95_ms` acompanha essa queda nesses quatro pares. Não é o caso em todo checkpoint. Na semente 1, λ = 0.5, p50 e p95 caem de 0,5× para 1× e sobem de 1× para 2×: p50 `valid` 57.36 / 48.24 / 50.07, `test` 56.51 / 46.91 / 49.34; p95 `valid` 63.81 / 52.56 / 54.42, `test` 62.85 / 51.91 / 54.50. Na semente 1, λ = 1, p50 e p95 caem nos dois splits. Na semente 2, λ = 0.5, o p50 cai nos dois splits; o p95 de `valid` sobe de 67.34 em 0,5×B para 68.64 em 1×B e cai para 61.28 em 2×B; o p95 de `test` cai (67.51 / 66.17 / 60.12). Na semente 2, λ = 1, p50 e p95 caem de 0,5× para 1× e sobem de 1× para 2×: p50 `valid` 65.82 / 43.90 / 48.06, `test` 61.51 / 44.56 / 47.93; p95 `valid` 73.25 / 49.62 / 52.47, `test` 72.47 / 50.75 / 52.77. A nota registra a parede e não atribui um mecanismo à subida.

`swaps` e `bytes_moved` sobem com `B` em λ = 0 e em λ = 0.5, nas duas sementes e nos dois splits. Em λ = 1 a contagem não sobe até 2×B. Na semente 1, 2×B tem menos `swaps` do que 0,5×B e do que 1×B: 3436 contra 3756 e 4734 em `valid`, 3763 contra 4255 e 5196 em `test`. Os bytes em 2×B também ficam abaixo de 0,5×B (3910027264 contra 3956299776 em `valid`; 4294930432 contra 4482123776 em `test`). Na semente 2, λ = 1, 2×B fica entre 0,5×B e 1×B em `swaps` (4598 entre 4124 e 4845 em `valid`; 5227 entre 4581 e 5694 em `test`) e os bytes de 2×B ficam logo abaixo de 1×B (5098389504 contra 5108905984 em `valid`; 5834543104 contra 6000703488 em `test`), acima de 0,5×B. São os números desta corrida.

A PPL em 1×B, com 2 casas, coincide com o hold-out de referência em [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). A parede em 1×B desta curva pode diferir um pouco daquela tabela: a dose mede de novo o mesmo teto por `set_hot_budget`, e o hold-out é o forward de referência feito antes da curva. Nesta corrida o `wall_p50_ms` de 1×B fica, dose contra hold-out (`valid` / `test`):

- semente 1, λ = 0: 90.50 / 90.69 contra 90.20 / 90.37
- semente 1, λ = 0.5: 48.24 / 46.91 contra 48.73 / 46.78
- semente 1, λ = 1: 46.14 / 44.76 contra 45.84 / 45.43
- semente 2, λ = 0: 89.48 / 89.17 contra 89.36 / 89.48
- semente 2, λ = 0.5: 60.72 / 59.28 contra 60.46 / 59.63
- semente 2, λ = 1: 43.90 / 44.56 contra 44.55 / 44.46

`swaps` e `bytes_moved` em 1×B coincidem com os inteiros do hold-out.
