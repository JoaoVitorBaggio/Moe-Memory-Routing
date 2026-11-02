# Grade λ no WikiText-2 — sementes 1 e 2 (2026-11-02)

**Status:** loss de treino das sementes 1 e 2 copiada em 2026-11-02. Host: Fedora, GTX 1060, America/Sao_Paulo. A semente 0 já estava em [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md). O JSON desta noite repete os mesmos `last_task` / `last_total` da semente 0; as três sementes ficam na tabela para leitura junta.

Fonte local (gitignored, fora deste commit): `done.json` da grade (`outputs/wikitext/seed-<n>/lambda-*/`). Este commit não reexecuta o treino.

## Corrida

Uma época do split `train` do WikiText-2. Batch 4. `finite` verdadeiro nas nove linhas. Tokens do treino: 2391808 em todas. Código de saída não veio neste resumo; o que entrou é o bloco `done` (`steps`, `epochs_completed`, `last_task`, `last_total`).

A semente 2 com λ = 0.0 tem 2583 passos. As outras oito linhas têm 2336. Épocas concluídas: 1 nas nove. Esta nota não explica a diferença de passos.

## Loss de treino no último passo

`last_task` é `L_task`. `last_total` é `L_total`. As duas saem do último passo de treino, não de uma média da época e não da perplexidade de `valid` ou `test`. Ponto decimal como no JSON.

| semente | λ | passos | last_task | last_total | tokens |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 2336 | 5.938449859619141 | 5.950239658355713 | 2391808 |
| 0 | 0.5 | 2336 | 5.904059886932373 | 6.441991806030273 | 2391808 |
| 0 | 1.0 | 2336 | 5.87575101852417 | 6.9235758781433105 | 2391808 |
| 1 | 0.0 | 2336 | 5.641011714935303 | 5.652388572692871 | 2391808 |
| 1 | 0.5 | 2336 | 5.599190711975098 | 6.13767147064209 | 2391808 |
| 1 | 1.0 | 2336 | 5.595114231109619 | 6.633786201477051 | 2391808 |
| 2 | 0.0 | 2583 | 5.332046031951904 | 5.342931747436523 | 2391808 |
| 2 | 0.5 | 2336 | 5.250917434692383 | 5.795291900634766 | 2391808 |
| 2 | 1.0 | 2336 | 5.242007732391357 | 6.277893543243408 | 2391808 |

Com λ = 0.0, `last_total` e `last_task` não coincidem: β continua na loss. Esta nota não separa `C_mem` e o termo de balanço.

## Ainda sem número

- Hold-out (`valid`, `test`), parede, `swaps` e `bytes_moved` das sementes 1 e 2.
- Loss de treino do LM denso nas sementes 1 e 2. Este JSON só traz a grade MoE. O denso da semente 0 segue em [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md).
