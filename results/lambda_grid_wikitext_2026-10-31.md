# Grade λ no WikiText-2 — semente 0 (2026-10-31)

**Status:** concluída para a semente 0, em 2026-10-31. Host: Fedora, GTX 1060.

Fonte local (gitignored, fora deste commit): `outputs/wikitext/summary.json` e `outputs/wikitext/lambda-*/done.json`. Os valores abaixo foram copiados desses arquivos. Este commit não reexecuta o treino.

## Corrida

Uma época do split `train` do WikiText-2. 2336 passos, batch 4, `seq_len` 256, β = 0.01, fp32, CUDA. Código de saída 0 em λ = 0.0, 0.5 e 1.0. Tokens do treino: 2391808 nos três λ.

O protocolo pede no mínimo 3 sementes (`eval.seeds`). Aqui só a semente 0. As sementes 1 e 2 não têm corrida registrada.

## Loss de treino no último passo

`last_task` é `L_task`. `last_total` é `L_total`. As duas saem do último passo de treino, não de uma média da época e não da perplexidade de `valid` ou `test`.

| λ | last_task | last_total | tokens | saída |
| --- | --- | --- | --- | --- |
| 0.0 | 5.938449859619141 | 5.950239658355713 | 2391808 | 0 |
| 0.5 | 5.904059886932373 | 6.441991806030273 | 2391808 | 0 |
| 1.0 | 5.87575101852417 | 6.9235758781433105 | 2391808 | 0 |

Ponto decimal como no JSON. Com λ = 0.0, `last_total` e `last_task` não coincidem: β continua na loss. Esta nota não separa `C_mem` e o termo de balanço.

## Sem medição neste repositório

- Perplexidade de hold-out (`valid`, `test`).
- Sementes 1 e 2.
- Dose–resposta de `B` (0,5× / 1× / 2×).
- Latência de parede (p50/p95), `swaps` e `bytes_moved` de relatório.
- Modelo 2 (LRU) como linha própria. No código ele reusa o checkpoint λ = 0; a tabela de comparação não foi preenchida.
- Correlação proxy×emulador e curva de entropia do roteador.

O LM denso da mesma data está em [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md). Ele fica fora desta tabela: é outra corrida, sem roteador.
