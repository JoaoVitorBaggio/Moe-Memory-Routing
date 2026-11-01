# Baseline denso no WikiText-2 — semente 0 (2026-10-31)

**Status:** concluída para a semente 0, em 2026-10-31, por volta de 11:23 (America/Sao_Paulo). Host: Fedora, GTX 1060.

Fonte local (gitignored, fora deste commit): `outputs/baseline/wikitext/summary.json`. O valor abaixo foi copiado desse arquivo. Este commit não reexecuta o treino.

## O que treinou

LM denso (modelo 3 do protocolo): sem roteador, sem caminho de memória, sem `C_mem`. O MoE com λ = 0 é a corrida em [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md).

Uma época, 2336 passos, batch 4. Código de saída 0. Parâmetros do denso: 16090880.

O trecho copiado do `summary.json` para esta nota não inclui tokens, `seq_len` nem device. Esses campos não foram preenchidos por analogia com a grade λ.

## Loss de treino no último passo

No denso, `L_task` e `L_total` coincidem no último passo.

| | last_task | last_total | parâmetros | saída |
| --- | --- | --- | --- | --- |
| semente 0 | 5.889871597290039 | 5.889871597290039 | 16090880 | 0 |

Ponto decimal como no JSON. Isso é loss de treino. Perplexidade de hold-out não foi medida.

## Sem medição neste repositório

- Perplexidade de hold-out (`valid`, `test`).
- Sementes 1 e 2. O protocolo pede no mínimo 3.
- Dose–resposta de `B` (o denso entra como `n/a` nessa curva; a curva em si não foi medida).
- Latência de parede (p50/p95), `swaps` e `bytes_moved` de relatório.
- Modelo 2 (LRU) como linha medida.
- Correlação proxy×emulador e curva de entropia do roteador. O denso não entra nessa correlação (`c_mem` ausente).
