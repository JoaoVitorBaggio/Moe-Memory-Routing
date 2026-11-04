# Proxy × emulador no WikiText-2 (2026-11-02)

**Status:** correlação e entropia copiadas em 2026-11-02. Host: Fedora, GTX 1060, America/Sao_Paulo. Não é perplexidade de hold-out e não retreina.

Fonte local (gitignored, fora deste commit): `outputs/proxy-emulator/summary.json`, escrito por `experiments/plot_proxy_emulator.py` a partir do `train.jsonl`. O CSV continua em `outputs/` e fica de fora do Git. As três figuras dessa corrida foram copiadas para esta pasta, sem alteração: [entropy.svg](entropy.svg), [c_mem_vs_bytes.svg](c_mem_vs_bytes.svg), [c_mem_vs_swaps.svg](c_mem_vs_swaps.svg). O script segue gravando em `outputs/proxy-emulator/`. Os números da tabela são os mesmos; a figura não muda a correlação.

## O que entrou

`c_mem` contra `bytes_moved` e contra `swaps`, no conjunto (`pooled`) e por λ. Pearson e Spearman com 3 casas. Entropia do roteador no primeiro e no último passo, também com 3 casas. O bloco `pooled` não traz entropia. O campo `note` de cada correlação é null.

O JSON não nomeia a semente. Cada λ tem 2336 passos; o conjunto soma 7008. A semente 2 com λ = 0.0, na loss de treino, tem 2583 passos, então este agregado não é essa corrida. Não dá para dizer, só com este resumo, se os 2336 passos são a semente 0 ou a semente 1.

| conjunto | n | Pearson bytes | Spearman bytes | Pearson swaps | Spearman swaps | H primeiro | H último |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 7008 | 0.970 | 0.803 | 0.934 | 0.794 | — | — |
| lambda-0 | 2336 | 0.760 | 0.792 | 0.557 | 0.578 | 3.458 | 3.029 |
| lambda-0.5 | 2336 | 0.136 | 0.316 | 0.087 | 0.313 | 3.458 | 2.362 |
| lambda-1 | 2336 | 0.159 | 0.294 | 0.110 | 0.291 | 3.458 | 2.244 |

A entropia inicial é a mesma nos três λ. A final cai de λ = 0.0 para λ = 1.0. O denso não entra: `c_mem` ausente.

## Figuras

As três SVG estão em `results/`, copiadas do `outputs/proxy-emulator/` local. O desenho abre no navegador.

- [entropy.svg](entropy.svg) — entropia do roteador por passo.
- [c_mem_vs_bytes.svg](c_mem_vs_bytes.svg) — `c_mem` contra `bytes_moved`.
- [c_mem_vs_swaps.svg](c_mem_vs_swaps.svg) — `c_mem` contra `swaps`.

## Ainda sem número neste repositório

- Um agregado que some as três sementes. O `n` acima não fecha com essa soma.
