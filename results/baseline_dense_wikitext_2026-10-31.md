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

Ponto decimal como no JSON. Isso é loss de treino.

## Medido depois, na semente 0 (2026-11-02)

- Perplexidade de hold-out (`valid`, `test`) e parede p50/p95: [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md). `swaps` e `bytes_moved` do denso são null.
- Dose–resposta de `B`: o denso entra como `n/a`. A curva dos MoE está em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md).
- Modelo 2 (LRU) como linha medida, cópia do λ = 0: [model_comparison_wikitext_2026-11-02.md](model_comparison_wikitext_2026-11-02.md).

## Medido depois

- PPL de hold-out do denso nas sementes 1 e 2 (`test` 318.16 e 327.09): seção LM denso de [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). A loss de treino dessas sementes está em [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md). O protocolo pede no mínimo 3; a loss e o hold-out das três sementes do denso estão nas notas.
- Correlação proxy×emulador. O denso fica de fora (`c_mem` ausente). O resumo do MoE e as figuras estão em [proxy_emulator_wikitext_2026-11-02.md](proxy_emulator_wikitext_2026-11-02.md).
