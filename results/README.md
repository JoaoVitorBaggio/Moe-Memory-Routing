# results/

Resumos pequenos que entram no Git, mais as três figuras SVG do proxy. O bruto fica em `outputs/` (gitignored): checkpoint, `train.jsonl`, CSV e os JSON de medição. O script do proxy ainda grava SVG em `outputs/proxy-emulator/`; a cópia publicada está nesta pasta. Em 2026-10-31 entrou a loss de treino da semente 0 no WikiText-2, copiada da Fedora com GTX 1060. Em 2026-11-02 (America/Sao_Paulo, a mesma máquina) entrou o resto que essa corrida já tinha número: hold-out da semente 0, dose–resposta de `B`, loss das sementes 1 e 2, correlação proxy×emulador e a tabela dos quatro braços. Em 2026-11-03, na mesma máquina, entrou o hold-out MoE das sementes 1 e 2, a dose de `B` dessas sementes e a loss de treino do LM denso nas sementes 1 e 2. No fim desse dia (~22:47–22:49) entrou o hold-out do LM denso nas sementes 1 e 2. As figuras do proxy, da corrida de 02/11, passaram a ficar nesta pasta.

| Arquivo | O que registra |
| --- | --- |
| [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md) | Grade λ = 0.0, 0.5 e 1.0, semente 0. `last_task` / `last_total` do último passo. |
| [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md) | LM denso (modelo 3), semente 0. `L_task` = `L_total` no último passo. O MoE com λ = 0 é a outra corrida. |
| [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md) | LM denso, sementes 1 e 2. `last_task` / `last_total`. Semente 1 em batch 2 (4522 passos) depois da recuperação; semente 2 em batch 4 (2336). O hold-out está na nota de 03/11. |
| [lambda_grid_seeds_wikitext_2026-11-02.md](lambda_grid_seeds_wikitext_2026-11-02.md) | A mesma grade, sementes 1 e 2 (a semente 0 repetida ao lado, igual à nota de 31/10). |
| [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md) | PPL de `valid` e `test`, parede p50/p95, `swaps` e `bytes_moved`. Semente 0: denso e λ = 0, 0.5, 1. |
| [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md) | O mesmo hold-out, sementes 1 e 2: grade MoE e, no fim do dia, o LM denso. A média ± dp amostral da PPL de `test` nas sementes 0–2 fica no MoE. O denso lista as três PPL de `test` sem média. |
| [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md) | PPL, parede p50/p95, `swaps` e `bytes_moved` em 0,5× / 1× / 2× `B`, sem retreino. O denso é `n/a`. |
| [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md) | A mesma curva, sementes 1 e 2, só a grade MoE. O denso é `n/a`. |
| [proxy_emulator_wikitext_2026-11-02.md](proxy_emulator_wikitext_2026-11-02.md) | Pearson e Spearman de `c_mem` contra bytes e contra `swaps`, no conjunto e por λ. Entropia do primeiro e do último passo. Figuras: [entropy.svg](entropy.svg), [c_mem_vs_bytes.svg](c_mem_vs_bytes.svg), [c_mem_vs_swaps.svg](c_mem_vs_swaps.svg). |
| [model_comparison_wikitext_2026-11-02.md](model_comparison_wikitext_2026-11-02.md) | `comparison.json` preenchido. 26 células `filled`, 4 `not_applicable` (denso fora de 1×B). Nenhuma `pending`. |

## Medido em 2026-11-02

- Perplexidade de hold-out em `valid` e `test`, semente 0, no `B` de referência.
- Latência de parede (p50/p95), `swaps` e `bytes_moved` desse hold-out. No denso, `swaps` e `bytes_moved` são null.
- Dose–resposta de `B` (0,5× / 1× / 2×): PPL, parede, `swaps` e `bytes_moved` em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md). A PPL em 1×B coincide com o hold-out em 2 casas. A parede em 1×B é a remedição por `set_hot_budget` e pode diferir um pouco da tabela de hold-out.
- Sementes 1 e 2 da grade MoE: `last_task` / `last_total`.
- Modelo 2 (LRU) como linha de relatório. No código ele é o mesmo checkpoint λ = 0; a tabela copia o modelo 1.
- Correlação proxy×emulador e entropia inicial/final do roteador. As figuras (`entropy.svg`, `c_mem_vs_bytes.svg`, `c_mem_vs_swaps.svg`) estão nesta pasta.

## Medido em 2026-11-03

- Hold-out MoE das sementes 1 e 2 em `valid` e `test`, no `B` de referência: PPL, parede p50/p95, `swaps` e `bytes_moved`. `wall_report_grade` verdadeiro, GTX 1060. Nota: [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). O LM denso dessas sementes entrou numa corrida à parte, no fim do dia; ver o item abaixo.
- PPL de `test` no `B` de referência, sementes 0–2, média ± desvio padrão amostral (n = 3). λ = 0.0: 351.06 ± 0.81. λ = 0.5: 333.42 ± 4.40. λ = 1.0: 333.78 ± 3.15. A semente 0 é a PPL já publicada em 2 casas na nota de 02/11.
- Dose–resposta de `B` (0,5× / 1× / 2×) nas sementes 1 e 2: PPL, parede, `swaps` e `bytes_moved` em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). A PPL em 1×B coincide com o hold-out em 2 casas. A parede em 1×B é a remedição por `set_hot_budget` e pode diferir um pouco da tabela de hold-out. A PPL fica quase plana no eixo de `B`, como na semente 0. O denso é `n/a`.
- Loss de treino do LM denso nas sementes 1 e 2: [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md). Semente 1: `last_task` 5.725013732910156, batch 2, 4522 passos, depois da recuperação de uma escrita de checkpoint que saiu com código 5. Semente 2: `last_task` 5.2047600746154785, batch 4, 2336 passos. As duas com 2391808 tokens e `status` done.
- Hold-out do LM denso nas sementes 1 e 2, ~22:47–22:49, GTX 1060 6GB, `wall_report_grade` verdadeiro: seção LM denso de [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md). PPL em 2 casas: semente 1, `valid` 317.19 e `test` 318.16 (batch do forward 2, 4522 passos); semente 2, `valid` 327.86 e `test` 327.09 (batch do forward 4, 2336 passos). `swaps` e `bytes_moved` null. Dose `n/a`. A parede da semente 1 fica de fora da comparação com as outras: o batch do forward é 2, não 4. A semente 0, já publicada, tem `test` 339.05 e também batch 4. As três PPL de `test` ficam lado a lado, sem média: a semente 1 é a recuperação em batch 2.

## Ainda sem número medido neste repositório

- Um agregado proxy×emulador que some as três sementes. O `n` da nota de 02/11 não fecha com essa soma: [proxy_emulator_wikitext_2026-11-02.md](proxy_emulator_wikitext_2026-11-02.md).

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (o markdown gerado pelo script é PPL por multiplicador; o denso entra como `n/a`). A seção MoE de [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md) copia a corrida pedida das sementes 1 e 2 (`scope` request no resumo), não esse `summary.json` da descoberta. A seção LM denso copia o `holdout.json` ao lado de `outputs/baseline/wikitext/seed-1/checkpoint.pt` e de `seed-2/checkpoint.pt`. O `done.json` do denso nessas sementes fica em `outputs/baseline/wikitext/seed-<n>/`. O JSON de cada split já tem `swaps`, `bytes_moved` e `wall_p50_ms` / `wall_p95_ms`, no `B` de referência e em cada ponto da curva. A nota publicada da dose copia esses campos do JSON para 0,5×, 1× e 2×. `experiments/run_model_comparison.py` grava `outputs/comparison/comparison.json`, com o schema dos quatro braços. Célula `pending` não é número; na corrida de 2026-11-02 não sobrou nenhuma. A coluna de parede do relatório exige a corrida na GTX 1060 (`wall_report_grade`); um JSON de CPU traz os mesmos campos com os ms null.

A correlação proxy×emulador e a curva de entropia também nascem em `outputs/`: `experiments/plot_proxy_emulator.py` escreve `outputs/proxy-emulator/` (`summary.json`, CSV e SVG) a partir de `outputs/wikitext/seed-*/lambda-*/train.jsonl` e, no layout antigo, `outputs/wikitext/lambda-*/train.jsonl`. Não treina. O resumo de 2026-11-02 está na nota. As três SVG dessa corrida estão copiadas nesta pasta; o script continua gravando a saída em `outputs/`, que o Git ignora.

O `.gitignore` versiona os `.md` e os `.svg` desta pasta e deixa de fora checkpoint, array binário, CSV e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto (`outputs/`, inclusive `cold/*.f32` e logs).
