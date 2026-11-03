# results/

Resumos pequenos que entram no Git. O bruto fica em `outputs/` (gitignored): checkpoint, `train.jsonl`, SVG e os JSON de medição. Em 2026-10-31 entrou a loss de treino da semente 0 no WikiText-2, copiada da Fedora com GTX 1060. Em 2026-11-02 (America/Sao_Paulo, a mesma máquina) entrou o resto que essa corrida já tinha número: hold-out da semente 0, dose–resposta de `B`, loss das sementes 1 e 2, correlação proxy×emulador e a tabela dos quatro braços. Em 2026-11-03, na mesma máquina, entrou o hold-out MoE das sementes 1 e 2, a dose de `B` dessas sementes e a loss de treino do LM denso nas sementes 1 e 2.

| Arquivo | O que registra |
| --- | --- |
| [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md) | Grade λ = 0.0, 0.5 e 1.0, semente 0. `last_task` / `last_total` do último passo. |
| [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md) | LM denso (modelo 3), semente 0. `L_task` = `L_total` no último passo. O MoE com λ = 0 é a outra corrida. |
| [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md) | LM denso, sementes 1 e 2. `last_task` / `last_total`. Semente 1 em batch 2 (4522 passos) depois da recuperação; semente 2 em batch 4 (2336). Sem PPL de hold-out. |
| [lambda_grid_seeds_wikitext_2026-11-02.md](lambda_grid_seeds_wikitext_2026-11-02.md) | A mesma grade, sementes 1 e 2 (a semente 0 repetida ao lado, igual à nota de 31/10). |
| [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md) | PPL de `valid` e `test`, parede p50/p95, `swaps` e `bytes_moved`. Semente 0: denso e λ = 0, 0.5, 1. |
| [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md) | O mesmo hold-out, sementes 1 e 2, só a grade MoE. Inclui a média ± dp amostral da PPL de `test` nas sementes 0–2. |
| [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md) | PPL, parede p50/p95, `swaps` e `bytes_moved` em 0,5× / 1× / 2× `B`, sem retreino. O denso é `n/a`. |
| [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md) | A mesma curva, sementes 1 e 2, só a grade MoE. O denso é `n/a`. |
| [proxy_emulator_wikitext_2026-11-02.md](proxy_emulator_wikitext_2026-11-02.md) | Pearson e Spearman de `c_mem` contra bytes e contra `swaps`, no conjunto e por λ. Entropia do primeiro e do último passo. |
| [model_comparison_wikitext_2026-11-02.md](model_comparison_wikitext_2026-11-02.md) | `comparison.json` preenchido. 26 células `filled`, 4 `not_applicable` (denso fora de 1×B). Nenhuma `pending`. |

## Medido em 2026-11-02

- Perplexidade de hold-out em `valid` e `test`, semente 0, no `B` de referência.
- Latência de parede (p50/p95), `swaps` e `bytes_moved` desse hold-out. No denso, `swaps` e `bytes_moved` são null.
- Dose–resposta de `B` (0,5× / 1× / 2×): PPL, parede, `swaps` e `bytes_moved` em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md). A PPL em 1×B coincide com o hold-out em 2 casas. A parede em 1×B é a remedição por `set_hot_budget` e pode diferir um pouco da tabela de hold-out.
- Sementes 1 e 2 da grade MoE: `last_task` / `last_total`.
- Modelo 2 (LRU) como linha de relatório. No código ele é o mesmo checkpoint λ = 0; a tabela copia o modelo 1.
- Correlação proxy×emulador e entropia inicial/final do roteador. O SVG não entra aqui.

## Medido em 2026-11-03

- Hold-out MoE das sementes 1 e 2 em `valid` e `test`, no `B` de referência: PPL, parede p50/p95, `swaps` e `bytes_moved`. `wall_report_grade` verdadeiro, GTX 1060. O denso dessas sementes não entrou. Nota: [holdout_wikitext_2026-11-03.md](holdout_wikitext_2026-11-03.md).
- PPL de `test` no `B` de referência, sementes 0–2, média ± desvio padrão amostral (n = 3). λ = 0.0: 351.06 ± 0.81. λ = 0.5: 333.42 ± 4.40. λ = 1.0: 333.78 ± 3.15. A semente 0 é a PPL já publicada em 2 casas na nota de 02/11.
- Dose–resposta de `B` (0,5× / 1× / 2×) nas sementes 1 e 2: PPL, parede, `swaps` e `bytes_moved` em [dose_response_wikitext_2026-11-03.md](dose_response_wikitext_2026-11-03.md). A PPL em 1×B coincide com o hold-out em 2 casas. A parede em 1×B é a remedição por `set_hot_budget` e pode diferir um pouco da tabela de hold-out. A PPL fica quase plana no eixo de `B`, como na semente 0. O denso é `n/a`.
- Loss de treino do LM denso nas sementes 1 e 2: [baseline_dense_wikitext_2026-11-03.md](baseline_dense_wikitext_2026-11-03.md). Semente 1: `last_task` 5.725013732910156, batch 2, 4522 passos, depois da recuperação de uma escrita de checkpoint que saiu com código 5. Semente 2: `last_task` 5.2047600746154785, batch 4, 2336 passos. As duas com 2391808 tokens e `status` done.

## Ainda sem número medido neste repositório

- PPL de hold-out do LM denso nas sementes 1 e 2 (`valid`, `test`), e a parede desse forward. A loss de treino dessas sementes já está na nota de 03/11. Na semente 0 o hold-out do denso continua em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md) (`test` 339.05).
- Figura do proxy (SVG). O resumo numérico está na nota; o desenho continua em `outputs/proxy-emulator/`.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (o markdown gerado pelo script é PPL por multiplicador; o denso entra como `n/a`). As notas de 2026-11-03 copiam a corrida pedida das sementes 1 e 2 (`scope` request no resumo), não esse `summary.json` da descoberta. O `done.json` do denso nessas sementes fica em `outputs/baseline/wikitext/seed-<n>/`. O JSON de cada split já tem `swaps`, `bytes_moved` e `wall_p50_ms` / `wall_p95_ms`, no `B` de referência e em cada ponto da curva. A nota publicada da dose copia esses campos do JSON para 0,5×, 1× e 2×. `experiments/run_model_comparison.py` grava `outputs/comparison/comparison.json`, com o schema dos quatro braços. Célula `pending` não é número; na corrida de 2026-11-02 não sobrou nenhuma. A coluna de parede do relatório exige a corrida na GTX 1060 (`wall_report_grade`); um JSON de CPU traz os mesmos campos com os ms null.

A correlação proxy×emulador e a curva de entropia também nascem em `outputs/`: `experiments/plot_proxy_emulator.py` escreve `outputs/proxy-emulator/` (`summary.json`, CSV e SVG) a partir de `outputs/wikitext/seed-*/lambda-*/train.jsonl` e, no layout antigo, `outputs/wikitext/lambda-*/train.jsonl`. Não treina. O resumo de 2026-11-02 está na nota; o SVG fica de fora do Git.

O `.gitignore` versiona os `.md` desta pasta e deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto (`outputs/`, inclusive `cold/*.f32` e logs).
