# results/

Resumos pequenos que entram no Git. O que está aqui, em 2026-10-31, é **loss de treino** da semente 0 no WikiText-2, copiada do `outputs/` local (gitignored) da Fedora com GTX 1060. Não é perplexidade de hold-out.

| Arquivo | O que registra |
| --- | --- |
| [lambda_grid_wikitext_2026-10-31.md](lambda_grid_wikitext_2026-10-31.md) | Grade λ = 0.0, 0.5 e 1.0, semente 0. `last_task` / `last_total` do último passo. |
| [baseline_dense_wikitext_2026-10-31.md](baseline_dense_wikitext_2026-10-31.md) | LM denso (modelo 3), semente 0. `L_task` = `L_total` no último passo. O MoE com λ = 0 é a outra corrida. |

## Ainda sem número medido neste repositório

- Perplexidade de hold-out em `valid` e `test`.
- Sementes 1 e 2. O protocolo pede no mínimo 3.
- Dose–resposta de `B` (0,5× / 1× / 2×).
- Latência de parede (p50/p95), `swaps` e `bytes_moved` de relatório.
- Modelo 2 (LRU) como linha medida. No código ele é o mesmo checkpoint λ = 0; `comparison.json` não foi preenchido.
- Correlação proxy×emulador e curva de entropia do roteador.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (a tabela é PPL por multiplicador; o denso entra como `n/a`). O JSON de cada split já tem `swaps`, `bytes_moved` e `wall_p50_ms` / `wall_p95_ms`, no `B` de referência e em cada ponto da curva. `experiments/run_model_comparison.py` grava `outputs/comparison/comparison.json`, com o schema dos quatro braços. Sem hold-out e sem essa curva as células ficam `pending`; isso não entra aqui. A coluna de parede do relatório exige a corrida na GTX 1060 (`wall_report_grade`); um JSON de CPU traz os mesmos campos com os ms null.

A correlação proxy×emulador e a curva de entropia também ficam em `outputs/` até alguém copiar o resumo para cá: `experiments/plot_proxy_emulator.py` escreve `outputs/proxy-emulator/` (`summary.json`, CSV e SVG) a partir de `outputs/wikitext/seed-*/lambda-*/train.jsonl` e, no layout antigo, `outputs/wikitext/lambda-*/train.jsonl`. Não treina.

O `.gitignore` versiona os `.md` desta pasta e deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto (`outputs/`, inclusive `cold/*.f32` e logs).
