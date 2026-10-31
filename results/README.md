# results/

Resultados. **Pasta vazia de propósito.** Não há tabela, curva nem perplexidade medida.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (a tabela é PPL por multiplicador; o denso entra como `n/a`). O JSON de cada split já tem `swaps`, `bytes_moved` e `wall_p50_ms` / `wall_p95_ms`, no `B` de referência e em cada ponto da curva. `experiments/run_model_comparison.py` grava `outputs/comparison/comparison.json`, com o schema dos quatro braços. Sem hold-out e sem essa curva as células ficam `pending`; isso não entra aqui. Não há número medido neste repositório. A coluna de parede do relatório exige a corrida na GTX 1060 (`wall_report_grade`); um JSON de CPU traz os mesmos campos com os ms null.

A correlação proxy×emulador e a curva de entropia também ficam em `outputs/` até alguém copiar o resumo para cá: `experiments/plot_proxy_emulator.py` escreve `outputs/proxy-emulator/` (`summary.json`, CSV e SVG) a partir de `outputs/wikitext/seed-*/lambda-*/train.jsonl` e, no layout antigo, `outputs/wikitext/lambda-*/train.jsonl`. Não treina.

O `.gitignore` deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto.
