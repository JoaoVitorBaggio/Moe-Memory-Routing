# results/

Resultados. **Pasta vazia de propósito.** Não há tabela, curva nem perplexidade medida.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (a tabela é PPL por multiplicador; o denso entra como `n/a`). O JSON de cada split já tem `swaps`, `bytes_moved` e `wall_p50_ms` / `wall_p95_ms`, no `B` de referência e em cada ponto da curva. Não há número medido neste repositório. A coluna de parede do relatório exige a corrida na GTX 1060 (`wall_report_grade`); um JSON de CPU traz os mesmos campos com os ms null.

O `.gitignore` deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto.
