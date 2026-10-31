# results/

Resultados. **Pasta vazia de propósito.** Não há tabela, curva nem perplexidade medida.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint da semente 0 (`seed-0/`, ou o layout antigo se só ele tiver o arquivo), `outputs/holdout/summary.json` na descoberta dos quatro, e a curva de `B` em `dose_response.json` mais `dose_response.md` (PPL por multiplicador; o denso entra como `n/a`). Quando a corrida na GTX 1060 existir, a tabela pequena pode ser copiada para cá, com o `λ`, o `seq_len` e a máquina. Ainda não há número medido neste repositório. Latência de parede também não.

O `.gitignore` deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto.
