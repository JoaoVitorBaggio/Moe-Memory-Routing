# results/

Resultados. **Pasta vazia de propósito.** Não há tabela, curva nem perplexidade medida.

O script `experiments/run_holdout_eval.py` grava o JSON em `outputs/` (gitignored): `holdout.json` ao lado de cada checkpoint e `outputs/holdout/summary.json` na descoberta dos quatro. Quando a corrida na GTX 1060 existir, a tabela pequena pode ser copiada para cá, com o `λ`, o `seq_len` e a máquina. Dose–resposta e latência de parede ainda não têm número.

O `.gitignore` deixa de fora checkpoint, array binário e outros artefatos grandes. Não commitar lixo de treino para “não perder”. Disco local ou outro armazenamento fica com o arquivo bruto.
