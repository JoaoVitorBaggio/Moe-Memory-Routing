# experiments/

Entradas finas do protocolo. O desenho está em [`docs/protocolo-experimental.md`](../docs/protocolo-experimental.md).

O que existe agora é só o passo de fumaça, lendo [`configs/default.yaml`](../configs/default.yaml):

```bash
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 experiments/smoke_train.py --lambda 0
```

Um batch aleatório, um passo. Não treina TinyStories nem WikiText-2, não varre `λ` e não mede latência na 1060. A saída numérica é init aleatória — não vai para `results/`.

Ainda não há script dos quatro modelos, do hold-out nem da dose–resposta. O emulador já aceita trocar o teto (`set_hot_budget`, 0,5× / 1× / 2×) para quando essa avaliação entrar.

Não colocar saída grande nesta pasta. Métrica pequena vai para `results/`; checkpoint e dataset ficam de fora do Git. O home frio do smoke cai em `outputs/` (gitignored).
