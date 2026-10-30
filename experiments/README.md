# experiments/

Entradas finas do protocolo. O desenho está em [`docs/protocolo-experimental.md`](../docs/protocolo-experimental.md).

O passo sintético e os dois loaders leem [`configs/default.yaml`](../configs/default.yaml):

```bash
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 experiments/smoke_train.py --lambda 0
python3 experiments/train_tinystories.py --steps 1
python3 experiments/smoke_wikitext.py
python3 experiments/wall_latency.py
```

`smoke_train.py` ainda é um batch aleatório. `train_tinystories.py` faz um passo curto no corpus de desenvolvimento (streaming, tamanho do `smoke`). `smoke_wikitext.py` só confere train/valid/test e um batch; não treina. `wall_latency.py` sem CUDA explica a medição e não imprime ms. Nada disso varre `λ`, nada disso é número de relatório e nada vai para `results/`.

Ainda não há script dos quatro modelos, do hold-out nem da dose–resposta. O emulador já aceita trocar o teto (`set_hot_budget`, 0,5× / 1× / 2×) para quando essa avaliação entrar.

Não colocar saída grande nesta pasta. Métrica pequena vai para `results/`; checkpoint e dataset ficam de fora do Git. O home frio do smoke cai em `outputs/` (gitignored).
