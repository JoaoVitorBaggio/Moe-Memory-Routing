# src/

Rede, roteador, emulador de três níveis e a loss `L_tarefa + λ · C_mem + β · L_balanço`. PyTorch, fp32. A definição da loss está em [`docs/funcao-de-custo.md`](../docs/funcao-de-custo.md); os números saem de [`configs/default.yaml`](../configs/default.yaml).

Não há resultado de treino aqui. O passo de fumaça usa batch aleatório e peso inicial.

A contagem da arquitetura fechada é 47.939.328 parâmetros (~48 M): a embedding do GPT-2 sozinha é ~12,9 M e os experts (`E=32`, `d_ff=512`, 4 camadas) ~33,7 M. A faixa pedida é 10–50 M. Sem amarrar o `lm_head` o total passa de 60 M; encolher `d_model` ou `d_ff` para forçar 40 M iria contra o desenho (expert maior, não modelo menor).

## Módulos

| Módulo | Papel |
| --- | --- |
| `moe_memory/model.py` | LM causal. 4 blocos, toda FFN é MoE. Mistura dos top-2 = média ponderada (pesos somam 1) no treino e na inferência, para o residual não mudar de escala. |
| `moe_memory/router.py` | Projeção linear + chave `expert + embed(ℓ(e))`. Sem `router_sees_level`, a chave ignora o nível. |
| `moe_memory/emulator.py` | Quente / morno / frio, um passo por batch. LRU com desempate pela contagem do batch, teto `B`. Conta trocas e bytes. |
| `moe_memory/loss.py` | CE do próximo token, `C_mem` e o balanceamento do Switch. Treino: média ponderada dos top-2. Inferência: soma dos top-2. Essa diferença é do **custo**, não do hidden (“custo real de acesso”). |
| `moe_memory/config.py` | Lê e valida o YAML. |
| `moe_memory/smoke.py` | Um passo sintético. Não é a corrida de WikiText-2. |

`w(a,b) = |c(b) − c(a)|` não aparece no YAML: a loss deriva da tabela `c`.

Homes físicos: morno = tensor de CPU (pinned se houver CUDA); frio = memmap. No smoke os `nn.Parameter` continuam no device de compute para o backward caber em CPU. Trocar o storage do parâmetro na 1060, com experts fora de `B` fora da VRAM, está marcado como `TODO(1060)` em `emulator.py`.

## Como rodar o passo de fumaça

Na raiz do repositório, com Python ≥ 3.10:

```bash
python3 -m pip install -r requirements.txt
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 -m pytest
```

`smoke_train.py` instancia modelo, emulador e loss a partir de `configs/default.yaml`, faz um forward e um backward num batch aleatório e dá um passo de AdamW. `smoke_eval.py` faz um forward com a agregação de inferência (`sum`). `--lambda 0` usa o ponto da grade sem `C_mem`. O device padrão é `cpu`.

A loss impressa é de init aleatória. Não é perplexidade de relatório, não é latência de parede e não substitui a corrida overnight de WikiText-2.
