# src/

Rede, roteador, emulador de três níveis e a loss `L_tarefa + λ · C_mem + β · L_balanço`. PyTorch, fp32. A definição da loss está em [`docs/funcao-de-custo.md`](../docs/funcao-de-custo.md); os números saem de [`configs/default.yaml`](../configs/default.yaml).

Não há resultado de treino aqui. O passo de fumaça usa batch aleatório e peso inicial.

A contagem da arquitetura fechada é 47.939.328 parâmetros (~48 M): a embedding do GPT-2 sozinha é ~12,9 M e os experts (`E=32`, `d_ff=512`, 4 camadas) ~33,7 M. A faixa pedida é 10–50 M. Sem amarrar o `lm_head` o total passa de 60 M; encolher `d_model` ou `d_ff` para forçar 40 M iria contra o desenho (expert maior, não modelo menor).

## Módulos

| Módulo | Papel |
| --- | --- |
| `moe_memory/model.py` | LM causal. 4 blocos, toda FFN é MoE. Mistura dos top-2 = média ponderada (pesos somam 1) no treino e na inferência, para o residual não mudar de escala. |
| `moe_memory/dense.py` | Baseline denso (modelo 3). Mesma atenção e o mesmo FFN de um expert, com largura `k * expert_hidden`. Sem roteador, sem residência, loss só `L_tarefa`. Não é `λ=0`. |
| `moe_memory/router.py` | Projeção linear + chave `expert + embed(ℓ(e))`. Sem `router_sees_level`, a chave ignora o nível. |
| `moe_memory/emulator.py` | Quente / morno / frio, um passo por batch. LRU com desempate pela contagem do batch, teto `B`. Conta trocas e bytes. |
| `moe_memory/residency.py` | No CUDA, põe o storage do expert no home do `ℓ(e)`: quente na VRAM, morno em CPU pinned, frio em memmap. A cópia para o compute é contador à parte (`staged_loads`), não entra no LRU. |
| `moe_memory/corpus.py` | TinyStories e WikiText-2, BPE do GPT-2, blocos para a CE do próximo token. |
| `moe_memory/loss.py` | CE do próximo token, `C_mem` e o balanceamento do Switch. Treino: média ponderada dos top-2. Inferência: soma dos top-2. Essa diferença é do **custo**, não do hidden (“custo real de acesso”). |
| `moe_memory/config.py` | Lê e valida o YAML. |
| `moe_memory/smoke.py` | Um passo de treino. O atalho sintético não lê corpus. |

`w(a,b) = |c(b) − c(a)|` não aparece no YAML: a loss deriva da tabela `c`.

Homes físicos, quando o device é CUDA: morno = CPU pinned (sobe no forward se o top-2 escolheu); frio = memmap (sobe se foi escolhido ou se o LRU promoveu). No smoke de CPU os `nn.Parameter` continuam no device de compute. O procedimento da parede na 1060 está em [`docs/plano-de-hardware.md`](../docs/plano-de-hardware.md).

## Como rodar o passo de fumaça

Na raiz do repositório, com Python ≥ 3.10:

```bash
python3 -m pip install -r requirements.txt
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 experiments/train_tinystories.py --steps 1
python3 experiments/smoke_wikitext.py
python3 experiments/run_lambda_grid.py --device cuda
python3 experiments/run_baseline.py --device cuda
python3 experiments/run_holdout_eval.py --device cuda
python3 -m pytest
```

`smoke_train.py` instancia modelo, emulador e loss a partir de `configs/default.yaml`, faz um forward e um backward num batch aleatório e dá um passo de AdamW. `smoke_eval.py` faz um forward com a agregação de inferência (`sum`) em init aleatória — não é o hold-out. `--lambda 0` usa o ponto da grade sem `C_mem`. O device padrão é `cpu`. `train_tinystories.py` e `smoke_wikitext.py` leem os corpora do YAML.

A loss impressa do smoke é de init aleatória ou de um passo curto. Não é perplexidade de relatório, não é latência de parede. A corrida overnight de WikiText-2, um λ por vez, o baseline denso e o hold-out (`run_holdout_eval.py`, valid e test, mesma CE do treino, mais a curva `eval.B_mult` sem retreino) estão em [`experiments/README.md`](../experiments/README.md).
