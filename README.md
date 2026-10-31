# Roteamento consciente de memória

Mixture of Experts (MoE) pequeno com hierarquia **emulada** VRAM / RAM / SSD e roteador treinado com custo explícito de memória e de troca. Trabalho final de **Aprendizado Profundo (AGL11157)**, UFRGS, 2026/2.

**Status:** blueprint revisado (out/2026) após feedback do prof. Joel Luís Carbonera. O esqueleto PyTorch está em `src/moe_memory/`: passo sintético, loaders de TinyStories (dev) e WikiText-2 (entrega), e residência física fora da VRAM quando o device é CUDA. Não há número de desempenho neste repositório. `notebooks/` e `results/` seguem vazios.

## Resumo

Cada expert fica num nível de memória emulado: quente (VRAM), morno (RAM) ou frio (SSD). A loss junta a entropia cruzada do próximo token, um *proxy* diferenciável do custo de memória/troca, e o balanceamento de carga. A hipótese é que variar esse custo (e o orçamento) melhora a fronteira qualidade × latência / I/O frente a baselines fortes, no mesmo corpus.

## Pergunta de pesquisa

Um roteador treinado com penalidade de memória e de troca melhora a fronteira qualidade × latência (e o volume de I/O) em relação a:

1. MoE top-k sem custo de memória;
2. o mesmo MoE com **LRU de experts** — o adversário principal, não só “qualquer MoE”;
3. um modelo denso com parâmetros ativos comparáveis?

Pergunta secundária (dose–resposta): treinado sob orçamento `B` de referência, o roteador reage de forma monótona em **0,5×`B`** e **2×`B`** na avaliação? (`λ` define curvas de treino separadas.)

## Experimento mínimo (entrega principal)

Um único emulador de três níveis (**por batch**), **um** vetor de custo (perfil consumer ~6 GB; partida `c`=(1,10,20|30), `w=Δc`), **`E=32`**, `k=2`, orçamento `B=25%` (8/32), roteador com **`ℓ(e)`**, treino por `λ ∈ {0, 0,5, 1}` (o ponto `λ=0` é o baseline 1) e avaliação em 0,5× e 2×`B`, contra top-k e **LRU** (e denso). Sem catálogo de várias GPUs de mercado no núcleo.

Extensões (não bloqueiam a entrega mínima):

- tabelas de custo “realismo de mercado” (consumer 6–12 GB, mid cloud, high-end);
- treino com orçamento variável por batch/época (**adiado**);
- calibrar pesos em preços Spot reais.

## Decisões fechadas neste blueprint

| Tema | Decisão |
| --- | --- |
| Corpus | TinyStories só no dev; WikiText-2 no treino/avaliação de entrega (overnight) |
| Experts / `k` | **`E = 32`**; **`k = 2`**. Preferir **aumentar o tamanho do expert** a encolher o modelo se faltar capacidade |
| Arquitetura fina | `d_model=256`, 4 camadas (todo FFN é MoE), 4 cabeças (`d_head=64`), `d_ff=512` (768 se faltar capacidade), contexto 256 (até 512), tokenizer GPT-2, fp32, `lm_head` amarrado (~48 M; sem o laço o vocab estoura 50 M) |
| Orçamento `B` | **25%** dos experts quentes (**8/32**) — teto **artificial** (o modelo inteiro cabe na 1060) |
| Emulador / roteador | Granularidade **por batch**; roteador recebe **`ℓ(e)`** |
| Proxy `k=2` | Treino: média ponderada top-2; inferência: soma top-2 |
| Hardware de referência | VRAM útil **5,5 GB** (6 GB − ~0,5 GB driver/display); RAM de trabalho **10 GB** dos 16 GB; SSD frio **1 GB** (2 GB se checkpoints) |
| `λ` / `β` | Treino **especializado por `λ`** (não condicionado); grade **`λ ∈ {0, 0,5, 1}`**; `β` fixo pequeno (ordem Switch) |
| Baseline-2 | **Só LRU de experts** como adversário; declarado no protocolo e não muda no meio |
| Denso | Ativos comparáveis: tronco + `k`·expert |
| Hardware no núcleo | Emulação VRAM/RAM/SSD; proxy alinhado a bytes/trocas; monitorar entropia |
| Hold-out / seeds | ~10% hold-out; **3 seeds** |
| Latência de parede | **Sim**, na 1060 (`B` fixo) |
| Orçamento variável no treino | **Adiado** (extensão) |
| Dose–resposta | Figura 0,5× / 1× / 2× + tabela contra baselines |
| Colocação | Política dinâmica de cache/residência (não tool-calls) |
| Custos por nível | Partida: c=(1,10,20|30), w=Δc amarrado a c; calibrar na 1060 |

## Motivação (curta)

MoE só calcula parte dos parâmetros por token, mas o tempo real depende de *onde* o expert está. PowerInfer e *LLM in a flash* exploram offload; o recorte daqui é **treinar o roteador com esse custo**. Detalhe e fichas em [docs/referencias.md](docs/referencias.md).

## Arquitetura em três níveis (emulada)

| Nível | Papel no emulador | Custo relativo (ordem) |
| --- | --- | --- |
| Quente | “VRAM” — capacidade cara, acesso barato | baixo `c(ℓ)`, alto preço de residência no orçamento |
| Morno | “RAM” — meio do caminho | médio |
| Frio | “SSD” — capacidade barata, troca cara | alto `c(ℓ)` e `w` de promoção |

Os pesos `c(ℓ)` e `w(a,b)` são variáveis de experimento (`w` amarrado a `c`): mudar a tabela deve forçar o roteador a adaptar. Fórmulas e decisões da call 30/10/2026 em [docs/funcao-de-custo.md](docs/funcao-de-custo.md). Hardware e emulação em [docs/plano-de-hardware.md](docs/plano-de-hardware.md).

## Baselines e métricas

| # | Modelo | Papel |
| --- | --- | --- |
| 1 | MoE top-k (`λ = 0`) | qualidade MoE sem cobrar memória |
| 2 | MoE + **LRU de experts** | único adversário de offload |
| 3 | Denso (ativos comparáveis) | separar esparsidade de penalidade |
| 4 | Este projeto (`λ > 0`) | roteador + emulador + política de colocação |

Métricas no hold-out (~10%, 3 seeds): perplexidade; **latência p50/p95 de parede na 1060**; bytes e trocas do emulador; curva qualidade × custo/orçamento. Protocolo em [docs/protocolo-experimental.md](docs/protocolo-experimental.md).

## Corpus (dev vs entrega)

- **Desenvolvimento:** TinyStories — iterar rápido, achar bug, checar conversão e se o treino “anda”.
- **Treino real / tabelas do relatório:** WikiText-2 — corrida de verdade, tipicamente **de madrugada com o PC ligado**.
- WikiText-103 só se o 2 ficar curto demais; não é o padrão.

O código e o protocolo devem aceitar os dois via config; a entrega numérica usa WikiText-2.

## Como contribuir e rodar

Python ≥ 3.10 e PyTorch em fp32. Na GTX 1060, o wheel CUDA; o passo abaixo também roda em CPU. Detalhe dos módulos em [`src/README.md`](src/README.md).

```bash
python3 -m pip install -r requirements.txt
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 experiments/train_tinystories.py --steps 1
python3 experiments/smoke_wikitext.py
python3 experiments/wall_latency.py
python3 experiments/run_lambda_grid.py --device cuda
python3 experiments/run_baseline.py --device cuda
python3 experiments/run_holdout_eval.py --device cuda
python3 -m pytest
```

`smoke_train.py` dá **um** passo num batch aleatório. `train_tinystories.py` dá um passo curto no corpus de desenvolvimento. `smoke_wikitext.py` só carrega os splits oficiais. `wall_latency.py` na GTX 1060 (`--device cuda`) cronometra o forward com experts fora de `B` fora da VRAM; sem CUDA ele não inventa ms. A loss impressa desses smokes é de peso inicial ou de um passo curto. A grade overnight (`run_lambda_grid.py`) treina `λ ∈ {0, 0,5, 1}` no WikiText-2, um por vez, e grava log em `outputs/`. `run_baseline.py` treina o LM denso (modelo 3) no mesmo WikiText-2, sem roteador e sem residência: não é o MoE com `λ=0`. `run_holdout_eval.py` lê esses quatro checkpoints e calcula a perplexidade de token em `valid` e `test` (mesmo BPE, mesmo bloco, `seq_len` do checkpoint). Não há número medido neste repositório. Dose–resposta continua fora. Detalhe em [`experiments/README.md`](experiments/README.md).

| Pasta | Para que serve |
| --- | --- |
| `src/` | Modelo, roteador, emulador de níveis e loss |
| `experiments/` | Fumaça, treino WikiText-2 por λ, baseline denso e perplexidade de hold-out. Dose–resposta ainda não. |
| `configs/` | Orçamento, tabela de custos e hiperparâmetros |
| `notebooks/` | Exploração |
| `results/` | Tabelas e curvas, quando existirem |

Documentação em português do Brasil. Não versionar checkpoint, dataset baixado nem resultado grande (veja `.gitignore`).

## Integrantes

- João Vítor Baggio
- Eric D Arriaga
- Taylor Souza Frutuoso da Costa

## Documentação

- [Problema e escopo](docs/definicao.md)
- [Função de custo](docs/funcao-de-custo.md)
- [Protocolo experimental](docs/protocolo-experimental.md)
- [Plano de hardware e emulação](docs/plano-de-hardware.md)
- [Referências](docs/referencias.md)
