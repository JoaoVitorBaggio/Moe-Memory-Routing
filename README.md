# Roteamento consciente de memória

Mixture of Experts (MoE) pequeno com hierarquia **emulada** VRAM / RAM / SSD e roteador treinado com custo explícito de memória e de troca. Trabalho final de **Aprendizado Profundo (AGL11157)**, UFRGS, 2026/2.

**Status:** blueprint revisado (out/2026) após feedback do prof. Joel Luís Carbonera. Código, treino, experimentos e resultados estão **em construção**. Pastas `src/`, `configs/`, `experiments/`, `notebooks/` e `results/` existem só como esqueleto. Não há número de desempenho neste repositório.

## Resumo

Cada expert fica num nível de memória emulado: quente (VRAM), morno (RAM) ou frio (SSD). A loss junta a entropia cruzada do próximo token, um *proxy* diferenciável do custo de memória/troca, e o balanceamento de carga. A hipótese é que variar esse custo (e o orçamento) melhora a fronteira qualidade × latência / I/O frente a baselines fortes, no mesmo corpus.

## Pergunta de pesquisa

Um roteador treinado com penalidade de memória e de troca melhora a fronteira qualidade × latência (e o volume de I/O) em relação a:

1. MoE top-k sem custo de memória;
2. o mesmo MoE com **LRU de experts** — o adversário principal, não só “qualquer MoE”;
3. um modelo denso com parâmetros ativos comparáveis?

Pergunta secundária (dose–resposta): treinado sob orçamento `B` de referência, o roteador reage de forma monótona em **0,5×`B`** e **2×`B`** na avaliação? (`λ` define curvas de treino separadas.)

## Experimento mínimo (entrega principal)

Um único emulador de três níveis, **um** vetor de custo (perfil consumer ~6 GB, proporções tipo HBM ≫ DDR ≫ NVMe), `k=2`, orçamento `B=25%` dos experts, treino por `λ ∈ {0, 0,5, 1}` (o ponto `λ=0` é o baseline 1) e avaliação em 0,5× e 2×`B`, contra top-k e **LRU** (e denso). Sem catálogo de várias GPUs de mercado no núcleo.

Extensões (não bloqueiam a entrega mínima):

- tabelas de custo “realismo de mercado” (consumer 6–12 GB, mid cloud, high-end);
- treino com orçamento variável por batch/época (**adiado**);
- calibrar pesos em preços Spot reais.

## Decisões fechadas neste blueprint

| Tema | Decisão |
| --- | --- |
| Corpus | TinyStories só no dev; WikiText-2 no treino/avaliação de entrega (overnight) |
| Experts / `k` | `E` ∈ 16–32; **`k = 2` no mínimo**. Preferir **aumentar o tamanho do expert** a encolher o modelo se faltar capacidade |
| Orçamento `B` | **25%** dos experts quentes (4/16 ou 8/32) — teto **artificial** (o modelo inteiro cabe na 1060) |
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
| Custos por nível | Partida nos configs: c=(1,10,20|30), w=Δc; HBM ≫ DDR ≫ NVMe |

## Motivação (curta)

MoE só calcula parte dos parâmetros por token, mas o tempo real depende de *onde* o expert está. PowerInfer e *LLM in a flash* exploram offload; o recorte daqui é **treinar o roteador com esse custo**. Detalhe e fichas em [docs/referencias.md](docs/referencias.md).

## Arquitetura em três níveis (emulada)

| Nível | Papel no emulador | Custo relativo (ordem) |
| --- | --- | --- |
| Quente | “VRAM” — capacidade cara, acesso barato | baixo `c(ℓ)`, alto preço de residência no orçamento |
| Morno | “RAM” — meio do caminho | médio |
| Frio | “SSD” — capacidade barata, troca cara | alto `c(ℓ)` e `w` de promoção |

Os pesos `c(ℓ)` e `w(a,b)` são variáveis de experimento: mudar a tabela (menos VRAM “barata”) deve forçar o roteador a adaptar. Fórmulas em [docs/funcao-de-custo.md](docs/funcao-de-custo.md). Hardware e emulação em [docs/plano-de-hardware.md](docs/plano-de-hardware.md).

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

**Em construção.** Ainda não há ambiente, script de treino nem de avaliação. Código previsto em PyTorch.

| Pasta | Para que serve |
| --- | --- |
| `src/` | Modelo, roteador, emulador de níveis e loss |
| `experiments/` | Treino, avaliação, ablação e varredura de λ / orçamento |
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
