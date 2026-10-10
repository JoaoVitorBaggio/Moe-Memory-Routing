# Roteamento consciente de memória

Mixture of Experts (MoE) pequeno em três níveis — VRAM, RAM e SSD — com o roteador treinado por um custo explícito de memória e de troca. Trabalho final de **Aprendizado Profundo (AGL11157)**, UFRGS, 2026/2, em grupo de 4 pessoas.

**Status:** documentação inicial. Código, treino, experimentos e resultados estão **em construção**. Não há número de desempenho neste repositório.

## Resumo

Cada expert fica num nível de memória: quente (VRAM da GPU), morno (RAM) ou frio (SSD NVMe). A loss junta a entropia cruzada do próximo token, o custo esperado de buscar e de trocar de nível, e o balanceamento de carga do MoE. A hipótese é que variar esse custo melhora a fronteira entre qualidade e latência, no mesmo corpus e no mesmo orçamento de VRAM, frente aos baselines abaixo.

## Pergunta de pesquisa

Um roteador treinado com penalidade de memória e de troca melhora a fronteira qualidade × latência (e o volume de I/O) em relação a um MoE top-k sem custo de memória, ao mesmo MoE com offload heurístico, e a um modelo denso com parâmetros ativos comparáveis?

Pergunta secundária: treinado com um orçamento de VRAM, o roteador se mantém quando o teste usa 0,5× e 2× desse orçamento? Extensão opcional, ainda não decidida como entrega: sortear o orçamento a cada batch ou época e informá-lo ao roteador.

## Motivação e trabalhos relacionados

Um MoE só calcula parte dos parâmetros por token, mas o tempo real depende de onde o expert está. VRAM é rápida e escassa; RAM fica no meio; SSD é barata, lenta e tem limite de escritas. PowerInfer e *LLM in a flash* já exploram offload e ativação esparsa. No recorte deste trabalho, o que falta é **treinar o roteador com esse custo**, em vez de otimizar só a loss de tarefa.

- Shazeer et al., 2017 — camada MoE com roteamento esparso.
- Fedus et al., 2022 (*Switch Transformers*) — roteamento e a loss de balanceamento usada aqui.
- Alizadeh et al., *LLM in a flash* — inferência com pouca memória e pesos em flash.
- Song et al., *PowerInfer* — servir um modelo de linguagem numa GPU de consumidor.

Fichas e links estão em [docs/referencias.md](docs/referencias.md).

## Arquitetura em três níveis

| Nível | Onde vive | Papel |
| --- | --- | --- |
| Quente | VRAM da GPU | Acesso imediato, capacidade cara |
| Morno | RAM em memória pinned | Reserva sem ocupar a VRAM |
| Frio | Arquivo no SSD NVMe, lido com mmap | Capacidade barata; troca mais cara |

O controlador entre SSD e RAM é mais lento e mais estável que o entre RAM e VRAM, para respeitar o SSD. A escolha do expert é discreta; no treino o custo entra como valor **esperado** sob o softmax do roteador, que é diferenciável. Fórmulas em [docs/funcao-de-custo.md](docs/funcao-de-custo.md).

O orçamento de VRAM do experimento é um teto artificial de residência dos experts, menor que o modelo inteiro. Na faixa de ~10 a 50 milhões de parâmetros, o modelo cabe na GTX 1060; sem esse teto a hierarquia não aparece. Detalhe em [docs/plano-de-hardware.md](docs/plano-de-hardware.md).

## Comparação com os baselines

Os quatro modelos usam o mesmo corpus pequeno e o mesmo orçamento de VRAM. Corpus final: **a definir** (WikiText-2 ou TinyStories).

| # | Modelo | O que muda |
| --- | --- | --- |
| 1 | MoE top-k clássico | Sem custo de memória |
| 2 | O mesmo MoE | Offload heurístico (LRU ou os k mais usados na VRAM; regra exata **a definir**) |
| 3 | Modelo denso | Parâmetros ativos comparáveis aos do MoE |
| 4 | Este projeto | Roteador treinado com a penalidade |

Métricas: perplexidade; latência por token (p50 e p95); volume de I/O; trocas por nível; curva qualidade × latência. Ablação do modelo 4: peso da penalidade zero versus positivo. Protocolo em [docs/protocolo-experimental.md](docs/protocolo-experimental.md).

## Como contribuir e rodar

**Em construção.** Ainda não há ambiente, script de treino nem de avaliação. O código será em PyTorch, sem TensorFlow. Quem não tiver GPU pode treinar no Colab ou no Kaggle; a latência que entra no resultado tem de ser medida numa máquina controlada pelo grupo, porque disco de nuvem gratuita é virtual.

| Pasta | Para que serve |
| --- | --- |
| `src/` | Modelo, roteador, níveis de memória e loss |
| `experiments/` | Treino, avaliação, ablação e varredura de λ |
| `configs/` | Orçamento, custos medidos e hiperparâmetros |
| `notebooks/` | Exploração; saída grande fica de fora do Git |
| `results/` | Tabelas e curvas, quando existirem |

Documentação nova em português do Brasil. Não versionar checkpoint, dataset baixado nem resultado grande (veja `.gitignore`).

## Integrantes

- João Vítor Baggio — cartão: **a definir**
- Nome: **a definir** — cartão: **a definir**
- Nome: **a definir** — cartão: **a definir**
- Nome: **a definir** — cartão: **a definir**

## Documentação

- [Problema e escopo](docs/definicao.md)
- [Função de custo](docs/funcao-de-custo.md)
- [Protocolo experimental](docs/protocolo-experimental.md)
- [Plano de hardware](docs/plano-de-hardware.md)
- [Referências](docs/referencias.md)
