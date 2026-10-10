# Problema e escopo

**Disciplina:** Aprendizado Profundo (AGL11157), UFRGS, 2026/2.  
**Grupo:** 4 pessoas. Nomes e cartões que faltam estão **a definir** (veja o README).  
**Estado:** definição de trabalho. Nenhum experimento foi executado. Não há resultado numérico.

## Problema

Modelos esparsos do tipo Mixture of Experts (MoE) ativam só parte dos parâmetros por token. Em hardware real, esses experts não custam o mesmo: moram numa hierarquia de memória. VRAM é rápida e cara, RAM é o meio do caminho, SSD é lenta, barata e tem limite de escritas.

Sistemas como PowerInfer e *LLM in a flash* já fazem offload e ativação esparsa. No recorte deste grupo, o que não está resolvido é **treinar o roteador com um custo explícito de troca entre níveis**, em vez de minimizar só a loss de tarefa e deixar a residência a cargo de uma heurística depois do treino.

Cada expert (ou bloco) fica em um de três estados:

| Estado | Nível | Residência |
| --- | --- | --- |
| Quente | VRAM | GPU |
| Morno | RAM | Memória pinned do hospedeiro |
| Frio | SSD | Arquivo no NVMe, lido com mmap |

O controlador SSD↔RAM deve ser mais lento e mais estável que o RAM↔VRAM. A penalidade de troca é maior nesse salto justamente para não martelar o SSD.

## Pergunta de pesquisa

Um roteador treinado com penalidade por acesso e por troca de nível melhora a fronteira qualidade × latência (e qualidade × I/O) em relação a:

1. um MoE top-k clássico, sem custo de memória;
2. a mesma rede com offload heurístico (LRU ou “k experts mais usados na VRAM”);
3. um modelo denso com quantidade comparável de parâmetros ativos.

Pergunta secundária, de generalização entre orçamentos: se o treino usa um orçamento de VRAM fixo, o roteador ainda se comporta bem em 0,5× e 2× desse orçamento, contra o top-k clássico e contra a heurística?

Extensão opcional: sortear o orçamento por batch ou por época, informar esse valor ao roteador, e comparar com o treino de orçamento fixo. Entra ou não na entrega mínima: **a definir**.

## O que conta como sinal

Não há dataset rotulado externo. O sinal de treino é o próximo token, mais métricas de recurso (latência, bytes movidos, trocas). Um registro auxiliar de quais experts seriam pedidos por token pode calibrar o modelo de latência e as taxas de miss entre níveis. No protótipo, esse registro pode ser sintético.

## Corpus

Escala de estudante, corpus público e compacto, para treinar e medir perplexidade com GPU de estudante ou Colab.

Candidatos: **WikiText-2** ou **TinyStories** (ou um subconjunto equivalente). A escolha final está **a definir**.

Hold-out: split temporal ou por domínio, no mesmo corpus, para ver se o roteador com custo generaliza ou só memoriza o padrão de acesso do treino. A forma exata do split fica **a definir** junto com o corpus.

## Dentro do escopo

- MoE pequeno, poucos experts, top-k baixo. Contagens exatas: **a definir**.
- Três níveis de residência.
- Loss de tarefa + penalidade de memória/troca + balanceamento de carga.
- Quatro sistemas comparados sob o mesmo corpus e o mesmo orçamento de VRAM.
- Ablação do peso da penalidade (zero versus positivo).
- Varredura que produz a curva qualidade × latência.
- Teste em 0,5× e 2× do orçamento de treino.
- Implementação em PyTorch. Modelos na faixa de ~10 a 50 milhões de parâmetros.
- Simulador da hierarquia com custos medidos na máquina do grupo, e/ou protótipo com arquivo real. Qual dos dois (ou ambos) está **a definir**.

## Fora do escopo

- Escala de modelo de linguagem de produção, cluster e serviço online.
- TensorFlow.
- BitNet ou quantização como entrega. Podem aparecer só como discussão de atrito com o hardware.
- Depender de `torch.compile` ou de bf16. A GPU de referência não oferece isso de forma confiável (veja o plano de hardware).
- Tratar latência medida em disco de Colab ou Kaggle como resultado. Esse disco é virtual.

## Relação com a disciplina

O trabalho usa rede profunda, loss composta e avaliação experimental. Liga computação condicional / MoE a um problema de sistemas e exige comparação controlada. O foco é o protocolo (custo no roteador + baselines), não bater recorde de escala.

## Itens em aberto

- Nomes e cartões dos integrantes.
- Corpus final (WikiText-2, TinyStories ou outro equivalente).
- Camada fria só simulada, ou também protótipo com arquivo mmap no NVMe.
- Regra exata da heurística do baseline 2 (LRU estrito ou k de maior uso).
- Número de experts, valor de k, tamanho exato do modelo dentro da faixa, e o orçamento de VRAM em bytes ou em número de experts.
- Se a extensão de orçamento variável entra na entrega ou fica de fora.

Nada disso será preenchido com um número inventado. Quando o grupo fechar um item, a atualização entra neste documento e no protocolo.
