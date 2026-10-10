# Protocolo experimental

Comparar quatro modelos no **mesmo corpus** e no **mesmo orçamento de VRAM**, e ver se a penalidade de memória move a fronteira entre qualidade e latência. Nada disto foi executado. Onde um número ainda não foi escolhido, o texto diz **a definir**. Não há meta numérica de perplexidade nem de latência.

## O que fica igual

- Corpus de modelagem de linguagem, pequeno e público. Candidatos: WikiText-2 ou TinyStories. Escolha final: **a definir**.
- Split de hold-out temporal ou por domínio, no mesmo corpus. Forma exata: **a definir**.
- Família do modelo de linguagem e faixa de tamanho (~10 a 50 milhões de parâmetros no MoE, contando todos os experts). Tamanho exato, número de experts, `k` do top-k e número de camadas MoE: **a definir**.
- Orçamento de VRAM do experimento principal: um teto de quantos experts (ou quantos bytes de expert) podem residir na VRAM. O teto é menor que o conjunto inteiro de experts, senão ninguém usa RAM nem SSD. Valor: **a definir**.
- Custos `c(ℓ)` e pesos de troca, medidos uma vez na máquina do grupo e reutilizados no treino. Ver a função de custo e o plano de hardware.
- Implementação em PyTorch. Os quatro modelos compartilham o máximo possível de código, para a diferença ser a política e não um detalhe de treino.

O modelo denso não tem experts para expulsar da VRAM. “Parâmetros ativos comparáveis” quer dizer: a contagem de parâmetros do denso fica na mesma ordem do que o MoE usa por token (tronco compartilhado + `k` experts), não na ordem do total de experts. A receita exata de igualar essa contagem está **a definir**. Por construção, esse denso deve caber no orçamento quente. Igualar também FLOPs é desejável como checagem, não como segundo baseline.

## Os quatro modelos

| # | Nome | Treino do roteador | Onde os experts ficam |
| --- | --- | --- | --- |
| 1 | MoE top-k clássico | `λ = 0`. Tarefa + balanceamento | Sem política de memória. Referência de qualidade do MoE, sem cobrar nível |
| 2 | MoE com offload heurístico | O mesmo roteador do modelo 1, sem penalidade | Heurística: LRU **ou** manter na VRAM os k experts mais usados. São regras diferentes; a escolhida fica **a definir** e deve ser declarada |
| 3 | Denso | Não há roteador | Todos os parâmetros no caminho ativo, tamanho comparável ao caminho ativo do MoE |
| 4 | Este projeto | `L` completa, com `λ > 0` e a variante de troca | Residência em três níveis. A política que promove e rebaixa experts está **a definir**; o gradiente da penalidade passa pelo roteador |

O modelo 1 existe para separar “MoE esparso” de “MoE que paga memória”. O modelo 2 existe para separar “qualquer offload” de “offload que o roteador aprendeu”. O modelo 3 existe para separar “esparsidade” de “a penalidade”.

## Métricas

Todas no hold-out, com a escolha discreta de experts (não com o custo esperado da loss).

| Métrica | Definição operacional |
| --- | --- |
| Perplexidade | Perplexidade do próximo token no hold-out. Accuracy de próximo token pode ser registrada junto, como leitura auxiliar |
| Latência por token | Tempo de parede por token. Reportar p50 e p95, não só a média |
| Volume de I/O | Bytes lidos da RAM e do SSD, e bytes escritos no SSD se o treino ou a promoção sujar peso frio |
| Trocas por nível | Contagem de transições VRAM↔RAM e RAM↔SSD (e VRAM↔SSD, se existirem) |
| Curva qualidade × latência | Perplexidade contra latência p50. No modelo 4, um ponto por `λ`. Nos outros, um ponto no orçamento principal; orçamentos extras entram na pergunta secundária |

O número de corridas independentes e o comprimento mínimo do hold-out para o p95 não ser anedota estão **a definir**. A semente de cada corrida será registrada quando o código existir.

Latência e I/O que entram nessa tabela são medidos na máquina do grupo. Medição em Colab ou Kaggle não substitui essa coluna.

## Ablação da penalidade

Obrigatória, no modelo 4, com a mesma arquitetura e o mesmo orçamento:

- `λ = 0`
- `λ > 0` (os valores da grade, **a definir**)

A diferença de perplexidade, latência, I/O e trocas entre esses treinos é o efeito atribuível à penalidade. Se `λ = 0` não chegar perto do modelo 1, há um confounder (código ou residência diferentes), e isso tem de ser explicado, não absorvido como ganho.

Ablação fina opcional, não exigida para fechar o protocolo: `C_mem` só com busca, contra busca mais troca.

## Generalização entre orçamentos

Treinar o modelo 4 com um orçamento `B`. Avaliar em `0,5 · B` e em `2 · B`, sem retreinar. Comparar, nos três orçamentos, com o top-k clássico (modelo 1) e com a heurística (modelo 2). O denso nesse par de orçamentos é opcional.

O que se quer ver é se o roteador aprendido em `B` só memorizou aquele teto. Não há previsão numérica do que vai acontecer.

## Extensão opcional

Sortear o orçamento a cada batch ou a cada época, informar o valor ao roteador (ele precisa enxergar o teto, senão não tem como se adaptar), e repetir o teste em `0,5 · B` e `2 · B`. Comparar com o roteador de orçamento fixo. Distribuição do sorteio: **a definir**. Esta extensão não bloqueia o experimento mínimo.

## Modo da camada fria

**A definir:** só simulação (tabela de custos medida, sem cópia real) ou também protótipo com arquivo no NVMe via mmap. O relatório declara qual modo produziu cada número. Se os dois existirem, não misturar a latência simulada com a latência de parede na mesma curva sem um rótulo.

## Ordem prática

1. Fechar corpus, arquitetura e orçamento `B`.
2. Medir `c(VRAM)`, `c(RAM)` e `c(SSD)` na máquina do grupo, para o tamanho real de um expert.
3. Treinar os quatro modelos nesse `B`.
4. Avaliar perplexidade e, na mesma máquina, latência, I/O e trocas.
5. Repetir a avaliação do modelo 4 em `λ = 0` e na grade `λ > 0`.
6. Reavaliar modelos 1, 2 e 4 em `0,5 · B` e `2 · B`.
7. Só então, se o grupo adotar a extensão, treinar com orçamento variável.

Não pular para a curva antes de existir um modelo 1 reproduzível. Sem esse ponto, a fronteira não tem âncora.
