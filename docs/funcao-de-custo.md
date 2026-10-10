# Função de custo

A loss de treino do roteador é

```text
L = L_tarefa + λ · C_mem + β · L_balanço
```

`L_tarefa` empurra o modelo a prever o próximo token. `C_mem` cobra o nível de memória dos experts que o roteador quer usar. `L_balanço` impede que o roteador resolva o problema escolhendo sempre os experts que já estão na VRAM. `λ` é o peso que se varia para traçar a curva qualidade × latência. `β` segura o balanceamento; o valor numérico está **a definir**.

Não há resultado de treino. Os custos `c(VRAM)`, `c(RAM)` e `c(SSD)` também estão **a definir**: saem de latência medida na máquina do grupo, não de um chute.

## Notação

| Símbolo | Significado |
| --- | --- |
| `T` | Número de tokens do batch |
| `E` | Número de experts da camada |
| `z(t, e)` | Logit do roteador para o expert `e` no token `t` |
| `p(t, e)` | Probabilidade do softmax do roteador |
| `ℓ(e)` | Nível atual do expert: VRAM, RAM ou SSD |
| `c(ℓ)` | Custo de buscar um expert que já está no nível `ℓ` |
| `w(a, b)` | Penalidade de trocar do nível `a` para o nível `b` entre passos |
| `λ`, `β` | Pesos da loss. `λ` é o eixo da curva; `β` fica fixo numa corrida |

A média por token, e não a soma, é de propósito: assim `λ` não muda de significado quando o batch cresce.

## Por que o custo é esperado

Na inferência o expert é uma escolha discreta (top-k). Esse indicador não tem gradiente. No treino, o custo entra pela distribuição do roteador, que é diferenciável:

```text
p(t, e) = exp(z(t, e)) / Σ_j exp(z(t, j))
```

A soma é sobre os `E` experts. Métricas de avaliação (latência, I/O, trocas) usam a escolha discreta. A loss usa o substituto esperado. Os dois não são o mesmo número; o relatório, quando existir, precisa dizer qual está em qual coluna.

`Σ_e p(t, e) · c(ℓ(e))` é o custo esperado de **um** expert sorteado pelo softmax. Isso coincide com a inferência quando `k = 1`. Se a inferência busca `k` experts, o custo realizado é a soma dos `k` acessos. Com `k` fixo, multiplicar o substituto por `k` só reescala `λ`, então a loss pode continuar na média ponderada acima. A contagem de I/O na avaliação soma os `k` de verdade. Para `k > 1`, a troca de nível entre conjuntos (em vez de entre um par de experts) fica **a definir**, no mesmo ponto em aberto da definição de `f(e)` mais abaixo.

## L_tarefa

Entropia cruzada do próximo token, média no batch:

```text
L_tarefa = − (1 / T) · Σ_t log p_modelo(x_t | x_<t)
```

`p_modelo` é a distribuição preditiva da rede, não a do roteador.

## C_mem: custo de busca

Forma base, sem troca entre passos. Cada expert já tem um nível, e buscar quem está mais frio custa mais:

```text
c(VRAM) < c(RAM) < c(SSD)

C_busca(t) = Σ_e p(t, e) · c(ℓ(e))

C_mem = (1 / T) · Σ_t C_busca(t)
```

Unidade: tempo medido para ler os pesos de um expert naquele nível (a medição está no plano de hardware). Normalizar, por exemplo dividindo por `c(VRAM)`, é permitido, desde que a mesma convenção valha para todas as corridas. Os três números concretos ficam **a definir** depois da medição. Não usar latência de disco do Colab ou do Kaggle.

## Variante: penalidade de troca entre passos

A forma acima cobra onde o expert está. A variante cobra também a **troca de nível de um passo para o outro**, com peso maior em SSD↔RAM do que em RAM↔VRAM, para respeitar o SSD.

```text
w(a, a) = 0
w(RAM, VRAM) = w(VRAM, RAM) = w_quente
w(SSD, RAM) = w(RAM, SSD) = w_frio

w_frio > w_quente > 0
```

O salto direto VRAM↔SSD está **a definir**. Proposta deste documento, ainda não fechada pelo grupo: tratar como a soma dos dois saltos (`w_frio + w_quente`), para um atalho não sair mais barato do que passar pela RAM.

O custo esperado da troca no token `t` usa as duas distribuições do roteador. `sg` é stop-gradient: o passo anterior já foi decidido, e o gradiente ajusta a escolha atual.

```text
C_troca(t) = Σ_e Σ_e' sg( p(t−1, e') ) · p(t, e) · w( ℓ(e'), ℓ(e) )

C_troca = (1 / T) · Σ_t C_troca(t)
```

No primeiro token do trecho não há passo anterior: `C_troca(t)` vale 0.

Nesta variante:

```text
C_mem = C_busca + C_troca
```

O modelo 4, o deste projeto, usa essa variante. Zerar só `C_troca` e manter `C_busca` é uma ablação fina opcional. A ablação obrigatória é outra: `λ = 0` contra `λ > 0`, no protocolo experimental.

`w_quente` e `w_frio` estão **a definir**. A única ordem imposta aqui é `w_frio > w_quente > 0`. Faz sentido medi-los na mesma unidade de `c(ℓ)`, na mesma máquina.

Leitura e escrita não são o mesmo custo. Na inferência, subir um expert do SSD para a RAM é leitura no NVMe. No treino, um expert frio que recebeu gradiente e depois é rebaixado volta a ser escrito no SSD. A penalidade maior em SSD↔RAM existe para as duas coisas: latência e limite de escritas. Se o protótipo real distinguir leitura e escrita, os pesos deixam de ser simétricos; isso fica **a definir** junto com a decisão “só simulação ou também arquivo”.

## L_balanço

Sem este termo, `λ · C_mem` tem um atalho: pôr probabilidade só nos experts que já estão na VRAM. Eles recebem todo o gradiente, o roteador fecha, e os experts frios nunca têm chance de reduzir a loss de tarefa. O balanceamento, como no Switch Transformer (Fedus et al., 2022), empurra o roteador a usar o conjunto de experts.

Para uma camada, com `E` experts e um batch de `T` tokens:

```text
f(e) = (1 / T) · Σ_t 1[ o expert e foi o escolhido no token t ]
P(e) = (1 / T) · Σ_t p(t, e)

L_balanço = E · Σ_e f(e) · P(e)
```

`f(e)` é a fração de tokens efetivamente despachada ao expert (indicador discreto, sem gradiente). `P(e)` é a fração média da probabilidade do softmax (por aqui passa o gradiente). O fator `E` mantém a escala quando o número de experts muda: sob roteamento uniforme, `Σ_e f(e) · P(e)` vale `1/E`, e o produto por `E` fica em 1.

O artigo do Switch usa um coeficiente `α` multiplicando essa expressão, e no trabalho deles `α = 10⁻²`. Aqui esse papel é o de `β`, fora da definição de `L_balanço`, para a loss continuar na forma do início desta página. O `10⁻²` é escolha daquele artigo, não um default deste projeto: a nossa loss ainda tem `λ · C_mem`, então a escala muda. `β` fica **a definir**.

O Switch despacha um expert por token (`argmax`). Se neste projeto `k > 1`, a definição de `f(e)` — fração de tokens em que o expert entrou no top-k, e se essa fração é dividida por `k` — fica **a definir** na implementação. A forma `E · Σ_e f(e) · P(e)` permanece.

Se houver mais de uma camada MoE, soma-se um `L_balanço` por camada. Quantas camadas: **a definir**.

## O que `λ` faz

`λ = 0` recupera treino de MoE com tarefa e balanceamento, sem pressão de memória. `λ` maior aceita pior perplexidade para gastar menos memória e menos troca. A curva qualidade × latência é o conjunto de pontos obtidos variando `λ` no mesmo orçamento. A grade de valores está **a definir**. Nenhum `λ` é declarado vencedor antes do experimento.

## O que não entra nesta loss

Perplexidade, latência p50/p95, volume de I/O e contagem de trocas são métricas de avaliação, não termos de `L`. Orçamento médio de VRAM pode virar restrição do controlador de residência; virar mais um termo de loss está **a definir** e não faz parte da fórmula acima.
