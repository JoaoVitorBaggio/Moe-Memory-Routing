# Referências

Só as quatro obras-âncora do rascunho da definição. Não há citação extra e não há número de desempenho deste projeto.

## Shazeer et al., 2017

Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.  
<https://arxiv.org/abs/1701.06538>

Define a camada MoE com roteamento esparso: vários experts, e uma rede de roteamento escolhe uma combinação esparsa por exemplo. É a base do modelo 1 (top-k clássico, sem custo de memória) e da arquitetura esparsa dos outros MoE deste trabalho.

## Fedus et al., 2022

Fedus et al. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022.  
<https://jmlr.org/papers/v23/21-0998.html>

Simplifica o roteamento e introduz a loss auxiliar de balanceamento que este projeto copia na forma `E · Σ_e f(e) · P(e)`. O coeficiente que o artigo usa não é o `β` daqui; a adaptação está em [funcao-de-custo.md](funcao-de-custo.md). O Switch também é top-1; se o nosso `k` for maior, a definição de `f(e)` continua em aberto.

## Alizadeh et al.

Alizadeh et al. *LLM in a flash: Efficient Large Language Model Inference with Limited Memory.* arXiv:2312.11514.  
<https://arxiv.org/abs/2312.11514>

Inferência de modelo de linguagem quando a memória é o limite e os pesos podem ficar em armazenamento flash. Entra como contexto de offload para um nível frio, não como sistema que vamos reproduzir. A diferença do nosso recorte: o custo do nível entra no treino do roteador.

## Song et al.

Song et al. *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.* arXiv:2312.12456.  
<https://arxiv.org/abs/2312.12456>

Serviço de modelo de linguagem numa GPU de consumidor, explorando ativação esparsa e a diferença entre o que está quente e o que está frio. Mesma ressalva: é motivação de sistemas, não baseline implementado. Os baselines deste trabalho são os quatro modelos do protocolo, em escala de estudante.

## O que estas obras não fecham

Nenhuma delas é o experimento do grupo. Elas não escolhem o corpus, não medem a GTX 1060 deste projeto e não dizem o valor de `λ`. Esses itens seguem **a definir** em [definicao.md](definicao.md), [protocolo-experimental.md](protocolo-experimental.md) e [plano-de-hardware.md](plano-de-hardware.md).
