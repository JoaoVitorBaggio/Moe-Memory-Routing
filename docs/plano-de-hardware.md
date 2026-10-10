# Plano de hardware

A máquina de um dos integrantes é a referência dos três níveis reais. Treino pode acontecer em outra GPU. A latência que o protocolo reporta, não.

Não há medição neste repositório. Tudo que seria um número de milissegundo continua **a definir**.

## Máquina de referência

| Peça | O que o grupo tem |
| --- | --- |
| GPU | GTX 1060, 6 GB de VRAM, arquitetura Pascal |
| RAM | 16 GB |
| Disco frio | SSD NVMe |
| Software previsto | PyTorch. Sem TensorFlow |

Essa GPU não tem tensor cores. O cálculo fica em **fp32**. fp16 entra só se for preciso economizar memória, não como aceleração: nesta placa ele não é um caminho rápido de treino. bf16 nativo não existe aqui. `torch.compile` costuma não funcionar nesta GPU; o projeto não depende dele. Se alguém compilar numa GPU mais nova, isso não pode ser requisito do resultado.

A faixa de modelo é ~10 a 50 milhões de parâmetros. O tamanho exato, dentro da faixa, está **a definir**.

## Por que o orçamento de VRAM é artificial

Em fp32, 50 milhões de parâmetros são cerca de 200 MB de pesos, mais o estado do otimizador se o treino for nesta placa (o Adam guarda dois momentos por parâmetro). Isso cabe em 6 GB, com folga para ativações, desde que o modelo fique na faixa combinada. Confiança alta nessa conta de ordem de grandeza; não é uma medição de pico de memória desta GPU.

Consequência: se todos os experts puderem morar na VRAM, a hierarquia não é exercitada e a penalidade não tem o que cobrar. O orçamento `B` do protocolo é um teto de residência, menor que o conjunto de experts, escolhido pelo grupo. Os 6 GB são o teto físico, incluindo ativações e, no treino local, o otimizador. Quanto de `B` sobra para pesos quentes versus ativações está **a definir**.

## Os três níveis na prática

**Quente.** Tensores na VRAM. É o único nível que a GPU lê sem transferência no meio do passo.

**Morno.** Pesos na RAM do hospedeiro, em memória pinned (memória travada, que o DMA da GPU pode ler sem o sistema operacional paginar a página no meio da cópia). Sobe para a VRAM quando o controlador promove o expert.

**Frio.** Pesos num arquivo no NVMe, mapeados com `mmap` / `np.memmap`. Não ficam todos residentes na RAM. Uma promoção para o nível morno lê o arquivo.

O controlador SSD↔RAM troca menos vezes e com penalidade maior do que o RAM↔VRAM. Isso está na função de custo (`w_frio > w_quente`).

## Simulação ou arquivo de verdade

**A definir.** As duas opções cabem no plano; o grupo ainda não fechou.

- Simulação: cada acesso soma o `c(ℓ)` medido (ou sorteia em torno dele) e nenhum byte precisa atravessar o NVMe durante o treino. Serve para iterar a loss. Não substitui a coluna de latência de parede.
- Protótipo: arquivo real, mmap, cópia para RAM pinned e então para a VRAM. A latência p50/p95 e o volume de I/O saem desse caminho.

Os custos usados na loss devem ser medidos nesta máquina, com uma leitura do tamanho real de um expert em cada nível, antes da grade de `λ`. Repetir a medição se o tamanho do expert mudar.

## Quem treina onde

Quem não tiver GPU treina no Colab ou no Kaggle. Isso vale para obter pesos e perplexidade, se o grupo quiser paralelizar.

Não vale para a latência, o I/O e a contagem de trocas que entram na curva. Disco de instância gratuita é virtual: a fila e a cache não são as do NVMe da máquina de referência, então p50 e p95 não se comparam. Essas métricas saem de uma máquina controlada pelo grupo, de preferência a descrita acima, com GPU, RAM e SSD no mesmo sistema.

Se a perplexidade for medida numa GPU e a latência noutra, o relatório separa as colunas e diz em qual máquina cada uma foi obtida. Misturar as duas numa “latência do Colab” invalida a fronteira.

## Cuidados que já dá para cravar

- Não planejar o resultado em cima de bf16 nem de `torch.compile`.
- Não encher os 6 GB com o conjunto inteiro de experts e chamar isso de orçamento apertado.
- No treino feito na própria 1060, reservar memória para ativações e para o estado do otimizador antes de fixar quantos experts cabem quentes.
- Checkpoint grande, dataset baixado e log de ativação ficam fora do Git (veja `.gitignore`). Disco local é o lugar deles; a pasta some se alguém depender de filesystem efêmero de nuvem.
