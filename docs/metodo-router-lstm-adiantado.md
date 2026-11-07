# Roteador LSTM adiantado

Este desenho substitui o roteador do OLMoE por um LSTM que lê só o fluxo de tokens. Fica ao lado do experimento dos quatro braços: não altera `configs/default.yaml`, não muda a loss com `λ` e não destila o roteador original. Não há número medido neste texto. `L`, `K` e o tamanho do estado do LSTM saem da máquina, na hora da corrida, pelo `scripts/hw_microbench.py`. O módulo `src/moe_memory/lstm_ahead_router.py` é um esqueleto. Ele fixa a forma do cálculo, com dimensão minúscula no teste, sem peso de OLMoE e sem laço de treino.

## O que corre no lugar do roteador

O LSTM consome a sequência de ids de token, passa por uma embedding de token e atualiza o próprio estado. O estado oculto do Transformer não entra nessa recorrência. No passo `t` a rede emite, para cada camada, os logits dos experts da posição `t+L`. Essa saída é o roteador da posição futura. O roteador original do OLMoE não roda.

Um único LSTM produz os logits de todas as camadas. A cabeça que separa as camadas é linear, para o custo recorrente continuar sendo o de uma sequência só. `L` é o deslocamento entre o passo em que a decisão sai e a posição que vai consumir o expert. O número de parâmetros não cresce com `L`.

O OLMoE público `allenai/OLMoE-1B-7B-0924` tem esparsidade fixa. O `config.json` desse checkpoint, lido sem os pesos, traz `num_experts` 64, `num_experts_per_tok` 8 e `num_hidden_layers` 16. Cada token executa 8 experts dentre 64, em cada camada. Isso é estrutura do modelo. Este método não varre essa esparsidade.

## Treino

A loss de treino é a de linguagem, e só ela. Não há divergência contra o roteador antigo e não há loss auxiliar de roteador.

O top-K escolhe quais experts são carregados para a memória quente. A mistura é mole: a saída de cada expert carregado é multiplicada pela probabilidade que o LSTM atribuiu a ele, a softmax sobre os experts da camada. Essa probabilidade não é renormalizada dentro do top-K. A massa dos experts que ficaram de fora sai da soma, então a soma dos pesos da mistura pode ser menor que um e a escala do residual acompanha essa massa. No config citado, `norm_topk_prob` é falso: o bloco original desse checkpoint também não renormaliza o top-K. O esqueleto segue essa regra de propósito, porque o top-K daqui só decide o que carregar.

A projeção final nasce com pesos de desvio pequeno e bias zero, para a softmax começar perto da uniforme. O valor do desvio está em `HEAD_INIT_STD`, no módulo. É constante do esqueleto, não cota de hardware.

Os experts foram treinados com o roteador original. Com outro roteador eles precisam adaptar, por fine-tuning ou por LoRA. O esqueleto não implementa nenhum dos dois.

## De onde saem L e K

Os símbolos abaixo são medidos ou entregues pela máquina. Este texto não escolhe valor para eles.

`S_e` é o tamanho em bytes de um expert. No OLMoE cada expert é um SwiGLU sem bias: as projeções `gate`, `up` e `down`, ou as mesmas contas fundidas em `gate_up` e `down`. Nos dois layouts o número de parâmetros é `3 * hidden_size * intermediate_size`. No config citado, `hidden_size` é 2048, `intermediate_size` é 1024, `hidden_act` é `silu` e `torch_dtype` é `bfloat16` (2 bytes por parâmetro). Ficam de fora o roteador linear, as normas e a atenção: não são o corpo do expert que sobe para a memória quente. Com esses campos, `S_e = 3 * 2048 * 1024 * 2 = 12582912` bytes. O script refaz a conta a partir do config. A igualdade aqui é aritmética do config, não uma medição de banda. O config tem um único `intermediate_size`, então o script repete esse `S_e` em todas as camadas e avisa isso.

`BW` é a banda sustentada de host para GPU, com a origem em memória pinned. `t_tok` é o tempo de um token. `m` é quantos experts ainda precisam atravessar o barramento. `M_hot` é o orçamento de memória quente que sobra para corpos de expert, já líquido dos descontos do parágrafo dos limites.

O adiantamento mínimo, em tokens, é o menor inteiro `L` tal que `L >= ceil(m * S_e / (BW * t_tok))`.

A memória limita quantos experts ficam quentes ao mesmo tempo. Somando as camadas, `sum_l K_l <= M_hot / S_e`.

A banda no horizonte de `L` tokens limita quantos corpos cabem na transferência: `K * S_e <= BW * L * t_tok`.

O `K` das duas linhas é a mesma quantidade: o total de experts quentes, já somado nas camadas. Cada camada do OLMoE tem os próprios parâmetros, então o expert 3 da camada 0 e o expert 3 da camada 1 ocupam `S_e` cada um. O script imprime o maior inteiro de cada cota e o compara com `8 * 16`, o total se as dezesseis camadas mantiverem os oito experts executados ao mesmo tempo e não houver reuso entre camadas. Reuso de um expert, de um token para o outro, é o que faz o conjunto de trabalho caber nessa cota. A fórmula não calcula o reuso.

O LSTM roda em paralelo com o token e precisa terminar dentro de `t_tok`. Esse prazo limita o tamanho do estado oculto. O limite se mede na máquina em que o passo cabe. O esqueleto recebe o tamanho do estado como argumento e não traz um valor de hardware.

O esquecimento do histórico de tokens fica nos portões do LSTM. O esqueleto deixa a inicialização do LSTM como a do PyTorch e não força o portão de esquecimento aberto: isso se aprende no treino da loss de linguagem. A saída de um expert da memória quente não é uma segunda política aprendida. Quem não entra nas vagas dadas por `K` não fica quente.

## Limites

O gradiente da loss de linguagem continua passando pelo Transformer que de fato executou. O que este desenho evita é a dependência recorrente dos estados ocultos: o LSTM não os recebe como entrada. As camadas que rodaram seguem no grafo, e o gradiente as atravessa.

O top-K duro não manda gradiente para o expert que não foi executado. A inicialização quase uniforme, sozinha, não garante exploração. A softmax pode concentrar, e o expert de fora continua sem sinal da loss. Se for preciso explorar, a ordem deste desenho é esta: primeiro ruído ou exploração do tipo epsilon sobre o conjunto carregado; depois Gumbel ou estimador straight-through; REINFORCE por último. REINFORCE fica por último porque a variância do estimador é alta, e a loss de linguagem já supervisiona os experts que rodaram. Nenhum desses três está no esqueleto.

Se a projeção final fosse exatamente zero, o primeiro passo não levaria gradiente até o LSTM: a derivada dos logits em relação ao estado passa por essa matriz. O desvio pequeno existe para a distribuição nascer perto da uniforme e o grafo já alcançar a recorrência.

As cotas de banda e de memória são estimativas de viabilidade. A demanda sustentada tem de caber em `BW`. Um `L` maior dá mais tokens de antecedência para uma transferência que já cabe; não corrige sobrecarga de banda. A conta útil desconta memória reservada, usa o tamanho de expert de cada camada quando eles diferirem, considera a sobreposição entre a cópia e o compute, e lembra que o treino guarda ativações. O script não aplica esses descontos. `M_hot` entra como o orçamento já líquido. `S_e` é o de uma camada, repetido, porque o config não lista tamanho por camada.

## O único passo empírico

Treinar no `L` mínimo da fórmula e comparar a perplexidade com a do OLMoE original, medida uma vez. Uma avaliação lenta serve. Não há varredura de `L`, de `K` nem de tamanho de estado neste desenho. Se `8 * 16` não couber nas cotas, essa conta já é a inviabilidade naquela máquina. Treinar no `L` mínimo não é um jeito de ignorar a cota de banda nem a de memória.

## O que o código faz hoje

`scripts/hw_microbench.py` lê só o `config.json`, por arquivo local ou pela URL do config no Hub. Pesos não entram. Ele calcula `S_e`, mede a banda pinned de host para GPU quando há CUDA e aceita `t_tok` como argumento. Sem `t_tok` e com CUDA, mede um proxy: o tempo das três multiplicações de um expert SwiGLU, um token, nas dimensões do config. Esse proxy não é o tempo de token do OLMoE. Faltam a atenção, os outros experts e o tráfego de ativação. Sem CUDA as duas medições são puladas e o processo termina sem erro. `L` e `K` numéricos só aparecem quando `BW`, `t_tok`, `m` e `M_hot` foram medidos ou passados no argumento. Nada disso é preenchido com um valor de placa inventado.

`src/moe_memory/lstm_ahead_router.py` é o esqueleto: embedding de token, um LSTM, logits por camada para a posição `t+L`, softmax, top-K só como índice do que carregar, peso mole igual à probabilidade do LSTM no expert escolhido, inicialização quase uniforme. O teste usa dimensões minúsculas e não carrega o OLMoE.
