# Roteador LSTM adiantado

Este desenho substitui o roteador do OLMoE por um LSTM que lê só o fluxo de tokens. Fica ao lado do experimento dos quatro braços: não altera `configs/default.yaml`, não muda a loss com `λ` e não destila o roteador original. Não há número medido nas seções do OLMoE. A extensão no fim deste arquivo é a decisão de 07/11/2026; as bandas medidas estão em [medicoes-hw-fedora-qwen3.md](medicoes-hw-fedora-qwen3.md) e não preenchem `L` nem `K` deste esqueleto. `L`, `K` e o tamanho do estado do LSTM saem da máquina, na hora da corrida, pelo `scripts/hw_microbench.py`. O módulo `src/moe_memory/lstm_ahead_router.py` é um esqueleto. Ele fixa a forma do cálculo, com dimensão minúscula no teste, sem peso de OLMoE e sem laço de treino.

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

## Extensão para SSD: roteamento em dois níveis (decidido em 07/11/2026)

Esta seção registra a decisão de 07/11/2026. Fica ao lado dos quatro braços: não altera `configs/default.yaml`, não muda a loss com `λ` e não mexe no esqueleto do OLMoE nem nos braços já medidos. As bandas e os tempos estão em [medicoes-hw-fedora-qwen3.md](medicoes-hw-fedora-qwen3.md). São estimativas de viabilidade. Não há treino nem perplexidade desta extensão.

No protocolo, **não cabe** é o MoE pequeno sob o teto artificial `B`, porque aquele modelo cabe na 1060. O alvo daqui é outro: o checkpoint inteiro é maior do que a RAM e a VRAM juntas. Não há teto artificial, e esta nota não reclassifica os quatro braços.

### Alvo

O modelo é o Qwen3-30B-A3B, licença Apache 2.0. Em bf16 o checkpoint fica em torno de 61 GB. São 48 camadas e 128 experts por camada, com 8 experts executados por token em cada camada. `hidden_size` é 2048 e `moe_intermediate_size` é 768. Cada expert é um SwiGLU de três projeções, `S_e = 3 * 2048 * 768 * 2 = 9437184` bytes, cerca de 9,4 MB. Há `48 * 128 = 6144` experts. Os corpos somam `6144 * 9437184 = 57982058496` bytes, 54 GiB. A máquina tem 16 GB de RAM e 6 GB de VRAM, 22 GB juntas. Os 61 GB do checkpoint não cabem nas duas.

O treino usa só LoRA nos experts e no roteador. Os pesos base ficam congelados. Ajuste fino completo e treino do zero ficam de fora: o estado do Adam fica na ordem de 2–3× os 61 GB, acima dos cerca de 44 GB livres em disco (a RAM dessa máquina é 16 GB), e gravar esse estado desgasta o SSD.

### Desenho

Um único LSTM, o mesmo que lê só tokens, trabalha com dois horizontes e dois tamanhos de seleção.

O horizonte longo `L₂` é de cerca de 3 tokens. Ele escolhe o conjunto grande que sobe do SSD para a RAM. O conjunto muda devagar: a cada passo há um teto de experts novos, e quem entra ocupa o lugar dos de menor score.

O horizonte curto `L₁` é de cerca de 1–2 tokens. Dentro do conjunto morno, ele escolhe os 8 experts por camada que sobem da RAM para a VRAM. O 8 por camada é a esparsidade do checkpoint. Esta extensão não varre esse número. O nível curto escolhe só dentro do conjunto do nível longo. A troca RAM↔VRAM é a mais rápida, então este horizonte é o menos adiantado.

A alternativa simples inicial renova a VRAM a cada `N₁` passos e a RAM a cada `N₂` passos. Os períodos são independentes: `N₂` não precisa ser múltiplo de `N₁`.

### Princípios

O LSTM não prevê o roteador original e não tem taxa de acerto. A loss de treino continua sendo a de linguagem, e só ela.

O orçamento de experts novos por token é restrição da arquitetura. Ele fica fora da loss. Não há penalidade de troca em nenhum dos dois níveis. No experimento anterior, penalidade de troca levou a colapso ou a troca constante, mesmo com a loss de balanceamento.

Adiantado é decidir cedo, com o passado. A decisão da posição `t+L` usa só os tokens de índice `≤ t`. A mesma regra vale no treino, no prefill e na geração.

A escolha fecha antes da cópia. O assincronismo é só o tempo de trazer o peso. Se a cópia atrasar, o token espera, e a saída é a da escolha já feita.

No treino, a regra dos dois níveis é uma máscara determinística, a mesma da inferência. O passo de treino não move memória para aprender.

O lote paralelo forma a união dos experts de várias sequências, maior do que a de uma sequência só. A regra vale por sequência. O lote fica pequeno, ou o gradiente acumula, para essa união continuar fiel à máscara.

O Pre-gated MoE adianta o gate a partir do estado interno da rede. Cada nível de memória veria um estado diferente e precisaria do próprio mecanismo. O LSTM deste desenho lê só tokens. A mesma recorrência emite os dois horizontes.

### O que a medição de 07/11/2026 fixa

Naquela máquina, o orçamento por token fica em cerca de 5–6 experts novos a partir do SSD (no piso, `5/384 ≈ 1,3%` dos 384 que o token executa: `48 * 8`) e cerca de 90 a partir da RAM. O cache morno, em cerca de 12–13 GB, comporta cerca de 1.400 dos 6.144 experts, ~22%. O `t_tok` de ~68 ms é limite superior de código ingênuo, batch 1, sem otimização. Um passo otimizado tende a ser mais curto, e o orçamento aperta. A conta, as bandas e as ressalvas estão na nota de medição.

### Movimento do byte

Quem executa a cópia é `src/moe_memory/offload/`. O `TierPlan` continua só listando `(camada, expert)`. Isto não altera `configs/default.yaml` nem os quatro braços.

O store guarda cada expert num bloco contíguo alinhado a 4 KiB, `gate|up|down`, com um índice pequeno. Atenção, normas, embeddings, `lm_head` e o gate original ficam num arquivo residente. O repack (`scripts/repack_qwen3_experts.py`) lê os safetensors em streaming, um expert por vez, e limita a taxa de escrita com fsync por pedaço. Não carrega o modelo. Um download de ~61 GB sem teto travou o PC por page cache suja; o padrão é 80 MB/s.

Na máquina de 07/11/2026 o checkpoint tem cerca de 61 GB e sobram cerca de 44 GB. A saída repete os corpos (~54 GiB) e o resto da rede. Sem apagar a fonte, os dois não cabem juntos, e o script sai sem escrever. `--delete-source-after-each-shard` apaga cada shard depois do fsync; o pico extra fica na ordem do maior shard. A flag é destrutiva e o padrão é não usá-la.

O morno é um pool fixo de vagas em RAM pinned. O tamanho padrão sai do orçamento de ~12 GiB. Threads leem o store com `O_DIRECT` quando o sistema aceita, senão `pread`, no horizonte `L2`, e soltam quem o plano tira do conjunto. O quente é um pool fixo na VRAM, `k1` vagas por camada mais a folga. A cópia sobe num stream CUDA separado, com events, no horizonte `L1`, e pode sobrepor o compute. Sem folga a VRAM reusa a vaga de quem saiu: residentes (~3 GB) mais os 384 experts (~3,6 GB) já encostam nos 6 GB da 1060. Se o expert não chegou, o token espera e o tempo entra no stall daquele nível. `--mode on-demand` não adianta a cópia. O bench (`scripts/bench_offload_decode.py`) grava tokens/s, stall, bytes lidos, picos e misses em `outputs/`. `--dry-run` não abre o checkpoint.
