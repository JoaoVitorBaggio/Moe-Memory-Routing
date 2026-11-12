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

O morno é um pool fixo de vagas em RAM pinned. O tamanho padrão sai do orçamento de ~12 GiB. Threads leem o store com `O_DIRECT` quando o sistema aceita, senão `pread`, no horizonte `L2`, e soltam quem o plano tira do conjunto. O quente é um pool fixo na VRAM, `k1` vagas por camada mais a folga. A cópia sobe num stream CUDA separado, com events, no horizonte `L1`, e pode sobrepor o compute. Sem folga a VRAM reusa a vaga de quem saiu: residentes (~3 GB) mais os 384 experts (~3,6 GB) já encostam nos 6 GB da 1060. Se o expert não chegou, o token espera e o tempo entra no stall daquele nível. `--mode on-demand` não adianta a cópia depois do conjunto inicial. O bench (`scripts/bench_offload_decode.py`) grava tokens/s, stall, bytes lidos, picos, misses, `experts_executed` por token, `warm_start_s` e `warm_start_bytes` em `outputs/`. `--dry-run` não abre o checkpoint.

### Conjunto inicial

O morno nascia vazio. Com `b2` de 5 experts novos por token, pôr `k1` experts em cada uma das 48 camadas leva dezenas de tokens: `48 * k1 / b2`. Numa corrida do `bench_offload_decode` na GTX 1060 (32 tokens, `k1=4`, `warm_slots=320`, `b2=5`, `l1=1`, `l2=3`) o lookahead teve stall zero e executou 2372 experts, cerca de 74 por token, de 192 possíveis (`48 * 4`). A camada que ainda não tinha expert quente soma zero no MoE. O on-demand executou outra quantidade, então tokens/s dos dois modos não mede a mesma carga.

Antes do primeiro token decodificado, `k1` experts de cada camada sobem SSD → RAM → VRAM. São todas as camadas. A escolha é o top-`k1` do score cru do LSTM, o mesmo desempate do resto da máscara (maior score, empate no menor índice). Com prompt, o score é o raw do último token do prompt, e o estado do LSTM segue para o decode; o prompt não gasta `b2` e não avança o passo da máscara. Sem prompt, o LSTM pontua o token 0 (`BOS_ID`). Esse token não entra no lote de treino nem na sequência decodificada: só escolhe o conjunto. O carregamento não entra no teto `b2`. O `b2` vale a partir do primeiro token, para crescer o morno até `k2` ou trocar quem tem score pior.

Depois desse carregamento toda camada tem `k1` experts quentes. O passo de decode não executa camada com zero experts. O tempo e os bytes dessa subida ficam em `warm_start_s` e `warm_start_bytes`. Não entram no stall: o token ainda não começou. O JSON do bench traz os dois, mais `experts_executed` e `experts_executed_per_token`.

A máscara de treino usa a mesma função, `initial_set` em `src/moe_memory/tier_mask.py`, chamada por `TwoHorizonLstmRouter.plan` no começo de cada sequência. O passo de treino continua sem mover byte. Inferência sem prompt e o `plan` do treino semeiam o mesmo conjunto e, dali em diante, aplicam o mesmo `b2`.

O bench liga o conjunto inicial no lookahead e no on-demand. Os dois podem executar a mesma quantidade de experts por token. A quantidade não diz qual expert rodou. `--no-warm-start` volta ao morno vazio: o `b2` enche o conjunto durante o decode, e uma camada pode executar zero experts nos primeiros tokens.

O prompt alimenta o roteador. O cache de chave e valor do Qwen3 continua começando no primeiro token decodificado; esta seção não faz o prefill da atenção.

### Instrumentação e comparação pareada

O `elapsed_s` do bench é a parede de `decode()`, do instante anterior ao warm-start até o fim do último token. Entram o warm-start, o enqueue do prefetch e o loop. Ficam de fora a construção do runtime e o `load_resident` (`resident_load_s`). `prefetch_enqueue_s` é null no on-demand; no lookahead é só o tempo de enfileirar, não o fim da cópia. `tokens_per_s` divide por `elapsed_s`, então o warm-start está no denominador. `tokens_per_s_excluding_warm_start` tira só o warm-start; o prefetch e o loop continuam. `stall_warm_s` é a thread de decode parada em `warm.ensure`. Não é o tempo de `read_into` no worker.

`--profile` liga a coleta por token (parede da thread: roteador, enqueue do plano, forward, stalls, atenção e `apply_experts` na CPU) e, na GPU, `cuda.Event` no stream que enfileira o trabalho. A atenção e o expert na GPU não incluem a espera do ensure. A resolução é um `stream.synchronize` por stream depois que `elapsed_s` já foi fechado. Não há `torch.cuda.synchronize` por token nem por camada. Somas do worker de SSD e dos events não são parcelas da parede: as threads e o stream de cópia se sobrepõem ao compute. Com mais de uma thread, a soma dos reads não é tempo de parede. Passou do teto (`--profile-max-tokens`, padrão 512), o decode segue e o relatório marca truncado. Desligado, esse objeto não existe.

O que isso não prova: a causa de uma diferença de tok/s. Um cgroup perto do teto de memória é observação (`memory.current`, se existir), não causa. Números antigos de uma placa não ganham diagnóstico aqui.

O modo simples do bench não chama `torch.manual_seed` antes do LSTM. O `--seed` só sorteia os ids dos tokens, num `Generator` local. Duas corridas contra o store real, no mesmo processo ou em processos diferentes, não têm o mesmo peso. O `--dry-run` é a exceção estreita: o gerador do bundle sintético semeia o RNG global antes de construir o roteador. Bytes de SSD diferentes entre corridas simples não separam agenda de carga.

`--paired` roda on-demand e depois lookahead, em sequência, no mesmo processo, com o mesmo `state_dict`, os mesmos tokens, os mesmos tetos e o mesmo conjunto inicial. Cada modo usa o runtime de verdade: o prefetch não vira oráculo e `b2`, `k1` e `k2` não afrouxam. O fingerprint cobre tetos, pesos, prefixo do store, tokens, conjunto inicial e as chamadas que o forward fez (camada, expert, peso mole, 8 bytes do buffer de host). Não entra modo, byte de SSD nem tempo. Se a execução divergir, o JSON guarda o ponto e o processo sai com erro. Traço truncado não certifica. O prefixo de host não é hash da cópia na GPU, e o hash do store é o tamanho mais no máximo 4 KiB lidos com `read` limitado, não os ~55 GiB. Byte depois desse prefixo não entra.

O bench continua sem prompt. O token 0 só pontua o conjunto inicial. O KV começa no primeiro token decodificado.

No PC, uma corrida por vez, `MemoryMax=7G` e `MemorySwapMax=0`. O pareado não reescreve o JSON de um modo só; o `--out` é outro arquivo. Repetir é outro `--out`, depois que o processo anterior saiu.

Para score finito, a máscara escolhe o melhor de fora e o pior residente por redução, em todas as camadas de uma vez, com o mesmo desempate de índice. O conjunto, a ordem, o desempate, a prioridade da vaga vazia e o `b2` global são os da regra anterior. NaN e `±inf` continuam no argsort estável, porque a margem `inf - inf` não tem ordem. `scripts/bench_tier_plan.py` compara as duas implementações nesta CPU; esse número não é tok/s e não é a 1060.

`schedule_enqueue_wall_s` é a parede da thread de decode dentro de `_install`, não a cópia. No lookahead essa função também chama `lookahead_plan` (`plan_from_scores` da janela `L2`) e monta a lista futura; no on-demand esses dois trechos não rodam e o campo fica null, não zero. O perfil ligado parte essa parede em ids, plano da janela, rank, `hot.sync` e `warm.sync`, com a espera do lock e o tempo segurando o lock separados da parede do sync. `thread_cpu` é `time.thread_time` e não se soma à parede. Não há synchronize de CUDA dentro de `_install`. O plano fica no device do LSTM, e o bench deixa o LSTM na CPU. Parede muito maior que a CPU da thread indica que a thread não executava; isso ainda não separa lock de um spin de driver, e não foi medido na 1060. A resolução dos events deduplica pelo handle `cuda_stream`: `current_stream()` devolve um wrapper novo por chamada, e contar `id` do objeto sincronizava uma vez por span.

O pareado no mesmo processo fechava o primeiro runtime sem soltar os blocos do `SlotPool`. O quente e o morno se apontam (o callback de pronto), então o ciclo só caía no coletor. O segundo modo alocava outro pool com o primeiro ainda vivo. `close` agora solta os blocos e tira o callback antes de o lookahead construir o runtime. O traço guarda 8 bytes copiados, não uma vista do bloco. Cada fase imprime uma linha no stderr, com flush, e o JSON repete esses instantâneos. `tracked_pool_bytes` é a soma dos pools ainda não soltos. `ram_peak_bytes` continua `max(ru_maxrss, morno lógico + residente lógico)`: residente na GPU entra nessa soma mesmo fora do RSS. `memory.current` do cgroup inclui cache de página; não é o RSS. `VmPin` é mlock do kernel e pode não listar o pinned do CUDA.

```
python3 scripts/bench_offload_decode.py \
    --store-dir models/qwen3-30b-a3b-store \
    --paired --profile --trace \
    --tokens 128 --seed 0 \
    --k1 4 --k2 6 --b2 5 --l1 1 --l2 3 \
    --warm-slots 320 --hot-slack 0 --warm-threads 2 \
    --vram-budget-gb 5.5 --device cuda \
    --out outputs/paired-128-seed0-a.json
```

A corrida de 09/11/2026 na 1060 usou essa forma, com `MemoryMax=9G` e `MemorySwapMax=0`, e `--out` em `outputs/paired-128t-seed0-pr27-9g-*.json`. A medição seguinte, se você autorizar, repete o pareado de 128 tokens com perfil e compara `schedule_lookahead_plan_wall_s` e `schedule_lookahead_plan_thread_cpu_s` com essa corrida. Não soma GPU, H2D nem SSD na parede, e não trata o microbench da máscara como ganho de tok/s. O controle teve pouca espera de SSD e de hot, então acelerar o plano não promete que o lookahead passe o on-demand.

## Smoke de loss de linguagem

O decode com pool não treina: o logit sai com `detach` e a vaga do expert é reaproveitada. O passo de linguagem mora em `scripts/train_qwen_lm.py`. Ele lê o store sem o pool, aplica LoRA local nas três projeções e faz a entropia cruzada do próximo token. Treinam o LSTM e a LoRA. O store e o residente não entram no otimizador. Não há `peft`, não há tokenizador GPT-2 e não há download de shard.

A GTX 1060 é sm_61 e não executa bf16. O smoke fica na CPU, em float32, com o residente bf16 no lugar. O teto de lançamento do script é 8 GiB, abaixo dos 9663676416 do cgroup, para sobrar folga. A VRAM de 5,5 GiB não entra porque a GPU não recebe o residente. A saída vai para `outputs/qwen-lm-smoke/`, não para os checkpoints antigos do WikiText com GPT-2.

```
python3 scripts/train_qwen_lm.py \
    --store-dir models/qwen3-30b-a3b-store \
    --model-dir models/qwen3-30b-a3b \
    --cache-dir data/cache \
    --steps 2 --seq-len 8 --seed 0 \
    --k1 4 --k2 6 --b2 5 --l1 1 --l2 3 \
    --lora-rank 4 --lora-alpha 8 \
    --lstm-embed 64 --lstm-hidden 64 \
    --lr 1e-4 --device cpu \
    --out outputs/qwen-lm-smoke
```

`--estimate-only` imprime a conta e não aloca o residente. Para `--seq-len 8`, rank 4 e LSTM 64, com o residente de 3082214946 bytes, a conta fecha em cerca de 7,5 GiB. Não é RSS medido. Isso não é corrida longa nem medida de qualidade. A entropia que o JSON guarda é diagnóstico, não termo de loss.

O smoke curto de 10/11 parou no tokenizador, antes da loss. Os 26 especiais do Qwen estão em `added_tokens`, com ids 151643–151668, e não são chaves de `model.vocab`. A auditoria exigia que fossem e recusava o arquivo. O id de cada um continua o do arquivo. O `vocab_size` do config é 151936: as linhas 151669–151935 da embedding ficam sem token, e o script só recusa se algum id não cabe. Isso se confere sem abrir o store:

```
python3 scripts/train_qwen_lm.py --check-tokenizer --model-dir models/qwen3-30b-a3b
```

A igualdade de ids foi medida contra a biblioteca `tokenizers` no `tokenizer.json` da máquina, SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. Codificar e decodificar o mesmo texto não prova esse id. `tokenizers` não entra no comando da 1060; o loader de lá segue no `tiktoken`.

O teto de 7,53 GiB deixava de fora o gradiente da LoRA, a cabeça inteira em float32 e o cache de página do residente. A cabeça agora entra em fatias de 4096 linhas, cada expert em float32 vive um de cada vez, e a leitura pede `DONTNEED` para o cache. `cgroup_bytes` sai de `memory.max`, não de uma constante. `--memory-log` imprime o RSS por fase. A conta nova fica abaixo de 8 GiB; a validação continua com `MemoryMax=9663676416` e `MemorySwapMax=0`.

Na corrida da PR32 os dois passos terminaram e o processo morreu na passagem extra de validação e no `checkpoint.pt` de 2,6 GB. O checkpoint agora é um arquivo por tensor, com o Adam incluído, gravado depois de `zero_grad` e `malloc_trim`. O Adam usa `foreach=False`. A loss de validação só roda com `--valid-forward`, também depois de soltar o passo. `DONTNEED` continua sendo conselho: a página residual medida foi 1,0–1,6 GiB e entra na conta do cgroup.

`--time-log` parte o tempo do passo sem mudar a loss. A leitura do expert aparece três vezes no código: o forward, a recomputação do checkpoint e o backward. O JSON também simula um LRU (`--expert-cache-sim`) e não guarda esses experts. `--expert-cache N` liga o LRU de verdade: guarda o bf16 do store, antes do float32, e solta os blocos antes do checkpoint. O padrão é 0. O backward relê a camada na hora, então um LRU de 64 vagas segura a camada corrente; a ordem entre camadas continua a do simulador.

`--expert-prefetch-layers D` (padrão 0) lê os experts numa thread enquanto a CPU computa a camada. No forward a fila é a das camadas `l+1..l+D`; no backward, que recomputa de trás para a frente, é `l-1..l-D`. O `pread` solta o GIL. A thread não chama o torch e não mexe no ioprio. O buffer tem teto (`--expert-prefetch-bytes`, ou `D * k2` blocos bf16) e entra na estimativa junto com o LRU. Com o cache de 64 o pico medido encostou em 9 GiB; a combinação que a conta ainda coloca sob 9 GiB é cache 32 mais uma camada de prefetch. Os dois são soltos antes do checkpoint. No Fedora o pytest usa `--basetemp` num diretório de disco, porque `/tmp` é tmpfs.

`--device-compute cuda` calcula uma camada por vez na GPU. O padrão continua a CPU. O Adam, o gradiente da LoRA e o LSTM ficam na RAM. A cópia usa pinned e um stream separado; o pinned entra na conta da RAM do host. A 1060 não executa bf16, então o compute nela é fp32. Ampere ou mais novo pode usar bf16. O orçamento `--vram-budget-gb` (padrão 5) recusa o passo se a estimativa não cabe. Não sobe o modelo inteiro. `--vram-expert-cache-gb` guarda na VRAM os experts já copiados, para o recompute e o backward não repetirem o H2D. A entropia cruzada da cabeça sobe em fatias para a GPU quando o compute é cuda. O `--time-log` separa a espera da volta para a CPU (`d2h_wait_s`), o agrupamento dos experts (`moe_group_s`) e a subida da ativação (`h2d_activation_s`).

`--profile-step N` liga o `torch.profiler` só nesse passo, com atividade de CPU, formas gravadas e sem `profile_memory`. `--profile-stack` e `--profile-trace` ficam desligados: o trace chrome pode ficar grande. O buffer esperado, sem stack e sem trace, é de dezenas a poucas centenas de MB em cima do RSS já estável. Não foi medido na 1060.

O gradiente da LoRA não passa mais por `SelectBackward` da fatia. Cada expert soma em `.grad[camada, expert]`. O zero do tensor inteiro, de até 201 MB, deixa de nascer a cada expert.
