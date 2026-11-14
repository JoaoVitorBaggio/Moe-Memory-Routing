# Relatório — roteamento consciente de memória

Documento de portfólio para o trabalho **Aprendizado Profundo (AGL11157)**, UFRGS, 2026/2. Integrantes: João Vítor Baggio, Eric D Arriaga, Taylor Souza Frutuoso da Costa.

O repositório guarda duas linhas de trabalho: **fase 1** (MoE pequeno com emulação de níveis) e **fase 2** (Qwen3-30B-A3B com offload real e roteador LSTM). Números da fase 1 vêm dos arquivos em [`results/`](../results/README.md). Na fase 2, [`metodo-router-lstm-adiantado.md`](metodo-router-lstm-adiantado.md) e [`medicoes-hw-fedora-qwen3.md`](medicoes-hw-fedora-qwen3.md) cobrem **arquitetura** e **bandas de hardware**; a **perplexidade do checkpoint L1**, a **curva de eval do treino de 1.000 passos** e os **tempos por passo** vêm de **logs em `outputs/` no Fedora** (não versionados), salvo quando indicado abaixo.

---

## 1. Fase 1 — MoE emulado (`E=32`)

### Objetivo

Treinar um roteador que enxerga **custo de memória e troca** (`λ` na loss), com experts distribuídos em três níveis **emulados** (quente/morno/frio ≈ VRAM/RAM/SSD), orçamento de referência **`B = 25%`** de `E` (**8/32** vagas quentes), `k=2`, arquitetura ~48 M parâmetros, corpus **WikiText-2**, hardware de referência **GTX 1060 6 GB**. Pergunta principal: `λ > 0` melhora a fronteira **qualidade × latência/I/O** frente a top-k sem custo (`λ=0`), **LRU** (mesmo checkpoint que `λ=0` no código) e LM **denso** de ativos comparáveis. Três sementes; detalhe em [`protocolo-experimental.md`](protocolo-experimental.md).

### O que os resultados mostraram

**Perplexidade de hold-out (1×`B`, WikiText-2).** Na semente 0 ([`holdout_wikitext_2026-11-02.md`](../results/holdout_wikitext_2026-11-02.md)):

| braço | split `test` PPL | p50 ms (`test`) |
| --- | ---: | ---: |
| denso | 339,05 | 12,16 |
| MoE `λ=0` | 351,88 | 82,43 |
| MoE `λ=0,5` | 337,37 | 51,26 |
| MoE `λ=1` | **330,18** | **45,20** |

Com `λ=1`, o MoE fica **abaixo** do denso em PPL nesta semente, com parede ainda **~3,7×** mais lenta que o denso. Em três sementes, média de PPL em `test` ([`holdout_wikitext_2026-11-03.md`](../results/holdout_wikitext_2026-11-03.md)): `λ=0` **351,06 ± 0,81**; `λ=0,5` **333,42 ± 4,40**; `λ=1` **333,78 ± 3,15**. O denso nas três sementes: `test` **339,05 / 318,16 / 327,09** (sem média — regimes de batch diferentes na semente 1).

**Dose–resposta de `B` (0,5× / 1× / 2×, sem retreino).** A PPL fica **quase plana** no eixo; na semente 1 a maior folga em 2 casas é **0,37** (`λ=0,5`, `valid`) ([`dose_response_wikitext_2026-11-03.md`](../results/dose_response_wikitext_2026-11-03.md)). Em `λ=0`, **`wall_p50_ms` cai** quando `B` sobe (mais experts quentes, menos pressão de troca); em `λ=0` na semente 0, `test`: **88,00** ms (0,5×) → **82,69** (1×) → **71,80** (2×) ([`dose_response_wikitext_2026-11-02.md`](../results/dose_response_wikitext_2026-11-02.md)).

**Proxy × emulador.** No conjunto agregado (7008 passos), Pearson entre `c_mem` e `bytes_moved` **0,970**; entre `c_mem` e `swaps` **0,934** ([`proxy_emulator_wikitext_2026-11-02.md`](../results/proxy_emulator_wikitext_2026-11-02.md)). A entropia do roteador cai do primeiro ao último passo conforme `λ` sobe (último passo: **3,029 / 2,362 / 2,244** para `λ=0 / 0,5 / 1`).

**Limite conceitual da fase 1.** O teto `B` é **artificial**: o modelo inteiro caberia na 1060; a emulação mede comportamento do roteador sob restrição, não offload de um checkpoint maior que RAM+VRAM. Isso motivou a fase 2.

---

## 2. Mudança para o modelo real (Qwen3-30B-A3B)

### Por quê

O MoE pequeno não stressa **I/O nem capacidade** reais. O alvo passou ao **Qwen3-30B-A3B** (Apache 2.0): **48** camadas, **128** experts por camada, **8** executados por token por camada (**6.144** experts no total). Os corpos somam **~54 GiB**; o checkpoint completo fica em torno de **~61 GB** ([`medicoes-hw-fedora-qwen3.md`](medicoes-hw-fedora-qwen3.md)). Numa máquina com **GTX 1060 6 GB**, **~14 GB** de RAM útil de trabalho (ordem de grandeza; a nota de 07/11 registra 16 GB físicos com ~3 GB de sistema) e SSD NVMe, **RAM+VRAM não comportam o modelo**.

### Arquitetura de offload

1. **Repack (`scripts/repack_qwen3_experts.py`).** Experts em blocos contíguos no SSD (~**57 GB** de store de corpos; ordem de grandeza alinhada aos **54 GiB** da conta acima). Atenção, normas, embeddings, `lm_head` e **gate original** ficam num **arquivo residente** (~**3,1 GB** em bf16 no orçamento medido).
2. **Três níveis reais.** **SSD** → pool **morno** em RAM pinned (ordem de **~12–13 GB**, ~**1.400** experts, ~**22%** dos 6.144) → pool **quente** na VRAM (`k₁` vagas por camada + folga). Cópias assíncronas; se o expert não chegou a tempo, o token **espera** (stall).
3. **Roteador LSTM (`src/moe_memory/lstm_ahead_router.py`).** Um LSTM lê **só a sequência de tokens** (sem estado oculto do Transformer). Dois horizontes: **`L₂` ~ 3 tokens** (conjunto grande SSD→RAM) e **`L₁` ~ 1–2 tokens** (top por camada RAM→VRAM). O **gate Qwen original não roda** no caminho de treino com LSTM; treino com **LoRA** nos experts + LSTM.
4. **Treino.** `scripts/train_qwen_lm.py` — leitura direta do store, máscara determinística de dois níveis, Adam esparso opcional, opção de compute na GPU com residente e cache de experts na VRAM.

Desenho detalhado: [`metodo-router-lstm-adiantado.md`](metodo-router-lstm-adiantado.md). Código de movimentação: [`src/moe_memory/offload/`](../src/moe_memory/offload/).

---

## 3. O que funcionou

**MoE de ~30B treinando e avaliando numa GPU de 6 GB**, com a maior parte dos pesos no SSD e tráfego controlado por antecipação.

**Velocidade de treino (LoRA + LSTM vs gate original sob o mesmo offload).**

| Configuração | Tempo por passo | Leituras SSD/passo (ordem) |
| --- | ---: | ---: |
| LoRA + LSTM (antecipação) | **~3,3 s** | não medido |
| Gate Qwen original, top-4 | **~320 s** | **~3.300** |

Razão direta: o **gate original só conhece os experts no instante** em que a camada executa; cada passo puxa do SSD sob demanda. O **LSTM decide com antecedência** a partir dos tokens, alinhado aos horizontes `L₁`/`L₂` e ao orçamento **`b₂` ~ 5–6** experts novos do SSD por token ([`medicoes-hw-fedora-qwen3.md`](medicoes-hw-fedora-qwen3.md)).

**Escala do treino longo.** **1.000 passos** ≈ **1 h** ao **~3,3 s/passo**; uso de VRAM no treino medido **~4,6 GiB** (com compute na GPU, cache bf16 e flags documentadas no README/método).

**Decode com lookahead.** Em bench na 1060 (32 tokens, `k₁=4`, `warm_slots=320`, `b₂=5`), modo lookahead com **stall zero** e **~74 experts executados por token** de 192 possíveis (`48×4`) — dado em [`metodo-router-lstm-adiantado.md`](metodo-router-lstm-adiantado.md); não confundir com qualidade de LM.

---

## 4. O que falhou (qualidade)

**Perplexidade WikiText-2 raw**, avaliação com **8.192 tokens**, janelas **determinísticas** (mesmo protocolo `scripts/eval_ppl.py`).

**Passo 0 — LSTM no lugar do gate, sem treino:** PPL **28.014** (ordem de magnitude inicial; não é baseline publicado do Qwen).

**Durante o treino (eval interna, ~992 tokens, ruidosa):** a partir do passo **~800**, valores **457 / 456 / 551** nos passos **800 / 900 / 1000** — curva instável.

**Checkpoint após 1.000 passos (LoRA nível L1, eval final):**

| split | ctx32 PPL | ctx128 PPL |
| --- | ---: | ---: |
| `validation` | **512,68** | **412,23** |
| `test` | **586,73** | **518,39** |

**Throughput na mesma avaliação:** ~**34 tok/s** (ctx32), ~**74 tok/s** (ctx128).

**Contexto.** O treino e a avaliação usaram **`seq_len` 32 e 128**; o modelo suporta contexto até **32K** — não explorado.

**Baselines em falta (mesmas janelas).**

| baseline | status |
| --- | --- |
| Qwen3-30B-A3B, **gate original top-8** | **não medido** — não coube no orçamento de **5 GiB** de VRAM (**5.397.186.048 B** alocados; log do lote de avaliação de **13/11/2026** no Fedora, em `outputs/` não versionado) |
| Qwen3-30B-A3B, gate top-4 com offload | treino inviável (**~320 s/passo**); PPL **não medida** nas janelas acima |
| Qwen3-1.7B / 4B denso | **não medido** |
| llama.cpp quantizado | **não medido** |

Sem essas referências, não dá para afirmar que a PPL alta vem só do roteador ou só do contexto curto — só que o sistema **não recuperou** qualidade de LM útil após 1.000 passos de LoRA+LSTM.

---

## 5. Hipóteses para a falha de qualidade

1. **Trocar o gate destrói a especialização aprendida.** Os experts foram treinados com o roteador linear do Qwen; um LSTM com softmax morna e top-K de carregamento **não reproduz** as mesmas escolhas nem a escala do residual.
2. **LoRA pequena não recupera.** Rank e α usados no treino longo podem ser insuficientes para realinhar **6.144** experts ao novo roteador.
3. **Contexto curto (32/128).** O LSTM vê poucos tokens por passo de treino; a perplexidade em ctx128 já melhora (412/518 vs 512/586), mas ainda longe de LM aceitável — sugere dependência de contexto, sem provar causalidade.
4. **LSTM só com tokens.** O gate original usa **estado oculto da camada**; o LSTM não recebe esse sinal e pode ser **cego** a padrões que o Qwen usa para rotear.

Confiança: hipóteses plausíveis, **não testadas isoladamente** neste repositório.

---

## 6. Próximos passos

**Roteador híbrido (prioridade).** **Destilar** ou reutilizar o **gate original** apenas para **antecipar RAM→VRAM**; se o expert certo não estiver na VRAM, **esperar pela RAM** (nunca bloquear no SSD no caminho crítico), preservando as **escolhas** do gate no compute. O LSTM ou um plano de horizonte curto ficariam subordinados à agenda do gate, não substituindo-o na seleção final.

**Infraestrutura e escala.** Repetir o offload em **GPUs com mais VRAM** para medir gate original top-8 e isolar gargalo de I/O vs qualidade. Manter repack e pools; validar `train_qwen_lm.py` em hardware com bf16 nativo.

**Ideia de capacidade.** Aumentar um modelo MoE **duplicando experts já presentes** (cópia de pesos + novos ids), em vez de treinar experts do zero — hipótese de expansão barata; **não implementada**, **não medida**.

**Baselines de avaliação (só inferência, mesmas janelas WikiText-2).**

1. Qwen3-30B-A3B com gate original (top-8 ou top-4) onde couber em VRAM ou com offload on-demand **sem trocar o gate**.
2. Qwen3-1.7B e 4B denso (`scripts/eval_dense.py` ou equivalente).
3. Mesmo prompt/tokens via **llama.cpp** quantizado, para latência e PPL comparáveis em hardware modesto.

**Fase 1.** Agregado proxy×emulador nas **três sementes** e braço “mesmo orçamento residente” denso vs MoE continuam **a definir** ([`results/README.md`](../results/README.md)).

---

## Referências rápidas no repositório

| tema | arquivo |
| --- | --- |
| Índice de números fase 1 | [`results/README.md`](../results/README.md) |
| Método LSTM + offload | [`metodo-router-lstm-adiantado.md`](metodo-router-lstm-adiantado.md) |
| Bandas SSD/RAM/VRAM (07/11/2026) | [`medicoes-hw-fedora-qwen3.md`](medicoes-hw-fedora-qwen3.md) |
| Porta de entrada | [`README.md`](../README.md) |
