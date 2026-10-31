# Problema e escopo

**Disciplina:** Aprendizado Profundo (AGL11157), UFRGS, 2026/2.  
**Estado:** blueprint revisado (out/2026). Nenhum experimento foi executado. Não há resultado numérico.  
**Feedback do professor:** proposta aprovada (100/100); este documento incorpora as correções de direção (emulação, colocação dinâmica, 16–32 experts, entropia, alinhamento proxy↔emulador, dose–resposta, adversário forte).

## Problema

MoEs ativam só parte dos parâmetros por token. Em hardware real, experts em VRAM, RAM e SSD não custam o mesmo. Sistemas como PowerInfer e *LLM in a flash* já fazem offload. O que este grupo ataca é **treinar o roteador com um custo explícito de acesso e troca entre níveis**, em vez de otimizar só a loss de tarefa e deixar a residência a cargo de uma heurística depois.

Como o modelo do protótipo (~10–50 M parâmetros) **cabe** na VRAM de uma GTX 1060, a hierarquia precisa ser **emulada**: um teto artificial de residência `B` e um emulador que conta bytes e trocas. Sem isso, a penalidade não tem o que cobrar.

## Pergunta de pesquisa

Um roteador treinado com penalidade de memória e de troca melhora a fronteira qualidade × latência / I/O em relação a:

1. MoE top-k clássico (`λ = 0`);
2. o mesmo MoE com **LRU de experts** — **baseline-2 / adversário único**;
3. modelo denso com parâmetros ativos comparáveis.

**Dose–resposta:** sob orçamento **`B`** de referência, o comportamento em 0,5×`B` e 2×`B` na *avaliação* é monótono e útil? (`λ` não é o eixo “0,5×” — é grade de treino separada.) Se baratear memória não muda nada, o sinal não está entrando; se encarecer e o modelo não economizar, o proxy está fraco.

## Experimento mínimo vs extensões

**Mínimo (entrega):**

- um emulador VRAM/RAM/SSD;
- uma tabela de custo (perfil consumer ~6 GB; proporções tipo aluguel HBM ≫ DDR ≫ NVMe, sem preços Spot exatos);
- **`E = 32`**, **`k = 2`**; se faltar capacidade, **aumentar o expert**, não encolher o modelo;
- orçamento **`B = 25%`** dos experts (**8/32**); teste em 0,5×`B` e 2×`B`;
- emulador **por batch**; roteador recebe **`ℓ(e)`** (ciente de memória);
- proxy `k=2`: média ponderada no treino, soma na inferência;
- hardware de referência: VRAM útil 5,5 GB; RAM de trabalho 10/16 GB; SSD frio 1 GB (2 GB com checkpoints);
- `λ ∈ {0, 0,5, 1}` com treino **separado por `λ`** (não condicionado); `β` fixo pequeno;
- comparação com top-k (`λ=0`), **LRU só** (adversário único) e denso (tronco + `k`·expert);
- checagem de correlação entre custo na loss e bytes/trocas do emulador;
- monitorar entropia do roteador (colapso por preferir VRAM).

**Medição:** latência de parede na 1060 entra no mínimo (`B` fixo).

**Extensões (apêndice / depois):**

- 2–3 vetores de custo (“consumer 6–12 GB”, “mid cloud”, “high-end”);
- treino com orçamento amostrado por batch/época (**adiado**);
- calibrar `c(ℓ)` em preços de aluguel reais.

Não é objetivo do núcleo emular N placas distintas como backends separados.

## Decisões de desenho (fechadas na discussão)

| Item | Escolha |
| --- | --- |
| Upsizing do modelo | Não: manter faixa pequena e **emular** a hierarquia; capacidade via **expert maior**, não modelo maior |
| `k` / `E` | `k = 2`; **`E = 32`** |
| `B` | 25% dos experts (**8/32**; artificial; tudo cabe na 1060) |
| `λ` / `β` | Grade `{0, 0,5, 1}`, um treino por `λ`; `β` fixo pequeno na ordem do Switch (exato em aberto) |
| Baseline-2 | LRU de experts apenas — o mesmo `ResidencyEmulator` do `λ=0`, não uma segunda política |
| Denso | tronco + `k`·expert |
| Roteador | Linear/MLP + **`ℓ(e)` como entrada** (ciente de memória); não tool-calls |
| Emulador | Granularidade **por batch** |
| Proxy `k=2` | Treino: média ponderada top-2; inferência: soma top-2 |
| Pesos de custo | Partida `c`=(HBM=1, DDR=10, NVMe=20|30), `w=Δc` amarrado a `c`; calibrar na 1060 |
| “Quebrar monopólio” de experts | Fora do caminho crítico do experimento mínimo |

## O que conta como sinal

Não há dataset rotulado externo. Sinal de treino: próximo token + métricas de recurso do emulador (e latência de parede quando o protocolo exigir máquina controlada).

## Corpus

Decisão (out/2026):

- **TinyStories** — só desenvolvimento: iterar, achar bug, validar pipeline e ordem de grandeza de perda/throughput.
- **WikiText-2** — treino e avaliação “de produção” do trabalho (tabelas, dose–resposta, baselines). Corrida longa; plano típico: deixar o PC ligado à noite.
- WikiText-103 — reserva se WikiText-2 for insuficiente; fora do caminho mínimo.

Os dois entram no código via config. Números do relatório vêm do WikiText-2, não do TinyStories.

## Fora de escopo (seminário vs trabalho)

No seminário: MoE **não** é aplicação direta de STE / Gumbel / REINFORCE; esparsidade é por expert/bloco, não por neurônio; esparso ≠ barato; experts não são necessariamente “especialistas” semânticos. Este repositório é o trabalho experimental, não o slide deck.
