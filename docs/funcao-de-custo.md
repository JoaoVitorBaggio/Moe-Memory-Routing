# Função de custo

A loss de treino do roteador é

```text
L = L_tarefa + λ · C_mem + β · L_balanço
```

`L_tarefa` é a entropia cruzada do próximo token. `C_mem` é o *proxy* diferenciável do custo de memória/troca sob o softmax do roteador. `L_balanço` evita colapso trivial (sempre os experts já “baratos” / na VRAM). `λ` varre a curva com grade **`{0, 0,5, 1}`** e **um treino por valor** (não há roteador condicionado a `λ`). O ponto **`λ=0` é o baseline 1** (top-k + balanço, sem `C_mem`). `β` fica **fixo** numa corrida, pequeno, **na ordem do Switch** (valor exato ainda em aberto; só a ordem de grandeza).

Não há resultado de treino. Os custos `c(VRAM)`, `c(RAM)`, `c(SSD)` e as penalidades de troca `w(a,b)` saem de uma **tabela de hiperparâmetros** com proporções defensáveis (tipo aluguel: HBM ≫ DDR ≫ NVMe), não de preços Spot exatos no experimento mínimo.

## Notação

| Símbolo | Significado |
| --- | --- |
| `T` | Tokens no batch |
| `E` | Experts da camada: **`32`**; inferência com **`k = 2`** |
| `z(t,e)`, `p(t,e)` | Logit e softmax do roteador |
| `ℓ(e)` | Nível atual do expert no emulador / cache |
| `c(ℓ)` | Custo de acessar um expert já no nível `ℓ` |
| `w(a,b)` | Penalidade de trocar `a → b` |
| `B` | Orçamento (teto) de residência no nível quente; referência **25%** de `E` |
| `λ`, `β` | Pesos da loss |

## Proxy diferenciável vs métrica do emulador

Na inferência a escolha é discreta (top-k); na loss o custo entra como valor **esperado** sob `p(t,e)`.

```text
C_acesso ≈ (1/T) Σ_t Σ_e p(t,e) · c(ℓ(e))
```

Mais um termo de troca esperado (ou surrogate alinhado à política de promoção/rebaixamento), com **`w(a,b)` amarrado a `c`**: se a tabela de `c` muda, `w` muda junto (`w = |c(b) − c(a)|`).

Com **`k = 2`**:
- **treino (proxy na loss):** média ponderada dos top-2 — o roteador aprende a *ranquear*;
- **inferência (custo realizado):** **soma** dos top-2 — custo real de acesso.

Troca entre conjuntos: política do emulador promove/rebaixa sob `B`. Granularidade do emulador: **por batch** (meio-termo entre fidelidade por token e custo por camada). O roteador **recebe `ℓ(e)`** como entrada (roteamento ciente de memória).

**Requisito do feedback (ponto 5):** o proxy e o emulador precisam falar a mesma língua. Se a loss acha barato misturar experts “frios” mas o LRU/emulador gera thrashing, o roteador otimiza o proxy errado. No relatório: correlacionar (ou ao menos ordenar) `C_mem` na loss com bytes e trocas contados pelo emulador.

## L_tarefa e L_balanço

`L_tarefa`: CE de next-token, média por token.  
`L_balanço`: forma à la Switch / Fedus (`E · Σ_e f(e) P(e)` ou equivalente). Impede resolver memória colapsando num subconjunto mínimo de experts quentes. Monitorar também a **entropia** do roteador ao longo do treino.

## Pesos de custo como hiperparâmetros

`c(ℓ)` e `w(a,b)` (e o orçamento `B`) são variáveis do experimento:

- mudar a tabela (ex.: VRAM mais “cara”) deve fazer o roteador adaptar;
- isso é o benefício pretendido: menor disponibilidade relativa de um nível ⇒ outra política.

No mínimo: **uma** tabela (consumer). Apêndice: 2–3 tabelas (consumer / mid / high-end) sem virar N backends.

### Valores de partida (configs/)

Proposta inicial para o perfil consumer (unidades relativas, não ms nem dólares Spot):

| Nível | `c(ℓ)` |
| --- | --- |
| HBM / VRAM (quente) | `1` |
| DDR / RAM (morno) | `10` |
| NVMe / SSD (frio) | `20` ou `30` (escolher um e fixar na corrida) |

`w(a,b)` de partida: **diferença de `c` entre os níveis** (`|c(b) − c(a)|`) — a troca só compensa se o ganho de acesso pagar a multa. **`w` fica amarrado a `c`**: muda a tabela de `c`, muda `w`. Proposta `100` para NVMe foi considerada agressiva demais. São **valores de partida** para calibrar com o que a 1060 aguentar (NVMe 20 vs 30 ainda aberto); sem fechar números fixos no sentido de “não recalibrar”. Números vivem em `configs/`; ablação pode trocar a tabela sem mudar a fórmula.

## Relação com dose–resposta

Varre-se `λ` e/ou `B` (0,5×, 1×, 2×). A figura dose–resposta e a tabela contra baselines estão no [protocolo](protocolo-experimental.md).
