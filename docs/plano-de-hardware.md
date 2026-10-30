# Plano de hardware e emulação

A máquina de referência do grupo ancora a *história* dos três níveis. O experimento mínimo **emula** a hierarquia; não depende de martelar o NVMe a cada passo de treino.

Não há medição neste repositório. Valores de ms em `c(ℓ)`/`w` continuam para `configs/` (razões tipo aluguel bastam no texto).

## Máquina de referência

| Peça | Valor |
| --- | --- |
| GPU | GTX 1060, 6 GB (Pascal); **VRAM útil ≈ 5,5 GB** (reserva ~0,5 GB driver/display) |
| RAM | 16 GB físicos; **orçamento de trabalho ≈ 10 GB** para o experimento |
| Disco frio | SSD NVMe; **reserva 1 GB** para pool frio (2 GB se houver checkpoints) |
| Software | PyTorch, fp32 (sem depender de `torch.compile`) |

Faixa de modelo: ~10–50 M parâmetros. Em fp32 o modelo + Adam + ativações fica na ordem de ~2 GB — **cabe inteiro** nos 5,5 GB. Por isso o orçamento `B` é um **teto artificial** de residência de experts (**referência: 25% de `E`**, com **`E = 32`** ⇒ **8** quentes). Sem `B` apertado, a hierarquia não é exercitada. Preferir expert maior a modelo maior se faltar capacidade com `k=2`.

## Emulação (caminho principal)

O emulador mantém o estado `ℓ(e)` de cada expert, aplica a política de promoção/rebaixamento sob o teto `B`, e acumula:

- custo/proxy alinhado à tabela `c(ℓ)`, `w(a,b)`;
- bytes e contagem de trocas (métrica “de verdade” do experimento mínimo).

**Granularidade:** por **batch** (meio-termo entre fidelidade por token e custo por camada). O roteador **recebe `ℓ(e)`** como entrada.

Simulação de latência por amostra dos `c(ℓ)` serve para iterar a loss. Latência de *parede* **entra no mínimo** e mede-se **sempre na 1060** (treino pode ser Colab/outra GPU). Para a parede refletir a hierarquia, experts fora de `B` ficam **fisicamente** em CPU pinned (morno) ou mmap/NVMe (frio), não só “etiquetados” na VRAM. Emulador cobre bytes/trocas; parede é coluna adicional.

## Tabela de custo (experimento mínimo)

Uma tabela, perfil **consumer ~6 GB / 5,5 GB úteis**, com razões na ordem de aluguel de memória rápida vs lenta:

| Nível | Papel | Ordem de custo de acesso / troca | Capacidade de referência |
| --- | --- | --- | --- |
| VRAM (quente) | residência cara no orçamento `B` | acesso barato | `B = 25%` dos experts |
| RAM (morno) | meio | médio | dentro dos ~10 GB de trabalho |
| SSD (frio) | capacidade | acesso/troca caros | pool ~1 GB (2 GB c/ checkpoints) |

Valores numéricos de `c(ℓ)` e `w(a,b)`: constantes no `configs/`. Partida: `c` = (HBM=1, DDR=10, NVMe=20 ou 30); `w(a,b)` = |c(b)−c(a)| e **amarrado a `c`**. Calibrar na 1060 (NVMe 100 rejeitado como agressivo). HBM ≫ DDR ≫ NVMe em espírito; não exige preço AWS Spot no mínimo.

Os pesos são **hiperparâmetros**: mudar a tabela ou `B` deve adaptar o roteador (benefício do desenho).

## Apêndice: realismo de mercado (extensão)

Fora da entrega mínima, 2–3 vetores de custo no *mesmo* emulador:

1. consumer 6–12 GB;
2. mid cloud;
3. high-end.

Não viram N backends de GPU. Calibrar em aluguel real fica como extensão opcional.

## O que a GPU real ainda faz

Treino e forward dos tensores quentes na GPU disponível (1060, Colab, etc.). O emulador decide *qual expert conta como quente* sob `B`. Bytes/trocas vêm do emulador; **latência de parede do relatório só na 1060**, com níveis morno/frio materializados fora da VRAM.
