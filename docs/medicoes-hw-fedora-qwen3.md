# Medições de hardware no Fedora — Qwen3-30B-A3B (07/11/2026)

Medições de 07/11/2026, no Fedora, com um script local que **não está no Git**. `scripts/hw_microbench.py` não produziu esta tabela. Os números são estimativas de viabilidade do desenho em [metodo-router-lstm-adiantado.md](metodo-router-lstm-adiantado.md) (seção «Extensão para SSD: roteamento em dois níveis»). Não há treino, perplexidade nem resultado de geração.

O plano de hardware do experimento mínimo continua com outro orçamento: VRAM útil cerca de 5,5 GB e cerca de 10 GB de RAM de trabalho, sob teto artificial. A conta abaixo é a da máquina física neste dia, para um checkpoint que não cabe.

## Máquina

| Peça | Valor neste dia |
| --- | --- |
| GPU | GTX 1060, 6 GB |
| RAM | 16 GB; cerca de 3 GB com o sistema |
| Cache morno | cerca de 12–13 GB |
| SSD | NE-256, firmware SN6329, NVMe PCIe 3.0 x4 |
| Sistema | Fedora |

RAM e VRAM somam 22 GB. O checkpoint em bf16 fica em torno de 61 GB. O modelo inteiro é maior do que as duas juntas.

## Checkpoint e tamanho do expert

O alvo é o Qwen3-30B-A3B, licença Apache 2.0. São 48 camadas, 128 experts por camada, 8 executados por token em cada camada, `hidden_size` 2048 e `moe_intermediate_size` 768.

Cada expert é um SwiGLU de três projeções. A conta é a mesma forma do OLMoE, com o intermediário deste checkpoint:

`S_e = 3 * 2048 * 768 * 2 = 9437184` bytes, isto é 9 MiB, cerca de 9,4 MB.

`48 * 128 = 6144` experts. Os corpos somam `6144 * 9437184 = 57982058496` bytes = 54 GiB. O checkpoint de ~61 GB inclui o resto da rede. Esta nota não decompõe esses bytes.

O token executa `48 * 8 = 384` experts. Os corpos desses 384 são `384 * 9437184 = 3623878656` bytes, cerca de 3,6 GB. É conta de tamanho, não uma residência medida na VRAM.

Cerca de 12–13 GB cabem em cache morno. Se essa faixa for só corpos de expert, a capacidade fica na ordem de 1.400 dos 6.144, ~22% (`1.400 / 6.144`). É teto de capacidade, não um cache observado em execução.

## Bandas e tempos

Leitura do SSD com `O_DIRECT`, banda sustentada, cerca de 0,71–0,80 GB/s. Em blocos do tamanho de um expert o pico chegou a cerca de 1,2 GB/s. O orçamento usa a banda sustentada.

A compressão btrfs `zstd:1` foi medida à parte. Sem compressão a leitura ficou em 0,75 GB/s; com `zstd:1`, em 0,80 GB/s. A compressão fica na mesma faixa e não explica a banda.

Cópia host→GPU a partir de memória pinned: 12,5 GB/s, 0,75 ms por expert. Os 0,75 ms fecham com `S_e / 12,5 GB/s`.

Trazer um expert do SSD levou cerca de 10–13 ms. A divisão de `S_e` pela banda sustentada (0,71–0,80 GB/s) dá cerca de 12–13 ms, junto dessa medição.

`t_layer` cerca de 1,4 ms. O `t_tok` estimado empilha as 48 camadas: `48 * 1,4 ms = 67,2 ms`, registrado como ~68 ms. Batch 1, código ingênuo, sem otimização. Esse ~68 ms é limite superior.

| Medida | Valor |
| --- | --- |
| Leitura `O_DIRECT` sustentada | ~0,71–0,80 GB/s |
| Pico em bloco de expert | ~1,2 GB/s |
| btrfs `zstd:1` | 0,75 GB/s sem; 0,80 GB/s com |
| H2D pinned | 12,5 GB/s; 0,75 ms/expert |
| Expert a partir do SSD | ~10–13 ms |
| `t_layer` | ~1,4 ms |
| `t_tok` estimado | ~68 ms (limite superior) |

## Orçamento por token

Com o expert do SSD em ~10–13 ms e `t_tok` ~68 ms, o passo admite cerca de 5–6 experts novos vindos do disco. A divisão `68/13 ≈ 5` e `68/10 ≈ 7`; o orçamento adotado fica no lado lento da faixa. No piso, `5/384 ≈ 1,3%` dos 384 experts que o token executa. O denominador é esse conjunto executado, não os 6.144 armazenados.

Da RAM para a VRAM, `68 ms / 0,75 ms ≈ 90` experts novos por token.

Os 384 corpos de um token, a 0,75 ms cada, levariam cerca de 288 ms, acima dos ~68 ms. A VRAM reaproveita quem já está quente: a cada token a banda pinned cobre cerca de 90 corpos novos, não os 384. O conjunto morno de ~1.400 experts, a 10–13 ms, levaria cerca de 14–18 s para ser refeito do zero. Por isso ele muda aos poucos, no teto de 5–6 experts novos por passo. Em `L₂` de cerca de 3 tokens essa banda deixa passar cerca de 15–18 substituições.

## Ressalvas

Estas linhas servem para julgar se o desenho cabe na máquina. Não são resultado de treino nem de inferência otimizada.

O `t_tok` de um passo otimizado tende a ser menor que ~68 ms. O número de experts novos por token cai junto, e o orçamento aperta. Os ~68 ms são o tempo do código ingênuo.

O pico de ~1,2 GB/s não entra no orçamento. A conta usa 0,71–0,80 GB/s.

## Nota operacional

O download de cerca de 61 GB travou o PC por pressão de I/O. Em download grande, limitar a velocidade.
