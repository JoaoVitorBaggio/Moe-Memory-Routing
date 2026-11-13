# Roteamento consciente de memória

Trabalho de **Aprendizado Profundo (AGL11157)**, UFRGS, 2026/2. O projeto começou na **fase 1** com um MoE pequeno (`E=32`, `k=2`): hierarquia **emulada** VRAM/RAM/SSD, roteador treinado com custo de memória (`λ`), baselines top-k, **LRU** e denso — números em [`results/`](results/README.md). Na **fase 2**, o alvo passou a ser o **Qwen3-30B-A3B** real: ~6.144 experts repackados num store de ~57 GB no SSD, pesos residentes ~3,1 GB, offload **SSD→RAM→VRAM** numa GTX 1060 6 GB, e um **roteador LSTM** que lê só tokens e antecipa quais experts carregar (em vez do gate original, que só decide na hora). Detalhe completo em [**docs/relatorio.md**](docs/relatorio.md).

## Resultados em destaque

| O quê | Número | Leitura |
| --- | --- | --- |
| Fase 1 — latência com `λ=1` vs `λ=0` (test, 1×`B`, semente 0) | p50 **45,20** ms vs **82,43** ms | Penalidade de memória no treino reduz parede e troca; PPL de test **330,18** vs **351,88** ([holdout 2026-11-02](results/holdout_wikitext_2026-11-02.md)) |
| Fase 1 — PPL quase plana em 0,5×–2×`B` | Δ ≤ ~0,37 em 3 sementes | Dose–resposta não mostrou ganho forte de qualidade ao relaxar o teto ([dose 2026-11-03](results/dose_response_wikitext_2026-11-03.md)) |
| Fase 2 — treino viável na 1060 | **~3,3 s/passo**, VRAM **~4,6 GiB**; gate Qwen top-4 **~320 s/passo** | LSTM antecipa carga; gate original força **~3.300** leituras de SSD/passo |
| Fase 2 — qualidade após 1000 passos | test ctx32 PPL **586,73**; ctx128 **518,39** | Muito acima do esperado para LM; referência Qwen/densos **não medida** nas mesmas janelas |

## Como rodar

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest                    # testes da fase 1

# Fase 1 (smoke / overnight na GTX 1060)
python3 experiments/smoke_train.py
python3 experiments/run_lambda_grid.py --device cuda

# Fase 2 (store Qwen repackado; ver docs/metodo-router-lstm-adiantado.md)
python3 scripts/repack_qwen3_experts.py   # uma vez, gera o store no SSD
python3 scripts/train_qwen_lm.py \
    --store-dir models/qwen3-30b-a3b-store \
    --model-dir models/qwen3-30b-a3b \
    --device cpu --device-compute cuda \
    --vram-budget-gb 5 --vram-expert-cache-gb 3.0 \
    --no-checkpoint
```

Medição longa, avaliação de PPL e bench de offload: ver [docs/relatorio.md](docs/relatorio.md) e [docs/metodo-router-lstm-adiantado.md](docs/metodo-router-lstm-adiantado.md).

## Pastas

| Pasta | Conteúdo |
| --- | --- |
| [`src/`](src/) | MoE da fase 1: modelo, roteador, emulador de níveis, loss com `λ` |
| [`src/moe_memory/offload/`](src/moe_memory/offload/) | Fase 2: store de experts, pools RAM/VRAM, runtime de decode e treino LoRA+LSTM |
| [`scripts/`](scripts/) | Repack Qwen, treino `train_qwen_lm.py`, bench offload, avaliação PPL |
| [`experiments/`](experiments/) | Corridas WikiText-2 da fase 1 (grade `λ`, hold-out, dose, denso) |
| [`results/`](results/) | Resumos versionados da fase 1 (PPL, parede, proxy×emulador) |
| [`docs/`](docs/) | Protocolo, método LSTM, medições de hardware, **relatório narrativo** |
| [`configs/`](configs/) | Orçamento, custos e hiperparâmetros da fase 1 (`default.yaml`) |
| [`tests/`](tests/) | Testes unitários e smoke |

## Integrantes

- João Vítor Baggio
- Eric D Arriaga
- Taylor Souza Frutuoso da Costa

## Documentação (histórico — fase 1)

- [Problema e escopo](docs/definicao.md)
- [Função de custo](docs/funcao-de-custo.md)
- [Protocolo experimental](docs/protocolo-experimental.md)
- [Plano de hardware e emulação](docs/plano-de-hardware.md)
- [Referências](docs/referencias.md)

Fase 2: [Método LSTM adiantado](docs/metodo-router-lstm-adiantado.md), [Medições Qwen3 no Fedora](docs/medicoes-hw-fedora-qwen3.md), [**Relatório completo**](docs/relatorio.md).
