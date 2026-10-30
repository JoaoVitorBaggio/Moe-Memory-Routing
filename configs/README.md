# configs/

Hiperparâmetros versionados do experimento mínimo.

| Arquivo | Papel |
| --- | --- |
| [`default.yaml`](default.yaml) | Partida fechada na call 30/10/2026 (e blueprint) |

## Como ler

- `c.*` — custo de **acesso** por nível. `w(a,b)` **não** aparece no YAML: é `|c(b)−c(a)|` e muda com `c`.
- `loss.beta` — fixo pequeno (ordem Switch); valor `0.01` é partida, não busca.
- `loss.lambda_grid` — treinos separados; **não** é Optuna/HPO no mínimo.
- `nvme: 20` — escolhido entre 20 e 30; recalibrar na 1060 se a dose–resposta sair flat/invertida.
- Granularidade do emulador e `router_sees_level` estão aqui para o código não divergir do doc.

Fórmulas e contexto: [docs/funcao-de-custo.md](../docs/funcao-de-custo.md).
