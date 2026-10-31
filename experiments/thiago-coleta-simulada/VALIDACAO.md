# Validação de implementação — 31/10/2026

- Suíte completa: `python -m pytest` — **365 testes passaram**. Um aviso de
  depreciação já existente em Starlette/AnyIO.
- `python -m compileall -q app tests scripts experiments/harth experiments/simulated_collection`
  e `git diff --check` concluídos sem erros.
- Os seis Composes normais e os seis Composes de smoke passaram em
  `docker compose -f <compose.json> config --quiet`.
- As agendas normais dos seis casos foram validadas contra os datasets reais,
  incluindo checagem de estoques disjuntos e dos arquivos HARTH preparados.
- Seis diagnósticos com HTTP local real concluíram cinco rounds em seis
  processos cada: MNIST, CIFAR-10 e HARTH, nos dois modos. Foram conferidos
  os identificadores liberados, a expiração, as contagens ativas, as métricas,
  os checkpoints e as matrizes de confusão. Não houve erros de vizinhança.

Os estoques completos por cliente foram:

| Dataset | Clientes 0–5 |
|---|---|
| MNIST | 10000, 10000, 10000, 10000, 10000, 10000 |
| CIFAR-10 | 8334, 8334, 8333, 8333, 8333, 8333 |
| HARTH (janelas) | 3669, 4281, 3844, 3157, 2756, 4254 |

## Integração Docker concluída

Os seis smokes **em contêineres passaram**, usando Docker Engine 29.9.2 e as
imagens próprias baseadas em Python 3.11. O comando executado foi:

```powershell
.\.venv\Scripts\python.exe scripts/run_collection_experiments.py smoke --all --run-id smoke-docker-20261031-01
```

As três imagens foram construídas com download/preparação dos datasets no
build. Todos os estoques e agendas foram validados antes do primeiro treino.
Cada um dos seis experimentos concluiu cinco rounds nos seis clientes,
totalizando 180 publicações, sem erros de vizinhança. Foram verificados:

- Hash do plano publicado contra a validação prévia e os registros de auditoria.
- Ausência de repetição da coleta e de compartilhamento de exemplos entre clientes.
- Reconstrução dos conjuntos ativos por liberações/expirações e correspondência
  com as contagens, distribuições por classe e pesos usados na agregação.
- Métricas finitas, pesos normalizados, checkpoints e matrizes de confusão finais.

Os contêineres e as redes de teste foram removidos pelo executor ao concluir.
Não houve alteração de código necessária para executar os smokes em Docker.
Esses smokes verificam funcionamento; não constituem coleta científica.

As evidências estão no diretório ignorado `results/thiago-coleta-simulada/`:

- `smoke-docker-20261031-01.log`: build, validações e execução dos seis casos.
- `smoke-docker-20261031-01-summary.json`: conclusão, IDs das imagens e verificações.
- `<experimento>/smoke-docker-20261031-01/`: logs, planos, auditorias, métricas e gráficos.
- Validações locais `*-real-preflight/` e `local-diagnostics/summary.json`.

Datasets, resultados e checkpoints não foram adicionados ao Git. Use outro
`run-id` para repetir a execução, pois diretórios existentes não são sobrescritos.
