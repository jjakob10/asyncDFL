# Validação da coleta e reorganização — 31/10/2026

## Implementação original, publicada no commit fa9a4b5

- Suíte completa: `python -m pytest` — **365 testes passaram**. Um aviso de
  depreciação já existente em Starlette/AnyIO.
- `python -m compileall -q app tests scripts experiments`
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

## Reorganização na branch codex/reorganize-experiments

A implementação original foi publicada em `origin/codex/thiago-experiments`
antes de criar esta branch. A refatoração extraiu as ferramentas compartilhadas
para `scripts.experiment_tools`, incorporou a fonte de coleta nos treinadores
originais de imagem e removeu as famílias aposentadas e seus testes exclusivos.
Os comandos de dados fixos e de coleta continuam separados. O executor de
dados fixos agora se chama `run_experiments` e exige um manifesto explícito.

Verificações da reorganização:

- **335 testes passaram**, mantendo a cobertura genérica de métricas,
  falhas, exportação, coleta e métodos M0–M4 de Johann.
- Compilação de `app`, `experiments`, `scripts` e `tests`, e verificação do diff,
  sem erros. Não restam imports ou comandos para os módulos removidos.
- Comparação antes/depois de 12 experimentos tradicionais, 12 variantes de
  coleta (normal/smoke) e 120 configurações Johann: igualdade, permitindo apenas
  as duas referências de treinadores de coleta deliberadamente migradas.
- Os 36 planos completos de estoque/liberação (seis clientes em seis casos)
  preservaram os identificadores, os metadados e os hashes da versão anterior.
- As 12 exportações tradicionais foram geradas e aceitas pelo Docker Compose.
- **20 smokes Docker passaram**: 12 Thiago tradicionais de um round, seis de
  coleta de cinco rounds e dois Johann M0 de 20 segundos. Os primeiros 18 usam
  seis clientes; os dois Johann usam oito clientes cada.
- Conferência de auditorias, contagens, pesos, métricas finitas, checkpoints,
  matrizes de confusão e resumos. Os 18 smokes por rounds não tiveram erros de
  vizinhança. Os planos de coleta Docker coincidem com os do smoke original.
- As imagens foram reconstruídas com Python 3.11 e sem as três dependências
  diretas retiradas. Os contêineres e as redes de teste foram encerrados.

As evidências desta etapa estão em `results/reorganization/` (ignorado):
`equivalence.json`, `before.json`, `after.json`, `pytest-final.log`,
`docker-summary.json`, `image-checks.json`, `exports/` e `docker/`, além dos
logs `docker-*.log`. O identificador dos smokes foi `reorganization-20261031`.
Datasets, resultados, imagens anteriores e entregáveis de pesquisa foram
preservados. Os testes sintéticos e fixtures exclusivos das famílias removidas
foram retirados do código ativo.
