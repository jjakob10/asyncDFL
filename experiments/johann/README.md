# Johann M0–M4: clientes offline e reintegração

M0 continua sendo o padrão. `--methods M0 M1 M2 M3 M4` seleciona métodos adicionais:
M1 usa idade, M2 limited merging, M3 substituição e adaptação no retorno, e M4
a mesma janela com destilação. Idade conta **visitas a exemplos**, incluindo
reapresentações; é separada de passos locais e publicações. As referências,
equações, adaptações e limites de novidade estão em
[implementacao_M1_M4_referencias_e_contribuicoes.md](../../outputs/tcc/implementacao_M1_M4_referencias_e_contribuicoes.md).

Os manifests fixam 600 s/20 s no MNIST e 1.800 s/60 s no CIFAR-10.
Os parâmetros dos manifests locais continuam sendo valores de desenvolvimento,
com `protocol_frozen=false`. A seleção calibrada e as imagens foram preservadas
em um pacote externo; `run` exige `--frozen-bundle`. Comandos de desenvolvimento,
executados da raiz:

```powershell
& $py $runner smoke --dataset all --methods M1 M2 M3 M4 --run-id smoke-metodos
& $py $runner calibrate --dataset all --run-id busca-validacao
& $py $runner select-calibration results/johann --run-id busca-validacao
& $py $runner pilot --dataset all --methods M0 M1 M2 M3 M4 --scenario concentrated-r --seeds 8 9 --selected-config results/johann/busca-validacao-selection.json --run-id confirmacao-validacao
```

Defina `$py` e `$runner` conforme a preparação abaixo. `calibrate --render-only`
permite inspecionar toda a grade sem Docker. O seletor exige os onze candidatos
completos por dataset, usa somente validação/semente 7 e não congela o protocolo
automaticamente. As sementes 8/9 confirmam a escolha; 42–46 ficam para coleta.
Overrides: `--merge-limit-examples` ou `--merge-limit-multiplier` (M2),
`--rejoin-batches` (M3/M4), `--distillation-weight` e
`--distillation-temperature` (M4). Cada configuração deve usar um run-id próprio.

Se Docker estiver indisponível, `smoke --backend local` inicia oito processos
Python com HTTP real e encerra apenas os processos criados pelo executor.
Esse backend é exclusivo de smoke/validação, registra `execution_backend=local_smoke`
e não deve ser usado para conclusões científicas ou misturado a execuções Docker.
Por exemplo: `smoke --backend local --dataset all --methods M1 M2 M3 M4
--rejoin-batches 2 --duration 45 --run-id verificacao-local`.

M3/M4 descartam o ciclo privado suspenso e inicializam pelo vizinho de maior
idade, mesmo se inferior à idade preservada. Sem vizinho elegível, aguardam dentro
do orçamento. Copiam após consultas feitas no retorno e publicam uma vez,
depois de K minibatches. O professor de M4 fica congelado e usa somente o treino
local. Sem ausência, ambos percorrem M0. Diagnósticos são avaliados após o orçamento.
Na intervenção, pesos convexos de parâmetros ficam ausentes, pois não representam
o treinamento nem a destilação.

Este experimento investiga se a ausência de clientes que concentram determinadas
classes afeta o reconhecimento dessas classes nos clientes que continuam online.
Há configurações para **MNIST e CIFAR-10**, com oito clientes e treinamento em CPU.
Os tempos/cadências e os parâmetros M1–M4 foram calibrados em validação; o pacote
selecionado não é substituído pelos defaults dos manifests locais.

## M0 e cenários compartilhados

M0 combina o modelo treinado localmente com os snapshots disponíveis dos vizinhos,
ponderando cada um pelo número de exemplos de treinamento do emissor:

`novo_modelo = soma(n_cliente * modelo_cliente) / soma(n_cliente)`.

Não há servidor agregador, rodada global, ponderação por idade, substituição no
retorno, destilação nem treinamento extra. Cada cliente percorre seus próprios ciclos.
As classes acompanhadas pertencem à avaliação e ao desenho dos dados; não alteram
os pesos da agregação. Versão significa publicações locais, não idade de treinamento.

| Elemento | Configuração |
|---|---|
| Grupo R | Clientes 0 e 4 |
| Grupo C | Clientes 2 e 6 |
| Classes acompanhadas | 3 e 8: dígitos no MNIST; `cat` e `ship` no CIFAR-10 |
| Rede | Anel 0–1–2–3–4–5–6–7–0, mais 0–4, 1–5, 2–6 e 3–7 |
| `distributed` | Classes distribuídas aproximadamente por igual |
| `concentrated` | 40% de cada classe acompanhada em cada cliente de R |
| Agenda `r` / `c` / `none` | Ausência de R / ausência de C / todos online |

São seis cenários por dataset: `distributed-r`, `distributed-c`, `distributed-none`,
`concentrated-r`, `concentrated-c` e `concentrated-none`. O executor verifica que a
rede ativa permanece conectada após remover R ou C.

O treino oficial é separado em 90% para treinamento e 10% para validação,
estratificando por classe antes do particionamento. Os mesmos exemplos globais
entram em todos os cenários de uma semente. Cada cliente recebe um total que difere
dos demais em no máximo um exemplo. Na distribuição concentrada, os outros 20%
das classes 3/8 são divididos entre os seis clientes restantes; as demais classes
preenchem as capacidades locais proporcionalmente. Arredondamentos usam maiores
restos com desempate por índice. Configurações inviáveis são rejeitadas.

Os artefatos incluem índices locais, hashes globais de treino/validação, nomes e
contagens das classes e proporções efetivamente obtidas. O limite de amostras do
smoke é global, aplicado de forma estratificada antes da separação de validação.
Partições e inicialização são reprodutíveis por semente; a ordem das trocas
assíncronas depende do escalonamento e não garante trajetórias idênticas de treino.

## Tempo, pausa e retorno

O executor aguarda os oito clientes, agenda um instante UTC comum dez segundos à
frente e confirma os oito agendamentos antes dele. A partir desse instante cada
processo usa seu relógio monotônico. Isso pressupõe os contêineres no mesmo host;
hosts distintos exigiriam sincronização e validação de relógios.

| Período | Comportamento |
|---|---|
| 0–30% | Todos treinam e trocam modelos |
| 30–60% | Grupo selecionado fica offline |
| 60–100% | Retorno e observação da recuperação |

Durante a pausa, o treinamento para no próximo limite de minibatch e preserva o
modelo privado, o otimizador e a posição do treino em memória. Um minibatch ou uma
requisição já iniciados podem terminar; os eventos registram os atrasos observados.
Não há reinício de processo. A API `/model` retorna HTTP 503 e nenhuma nova consulta
de modelos é iniciada enquanto o cliente está offline. `/health` e `/status`
continuam acessíveis para observação. Disponibilidade é separada de conclusão/falha.

O retorno expõe imediatamente o último snapshot publicado. O cliente retoma seu
treino suspenso e só publica novamente após o ciclo normal de treino e agregação.
Os snapshots preservados na saída e no retorno são registrados para auditoria.
Requisições que atravessam o início da janela são identificadas; um 503 programado
é contado e ignorado sem novas tentativas.

Todos continuam até o orçamento terminar, independentemente de quantos ciclos
concluíram. O fim interrompe no próximo limite de minibatch e descarta trabalho
ainda não publicado, mas contabiliza seus passos e exemplos. O checkpoint final
corresponde ao último modelo publicado. Avaliação final, gravação e limpeza podem
terminar depois do orçamento; não representam treinamento adicional.

## Preparação

Execute os comandos abaixo na raiz do repositório, no PowerShell. Docker Desktop
precisa estar ativo, com contêineres Linux. As portas 8000–8007 devem estar livres.
O build inicial baixa os datasets; as execuções reutilizam o cache da imagem.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
docker info
$py = ".\.venv\Scripts\python.exe"
$runner = "scripts\run_johann_experiments.py"
& $py $runner list --dataset all
```

Não execute duas redes ao mesmo tempo: elas compartilham portas, e a contenção
alteraria os tempos medidos. O executor percorre datasets, cenários e sementes
sequencialmente e encerra o lote na primeira falha.

## Executar

Smoke funcional: 30 segundos, 800 exemplos globais antes da divisão e até 100
exemplos de avaliação. Exercita saída e retorno; não serve para concluir sobre
aprendizado ou comparar qualidade.

```powershell
& $py $runner smoke --dataset all --scenario concentrated-r --run-id smoke-01
# Opcional: conferir também toda a matriz, com custo maior.
& $py $runner smoke --dataset all --all --run-id smoke-matriz-01
```

Gerar Compose e a configuração resolvida, sem iniciar contêineres:

```powershell
& $py $runner render-compose --dataset mnist --scenario concentrated-r --run-id inspecao-01
docker compose -f results/johann/mnist/M0/concentrated-r/inspecao-01-s7/compose.json config --quiet
```

Cada diretório é exclusivo e não pode ser sobrescrito. Use outro `--run-id` para
executar depois de uma geração de Compose. Prefira os comandos do executor:
`up` sozinho deixa os clientes aguardando o início comum e a execução manual
não realiza todas as verificações de imagem antes de chamar `/start`.

Pilotos usam **validação**, semente 7, dez minutos no MNIST e trinta no CIFAR-10.

```powershell
& $py $runner pilot --dataset mnist --scenario concentrated-r --run-id piloto-01
& $py $runner pilot --dataset cifar10 --scenario concentrated-r --run-id piloto-01
# Ajustar o orçamento e a cadência somente na etapa de desenvolvimento:
& $py $runner pilot --dataset mnist --scenario concentrated-r --duration 900 --interval 10 --run-id piloto-02
```

### Usar o pacote congelado

`run` usa o **teste oficial**, as sementes 42–46 por padrão e exige o pacote.
O exemplo abaixo descreve uma seleção explícita; não representa toda a fila
principal. `--all` inclui os seis cenários do catálogo, inclusive controles sem
ausência, e não equivale aos quatro cenários da matriz principal do TCC.

```powershell
$frozen = "results/johann/desenvolvimento-docker-20261029/frozen"
# Inspeção offline: gera arquivos, não inicia contêineres nem usa o teste oficial.
& $py $runner render-compose --dataset all --methods M0 M1 M2 M3 M4 --frozen-bundle $frozen --run-id inspecao-congelada-01
# Coleta oficial, somente quando a fila final estiver fechada:
& $py $runner run --dataset mnist --scenario concentrated-r --methods M0 M1 M2 M3 M4 --frozen-bundle $frozen --run-id coleta-01
```

O leitor confere o hash de `source_snapshot.zip`, lê seus manifests como dados
e carrega os parâmetros de `selected_configuration.json`, verificando-os contra
`manifest.json`. O indicador global do manifesto e os indicadores da seleção
final são autoritativos; indicadores históricos internos do manifesto podem
permanecer falsos. Nenhum código do ZIP é importado ou extraído.

| Dataset | L de M2 | K de M3/M4 | λ de M4 | T |
|---|---:|---:|---:|---:|
| MNIST | 33750 | 64 | 0,1 | 2 |
| CIFAR-10 | 56250 | 64 | 0,5 | 2 |

No caminho congelado, o Compose usa IDs completos `sha256:…`, sem `build`, e
inicia com `up --no-build --pull never -d`. Todos os datasets selecionados precisam
ter a imagem local antes da primeira rede. Se faltar alguma, o executor termina
com instruções para restaurar `images.tar`; não faz build, pull ou load automático.
Tags são apenas informação descritiva e não decidem qual imagem será usada.

Antes de `/start`, o executor inspeciona os oito contêineres do projeto e confere
serviços, estado em execução e `.Image` com o ID esperado. Divergências ficam em
`outcome.json` e `container_inspection.json`, com logs coletados antes da limpeza.

`--frozen-bundle` também funciona em `pilot`, `smoke`, `list` e `render-compose`.
Não pode ser combinado com `--selected-config`, overrides de L/K/λ/T ou backend
local. Duração/cadência congeladas não admitem overrides, exceto em `smoke`:
nesse caso permanecem os limites funcionais de amostras e a execução recebe
`protocol_frozen=false` e `scientific_protocol_complete=false`, embora use
a imagem e os parâmetros dos métodos preservados. Pilotos/smokes sem pacote
continuam disponíveis para desenvolvimento.

Atalho PowerShell equivalente, com smoke nos dois datasets como padrão:

```powershell
.\experiments\johann\run_experiments.ps1 -Mode smoke -Dataset all -RunId smoke-atalho-01
# O wrapper também aceita -FrozenBundle $frozen.
```

Os manifests `mnist/manifest.json` e `cifar10/manifest.json` registram os parâmetros.
Classes, grupos e proporções estruturais são fixos nesta versão e validados pelo
executor. Ajustes nesses elementos exigem atualizar também o protocolo e os testes.
Hiperparâmetros iniciais: uma época local, lote 64, SGD com taxa 0,01, momentum 0,9,
uma thread PyTorch por cliente e as mesmas CNNs/normalizações dos experimentos originais.

## Conferir os resultados existentes (E1)

`audit-m0` confere os arquivos já disponíveis e lista comparações possíveis e
lacunas para E2. Não chama Docker, não treina, não avalia checkpoints e não
reescreve resumos ou gráficos de origem. O pacote congelado é usado como
referência de código, parâmetros e ambiente; as confirmações M0 registradas nele
precisam estar presentes nos resultados para conferir os dados por semente.

```powershell
& $py $runner audit-m0 --results-root results/johann --frozen-bundle $frozen --output-dir outputs/tcc/auditoria_E1_M0_2026-11-01
```

A pasta de saída deve ser nova ou vazia e ficar fora dos diretórios das
execuções e do pacote preservado. Ela recebe `relatorio.md`,
`inventario_execucoes.csv`, `comparacoes_disponiveis.csv`, `lacunas_E2.csv` e
`auditoria.json`. Este último registra hashes dos arquivos lidos e ausências.

Os pilotos antigos são conferidos entre si dentro de cada lote/semente. Sua
utilidade histórica é separada da compatibilidade com o congelamento atual.
O ID exato da imagem preservada é ancorado nos registros de confirmação. Para
comparar imagens de projetos Compose diferentes, somente os rótulos de projeto,
serviço e versão do Compose são ignorados; conteúdo, configuração de execução,
plataforma e dependências continuam sendo exigidos. Metadados de imagens de
outros métodos podem fornecer essa referência, sem ler suas avaliações.
Controles sem ausência usam os mesmos observadores do cenário correspondente.
A lista de lacunas se refere a M0, dois datasets, sementes 8/9 e seis cenários;
não inicia suas execuções e não inclui novas estimativas estatísticas.
Duplicatas e referências não verificáveis pedem revisão, sem escolher um
resultado automaticamente. Arquivos com erros são registrados individualmente;
smokes e configurações apenas geradas não completam controles científicos.
Dados de avaliação do teste oficial não são lidos para orientar desenvolvimento.

## Resultados, métricas e gráficos

```text
results/johann/<dataset>/<metodo>/<cenario>/<run_id>-s<semente>/
  compose.json, resolved_experiment.json, schedule.json
  image.json, container_dependencies.txt, working_tree.patch
  frozen_manifest.json, selected_configuration.json  Quando há pacote congelado
  container_inspection.json                        Conferência antes de /start
  client_<id>_metrics.jsonl       Ciclos publicados e participantes
  client_<id>_events.jsonl        Agenda, pausa real, retorno e pontos perdidos
  client_<id>_evaluations.jsonl   Captura, versão, métricas e duração da avaliação
  client_<id>_cost.json          Comunicação, computação e memória
  client_<id>_class_distribution.json
  client_<id>_{offline,returned,final}.pt
  client_<id>_confusion.json
  summary.json, summary.csv, summary.md
  temporal_groups.csv, diagnostics.csv, client_recovery.csv
  charts/*.png
  docker_start.log, container.log, last_status.json, outcome.json
```

No caminho congelado, `source_sha256` identifica as fontes preservadas, e o bloco
`orchestrator` registra commit, alterações, hashes e dependências do host atual.
`frozen_bundle` registra os hashes do manifesto, seleção e arquivo de fontes;
`expected_image_id`, `observed_image_ids` e `image_id` identificam as imagens
esperada e verificadas. As cópias de manifesto/seleção preservam os bytes usados.
O pacote original não é alterado. No desenvolvimento sem pacote permanecem os
campos de proveniência anteriores. As dependências do contêiner são exportadas.
Não publique datasets, checkpoints ou resultados no Git.

As avaliações usam snapshots publicados em uma thread observadora, fora do lock
do modelo. A cadência é vinte segundos no MNIST e sessenta no CIFAR-10, com
instantes reais de captura registrados. Pontos atrasados são pulados, sem rajadas
de avaliações compensatórias. Durante a ausência, métricas de um snapshot já
avaliado podem ser reutilizadas com `frozen=true`. O custo da observação compete
por CPU com o treino e é contabilizado; a cadência deve permanecer igual em pares.

São registrados perda, acurácia, macro-F1 sobre dez classes, recall e suporte por
classe, média do recall de 3/8 e das demais classes. Divisões por suporte zero
usam zero nas métricas descritivas; o desfecho de recall acompanhado exige suporte
positivo em ambas as classes. `diagnostics.csv` separa o modelo preservado, o
modelo local antes da primeira mistura e o modelo publicado depois dela.

Cada resumo também gera `charts/client_<id>_recall_by_class.png` e CSV, com uma
curva por classe, eixo Y entre 0 e 1 e os tempos reais das avaliações periódicas
e final. Suporte zero aparece como lacuna; as avaliações de diagnóstico não entram
nessas curvas. A cadência de observação permanece igual.
`charts/client_<id>_aggregation_contributions.png` e CSV mostram os pesos efetivos
das combinações de parâmetros por cliente e ciclo publicado, incluindo o modelo local;
cada coluna conhecida soma 100%. A escala de cores é sempre 0–100%, e ciclos sem
registro ficam cinza. Não é uma atribuição histórica de conhecimento. A intervenção
M3/M4 fica sem pesos, pois treino e destilação não são uma combinação convexa direta.

A propagação de `/start` tenta novamente em segundo plano enquanto o serviço
estiver ativo, mesmo após o treino local. Um vizinho que receber o início tarde
usa somente o tempo restante da janela original, sem alterar `start_at` ou o
término. Depois dessa janela, uma execução ainda não aceita retorna HTTP 422 com
`run_expired`; a propagação registra o erro e encerra as tentativas para aquele
vizinho. Uma execução já aceita continua respondendo `already_started` para os
mesmos `run_id` e `start_at`, inclusive depois de terminar.

O **desfecho principal** é a área normalizada do recall médio de 3/8 dos seis
clientes continuamente online, entre retorno e término. No controle sem ausência,
há resultados excluindo R e excluindo C para os respectivos pares. A integração
usa retenção do último valor entre observações reais; não extrapola além do
trecho observado. Lacunas maiores que duas cadências invalidam aquele trecho.
Cobertura abaixo de 100% produz desfecho ausente, acompanhado da cobertura obtida.

Recuperação significa alcançar o recall anterior à saída, com tolerância de dois
pontos percentuais, em três avaliações consecutivas. Ausência de recuperação é
registrada como `recovered=false`, sem eliminar a execução da tabela. Os gráficos
separam R, C, demais, retornantes e continuamente online e marcam a janela offline.

Custos incluem passos, exemplos, ciclos, tempo ativo de treino, avaliação e
requisições, além de payloads de modelos enviados/recebidos. Bytes não incluem
cabeçalhos HTTP, TCP ou outros overheads. Tempos somados por processo não são
tempo de parede da rede. A memória é o pico RSS do processo no Linux; em ambientes
sem essa informação, o valor é ausente, nunca zero presumido.

Regenerar resumos e analisar uma coleção:

```powershell
& $py $runner summarize results/johann/mnist/M0/concentrated-r/coleta-01-s42
& $py $runner summarize results/johann --run-id coleta-01 --mode collection
```

Uma coleção gera `collection.csv`, `collection.md`, `paired_comparisons.csv`,
`excluded_runs.json` e figuras de desfecho, contrastes e qualidade versus custo.
São comparadas diferenças R–C, interação concentração × ausência e cenários com
ausência versus seus controles. Intervalos de 95% usam bootstrap das diferenças
pareadas por semente, com 10.000 reamostragens e gerador fixo. A unidade independente
é a rede completa, não cada cliente ou ponto da curva. IC fica ausente com menos
de duas redes. Cinco sementes ainda implicam incerteza considerável.

Pilotos, smoke e coleta ficam separados na análise. Um par precisa compartilhar
orçamento, cadência e seleção global de dados. Repetições duplicadas da mesma
semente/cenário na mesma coleção são rejeitadas: selecione uma pasta que represente
uma única coleta. Execuções falhas e desfechos incompletos são identificados.
Os filtros `--run-id` (sem o sufixo de semente) e `--mode` permitem selecionar uma
coleta mantendo pilotos e tentativas anteriores na mesma árvore de resultados.

O pareamento verifica parâmetros, seleção global de dados, hashes do código,
camadas do sistema de arquivos da imagem, configuração de execução, plataforma e
dependências exportadas. IDs Docker distintos são aceitos quando esses elementos
coincidem: os rótulos `com.docker.compose.project`, `com.docker.compose.service` e
`com.docker.compose.version` são metadados de orquestração e não entram nessa
comparação. Os IDs originais continuam registrados para auditoria. Na ausência de
`image.json`, exige-se igualdade dos IDs disponíveis. Diferenças reais continuam
bloqueando a comparação, com os campos incompatíveis identificados na mensagem.
Essa regra também se aplica a resultados já existentes ao regenerar o resumo;
os treinamentos não precisam ser refeitos quando só esses metadados diferem.

## Diagnóstico e limpeza

O executor coleta logs antes de remover os contêineres. `outcome.json` informa
conclusão ou falha; `last_status.json` contém erros recentes e versões dos clientes.
Uma falha não deve ser substituída silenciosamente: preserve a pasta e identifique
uma nova execução com outro `--run-id`.

```powershell
Get-Content results/johann/mnist/M0/concentrated-r/smoke-01-s7/outcome.json
Get-Content results/johann/mnist/M0/concentrated-r/smoke-01-s7/container.log -Tail 80
docker compose -f results/johann/mnist/M0/concentrated-r/smoke-01-s7/compose.json ps -a
docker compose -f results/johann/mnist/M0/concentrated-r/smoke-01-s7/compose.json down
```

Os experimentos antigos continuam usando limite por rodadas e seu contrato de
treinador original. O modo temporal exige as capacidades opcionais
`train_local_controlled` e `evaluate_detailed`. Suas variáveis principais são
`METHOD`, `MERGE_LIMIT_EXAMPLES`, `REJOIN_BATCHES`, `DISTILLATION_WEIGHT`,
`DISTILLATION_TEMPERATURE`,
`RUN_DURATION_SECONDS`, `EVALUATION_INTERVAL_SECONDS`, `OFFLINE_CLIENT_IDS`,
`OFFLINE_START_FRACTION`, `OFFLINE_END_FRACTION`, `AUTO_START_HEAD`,
`JOHANN_DISTRIBUTION` e `EVALUATION_SPLIT`. `/start` aceita `start_at` como timestamp
Unix em segundos; repetir o mesmo identificador com outro instante gera conflito.
M3/M4 exigem também a capacidade opcional `train_reintegration` do treinador.

## Verificação

```powershell
& $py -m pytest
& $py -m compileall -q app tests scripts
```

Os testes cobrem partições, controle por relógio injetável, preservação do
otimizador na pausa, bloqueio HTTP, rede de oito clientes, métricas, recuperação,
integração temporal, pareamento e falhas de agendamento/execução. O smoke real
complementa os testes, mas não substitui pilotos nem comprova uma hipótese científica.
