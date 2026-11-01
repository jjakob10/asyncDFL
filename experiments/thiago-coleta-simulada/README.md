# Thiago — coleta simulada por round

Seis exemplos com seis clientes em anel bidirecional: MNIST, CIFAR-10 e HARTH,
cada dataset nos modos `accumulative` e `sliding_window` com janela de três
rounds. Os pares usam o mesmo estoque, agenda, semente, arquitetura e teste.
São exemplos de desenvolvimento, sem reivindicação de calibração científica.

## Executar

Na raiz do repositório, com `.venv` e Docker Desktop em execução:

```powershell
$python = ".\.venv\Scripts\python.exe"
$runner = "scripts\run_collection_experiments.py"
& $python $runner list
& $python $runner render-compose --all --run-id inspecao
& $python $runner validate --all --run-id verificacao
& $python $runner smoke --all --run-id smoke-01
& $python $runner run --all --run-id coleta-01
```

`render-compose` valida a estrutura das agendas, sem baixar datasets nem iniciar
Docker. `validate` constrói as imagens e verifica todas as cotas contra os
estoques reais em um contêiner temporário. `run` e `smoke` fazem essa mesma
validação para todos os casos selecionados antes de iniciar qualquer cliente.
As imagens registram downloads e preparação no build; os serviços não baixam
dados ao iniciar. Cada execução exige um `run-id` e diretório novos.

Use `--dataset mnist`, `--dataset cifar10` ou `--dataset harth` para selecionar
um par; também é possível informar IDs individuais exibidos por `list`.
`--manifest caminho.json` aceita um manifesto editado, com caminhos das agendas
relativos a esse manifesto. `--smoke` usa a agenda reduzida também em `validate`
ou `render-compose`. O script `run_experiments.ps1 -RunId coleta-01` executa
os seis casos; acrescente `-Smoke` para os seis smokes.

O smoke usa cinco rounds, duas imagens por classe selecionada ou duas janelas
HARTH por round, e até 200 exemplos de teste. Não reduz o estoque de treino nem
trunca cotas. A execução normal usa dez rounds, batch 64, uma época local,
SGD com learning rate 0,01 e momentum 0,9, e semente 42.

Os projetos Compose têm nomes próprios por execução. As portas publicadas são
8000–8005; execute as redes sequencialmente e mantenha essas portas livres.
O executor salva logs e remove somente a rede que acabou de executar, inclusive
em caso de falha. Os arquivos de resultados são preservados.

## Liberação e retenção

Uma partição fixa define o estoque local; liberar um exemplo nunca o transfere
para outro cliente. Cada exemplo chega no máximo uma vez durante a execução.
No início do round local, a agenda determina as chegadas e o conjunto de treino:

- `accumulative`: união de todas as chegadas até o round atual.
- `sliding_window`: união das chegadas dos últimos N rounds, incluindo o atual.
  Com N=1 somente a chegada atual participa; com N=3 participam o round atual
  e os dois anteriores.

Rounds vazios avançam a janela. Dados expirados saem do treino, mas o modelo
mantém o que aprendeu. Épocas locais podem reapresentar os dados ativos;
essas reapresentações não contam como coleta nova. O teste é fixo e nunca entra
na coleta. Cada cliente avança seus rounds independentemente, sem barreira.

Clientes sem dados ativos pulam o treino, mas continuam agregando e avaliando.
Neste recurso, os pesos de agregação são as quantidades de exemplos ativos
usadas na última publicação de cada participante, sem multiplicar pelas épocas.
Pesos zero ficam fora da média. Sem nenhum peso positivo, conserva-se o modelo.
Antes da primeira publicação, o peso é zero. Modelo, versão e contagem são
publicados atomicamente; uma chegada ainda em treinamento não altera o peso
do snapshot servido a vizinhos.

## Editar a agenda

`agendas/mnist.json` e `agendas/cifar10.json` têm uma entrada para cada cliente
0–5, com cotas por classe original e por round. Exemplo de uma entrada:

```json
"0": [
  {"rounds": [1, 3], "counts_by_class": {"0": 40, "1": 20}},
  {"rounds": [4, 6], "counts_by_class": {"2": 60}}
]
```

O documento completo tem `version: 1`, `selection: "by_class"` e `clients`.
Os intervalos são inclusivos; a primeira regra acima libera 40 exemplos da
classe 0 e 20 da classe 1 **em cada um** dos rounds 1, 2 e 3. Rounds omitidos
não liberam dados. Não há repetição da sequência ao esgotar o estoque.

Quantidades devem ser inteiros não negativos, N deve ser inteiro positivo,
e todos os clientes precisam ser declarados, mesmo os que tenham uma lista
vazia. A validação rejeita classes desconhecidas, chaves JSON duplicadas,
intervalos sobrepostos ou fora da execução e pedidos superiores ao estoque.
O erro identifica cliente, round, classe quando aplicável, pedido e disponível.
Não existe substituição por outra classe nem redução automática de cotas.

Nos exemplos de imagem, a partição é IID. As filas por classe usam a semente
e o ID do cliente. As classes são divididas em `[0,1,2,3]`, `[4,5,6]`, `[7,8,9]`,
com rotação por `client_id % 3` nas fases 1–3, 4–6 e 7–10. Cada classe da fase
libera 100 exemplos por round. As agendas de smoke usam fases 1–2, 3–4 e 5.

## HARTH

Fonte: [UCI HARTH, DOI 10.24432/C5NC90](https://archive.ics.uci.edu/dataset/779/harth),
de Logacjov, Kongsvold, Bach, Bårdstu e Mork. Licença CC BY 4.0.
São gravações de 22 participantes, dois acelerômetros e frequência de 50 Hz.
Usamos `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`.

As janelas têm 250 registros (cinco segundos), sem sobreposição. Uma diferença
entre timestamps fora de 20 ± 10 ms interrompe o segmento; não se atravessam
lacunas nem participantes. Segmentos finais incompletos são descartados.
Timestamps não crescentes, valores não finitos e códigos desconhecidos geram
erro. A classe da janela é a majoritária; empates escolhem o menor código.
Entradas são `float32` em g, sem normalização ajustada nos dados futuros.

Os códigos originais `1,2,3,4,5,6,7,8,13,14,130,140` correspondem, nessa ordem,
aos alvos internos `0..11`: walking, running, shuffling, stairs ascending,
stairs descending, standing, sitting, lying, cycling sit, cycling stand,
cycling sit inactive, cycling stand inactive. A configuração conserva os códigos
originais; o treinador converte somente os alvos entregues ao modelo.

Os IDs dos participantes são ordenados e permutados com semente fixa 42.
Os quatro primeiros formam o teste compartilhado; os outros 18 formam seis
grupos de três para treino. Este é o protocolo dos exemplos, não uma divisão
oficial da UCI. Os grupos nunca compartilham participantes com o teste.
Dentro do cliente, as janelas seguem `(timestamp inicial, ID do participante)`.

Para preservar essa cronologia, `agendas/harth.json` usa
`selection: "chronological"` e regras como
`{"rounds": [1, 10], "count": 100}`. Cada round libera as próximas 100 janelas.
Não há cotas por classe nem reorganização das gravações por atividade.

A CNN 1D usa convoluções de 32 e 64 canais, kernels 5, ReLU, max pooling
intermediário, pooling global e uma saída de 12 classes. O treinador fornece
loss, accuracy, recall e suporte por classe, além da matriz de confusão final.

O build prepara arrays `.npy` e metadados com hashes SHA-256 por arquivo e
participante. Os clientes leem arrays com memory mapping. Também é possível
preparar dados locais e validar sem Docker:

```powershell
& $python -m scripts.experiment_tools.prepare_data --trainer-module experiments.harth.trainer --data-dir results/coleta-data
& $python $runner validate --all --dataset harth --data-dir results/coleta-data --run-id verificacao-local
```

Para MNIST/CIFAR-10, o mesmo módulo de preparação aceita o módulo do treinador correspondente.
Dados e downloads devem permanecer em `results/` ou fora do repositório.

## Auditoria e compatibilidade

Resultados ficam em `results/thiago-coleta-simulada/<experimento>/<run-id>/`:

- `compose.json`, `resolved_experiment.json`, `schedule.json`: execução e agenda.
- `preflight.json`: hashes dos planos e estoques dos seis clientes. Os serviços
  recusam iniciar se seu plano não coincidir com o validado.
- `client_<id>_collection_plan.json`: chegadas resolvidas por identificador,
  hash do plano, modo, janela e metadados da fonte e da separação HARTH.
- `client_<id>_collection.jsonl`: liberações e expirações por round, com run-id
  e hash do plano. Permite reconstruir o conjunto ativo sem guardar entradas.
- `client_<id>_metrics.jsonl`: loss, accuracy, pesos e campo `collection`, com
  contagens e distribuições dos novos, acumulados, expirados no round,
  expirados totais e ativos, além de estoque restante e ocorrência de treino.
- Checkpoints finais, matrizes de confusão, resumos, gráficos e `container.log`.

As novas configurações são `COLLECTION_MODE` (`disabled`, `accumulative` ou
`sliding_window`), `COLLECTION_WINDOW_ROUNDS`, `COLLECTION_SCHEDULE_PATH` e
`COLLECTION_PREFLIGHT_PATH` (opcional fora do executor). Coleção habilitada exige
`collection_source` no `DataBundle` do treinador e execução por rounds.
Sem configurar o recurso, os treinadores, Composes, pesos mínimos e o runtime
temporal dos experimentos anteriores mantêm seu comportamento.


## Organização do código

Os treinadores são `experiments.mnist.trainer`, `experiments.cifar10.trainer`
e `experiments.harth.trainer`. Os dois treinadores de imagem fornecem o estoque
somente quando a coleta está habilitada. Não há um segundo treinador ou modelo
para os modos de retenção.

As funções reutilizáveis ficam em `scripts.experiment_tools`: `catalog` e
`topology` descrevem experimentos; `compose` gera os serviços; `collection`
acrescenta a configuração de coleta; `validation` valida os estoques;
`execution` monitora e encerra redes; `reporting` gera resumos; `prepare_data`
invoca a preparação oferecida pelo treinador. A lógica de chegadas e retenção
permanece em `app.collection`.

Os executores de dados fixos e de coleta continuam separados. Manifestos e
Composes novos usam os módulos originais dos treinadores. Configurações geradas
antes da reorganização devem ser regeneradas; seus resultados continuam
legíveis e não são modificados pela migração.
