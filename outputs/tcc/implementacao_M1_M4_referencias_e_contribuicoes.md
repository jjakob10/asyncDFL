# M1–M4: implementação, referências e contribuição investigada

Documento técnico de 29/10/2026. Projeto asyncDFL, experimento Johann.
Este registro separa a origem das regras da contribuição experimental. A busca foi
dirigida, não uma revisão sistemática nem uma demonstração de ineditismo.

## 1. Estado e classificação da entrega

As regras M1–M4, a instrumentação, os parâmetros, o executor e os contrastes foram
implementados em Python/PyTorch. A calibração científica e a coleta confirmatória
continuam separadas da verificação funcional. Os valores iniciais dos manifests
não são hiperparâmetros cientificamente selecionados; `protocol_frozen=false`.
As sementes 42–46 permanecem destinadas à coleta final.

| Parte | Proveniência | O que existe nesta implementação |
|---|---|---|
| M0 | Regra conhecida, adaptada à execução descentralizada | Média de parâmetros por amostras locais, com execução assíncrona sem servidor ou rodadas globais. |
| M1 | Regra reproduzida; protocolo adaptado | Média binária por idade e herança do máximo, aplicada sucessivamente aos vizinhos. Não reivindicada como novidade. |
| M2 | Regra reproduzida; protocolo adaptado | Limited merging com desigualdade estrita e limiar em visitas a exemplos. Não reivindicada como novidade. |
| M3 | Controle experimental adaptado | Substituição única no retorno pelo vizinho com maior idade observada e K minibatches supervisionados. A combinação específica não foi confirmada como método publicado integralmente. |
| M4 | Mecanismo conhecido + contribuição candidata | Mesma substituição e adaptação de M3, acrescentando preservação temporária por professor local congelado. Destilação não é novidade. |
| Cenário e avaliação | Contribuição experimental candidata | Associação controlada entre concentração de classes e indisponibilidade; retenção nos seis clientes online, contraste M4–M3 e custos explícitos. |

Não se deve escrever que M4 é um algoritmo comprovadamente inédito, que supera o
estado da arte ou que os pilotos provaram apagamento do snapshot preservado.
Resultados negativos também respondem à investigação.

## 2. M1: média ponderada por idade

**Fontes primárias.** Hegedűs, Danner e Jelasity (2021), *Decentralized learning
works: An empirical comparison of gossip learning and federated learning*, JPDC
148, 109–124, [DOI](https://doi.org/10.1016/j.jpdc.2020.10.006);
[texto integral institucional](https://publicatio.bibl.u-szeged.hu/19934/1/1-s2.0-S0743731520303890-main.pdf).
Danner et al. (2023), seção 3, também explicita a regra antecedente.

**Implementação dos autores consultada.**
[`LinearModel.java`, revisão c8e47f1](https://github.com/ormandi/Gossip-Learning-Framework/blob/c8e47f170a03880fb7cdd8b144fdd6f2dfe3d974/src/gossipLearning/interfaces/models/LinearModel.java),
métodos `update` e `merge`. O update de um exemplo incrementa a idade em um;
o update em lote acrescenta o número de exemplos. `merge` usa a proporção de
idades e atribui a maior idade ao resultado. São visitas, incluindo repetições.

Para dois modelos de parâmetros w e idades a:

```text
w' = (a_i*w_i + a_j*w_j) / (a_i + a_j)
a' = max(a_i, a_j)
```

Se ambas as idades forem zero, conservar o modelo corrente, conforme o código
consultado. Uma entrada de idade zero recebe peso zero diante de idade positiva.
O tamanho local do dataset não é multiplicado pela idade.

**Correspondência local.** `app/aggregation.py`, `aggregate_models("M1", ...)`.
Os testes em `tests/test_johann_methods.py` verificam valores numéricos, idades
zero, herança, múltiplos vizinhos e ordem.

## 3. M2: limited merging

**Fonte primária.** Danner, Hegedűs e Jelasity (2023), *Improving Gossip Learning
via Limited Model Merging*, ICCCI, [DOI](https://doi.org/10.1007/978-3-031-41774-0_28),
[manuscrito aceito](https://www.inf.u-szeged.hu/~jelasity/cikkek/iccci23.pdf),
seção 3 e Algoritmo 3, páginas 7–8 do PDF. A rotina `merge` do código dos autores
acima também contém `mergeLimit`, substituição e herança do máximo.

```text
se a_i > a_j + L: conservar w_i
senão, se a_j > a_i + L: copiar w_j
senão: executar a média de M1
idade resultante = max(a_i, a_j)
```

A igualdade à diferença L produz mistura. Não aplicar decaimento exponencial,
limiar relativo, limiar adaptativo ou peso adicional por quantidade de dados.

**Correspondência local.** `app/aggregation.py`, `aggregate_models("M2", ...)`.
Os testes cobrem diferenças abaixo, iguais e acima de L, nas duas direções,
pesos zero e compatibilidade dos modelos.

## 4. Adaptações comuns de M1/M2

Estas regras foram reimplementadas, não copiadas como um simulador completo.
As diferenças que precisam constar na metodologia são:

- CNNs MNIST/CIFAR-10, oito clientes, datasets locais e minibatches, em vez da
  configuração original de modelos lineares e amostras distribuídas.
- Pull HTTP para vários vizinhos fixos por ciclo; combinação binária por ID
  crescente. O artigo emprega gossip por eventos e seleção de peers.
- Treino local antes da coleta/mistura, preservando o fluxo do asyncDFL.
- Sem token account, envio parcial de parâmetros ou reprodução dos traces originais.
- SGD com taxa e momentum comuns aos métodos do estudo. O estado do otimizador
  não é transmitido nem misturado; um novo otimizador é criado a cada chamada de
  treino, como no adaptador Johann anterior. O código dos autores possui também
  operações sobre o otimizador; isso não foi reproduzido.

A combinação sucessiva com idade herdada pelo máximo não é associativa. Por
exemplo, três modelos de mesma idade recebem pesos finais 1/4, 1/4 e 1/2 na ordem
local, vizinho 1, vizinho 2. Registrar `merge_trace` e os pesos efetivos evita
apresentar essa execução como uma média conjunta uniforme. A ordem fixa pode
favorecer determinadas posições e deve ser reconhecida como limitação.

## 5. M3: substituição e adaptação supervisionada

**Antecedente publicado.** O Algoritmo 1 de Danner et al. (2023), seção 2.2,
descreve o gossip básico que recebe um modelo, substitui o local e atualiza com
dados locais. Esse antecedente fundamenta o mecanismo, não a política inteira.

**Adaptação do estudo.** Fora do retorno, M3 executa M0. No retorno, abandona o
ciclo privado ainda não publicado e respostas coletadas antes do retorno. Em uma
nova varredura, escolhe o snapshot com maior idade entre os vizinhos elegíveis;
em empate, menor ID. Copia esse modelo e executa exatamente K minibatches
supervisionados antes de uma única publicação, retomando M0 depois.

Idade mede maturidade de treinamento, não recência cronológica ou qualidade.
O vizinho escolhido pode ter idade menor que o snapshot preservado: a política
continua copiando o melhor disponível e registra a diferença. Sem vizinho
elegível, mantém a publicação antiga acessível e repete consultas dentro do
orçamento. Se a janela terminar, registra a intervenção incompleta.

**Correspondência local.** `app/timed_runtime.py`, `_reintegrate`, e
`experiments/johann/trainer.py`, `train_reintegration`.
M3 é o controle de substituição e treinamento adicional, sem alegação de novidade.

## 6. M4: preservação temporária por destilação local

**Fundamentação do mecanismo.**
[Hinton, Vinyals e Dean (2015)](https://arxiv.org/abs/1503.02531) fundamentam
alvos suavizados, temperatura e combinação com rótulos supervisionados.
[Li e Hoiem (2016), Learning without Forgetting](https://arxiv.org/abs/1606.09282),
seção III, fundamentam a preservação de respostas do modelo anterior nos dados
disponíveis durante adaptação. Este estudo não reproduz sua expansão de cabeças,
warm-up ou protocolo de novas tarefas.

M4 conserva o último snapshot publicado anterior à intervenção como professor
congelado. O estudante tem a mesma inicialização, dados, SGD e K de M3. Usar:

```text
p_t = softmax(logits_professor / T)
L = CE(rotulos, logits_estudante)
    + lambda * T^2 * KL(p_t || softmax(logits_estudante / T))
```

KL é reduzida por média sobre os exemplos (`batchmean`). O professor opera em
`eval`, com parâmetros sem gradiente e inferência sob `no_grad`. As duas perdas
usam os mesmos exemplos locais de treino e todas as dez classes. Não há dataset
público, gerador sintético, replay, identificação de classes raras ou consulta
à validação/teste pelo método. Depois de K minibatches, descartar o professor.

Com lambda zero, não construir nem executar o professor. A janela usa um DataLoader
próprio, de mesma semente para M3/M4 (`seed + 100000 + client_id`), sem consumir
o gerador do treino normal. A construção do professor usa deepcopy. Os testes
verificam a igualdade numérica de M3 e M4 com lambda zero sob entradas iguais,
e a imutabilidade dos parâmetros do professor.

**Trabalho próximo.**
[Askin et al., FedRevive, arXiv v3 de 09/10/2026](https://arxiv.org/html/2511.00655v3)
recuperam conhecimento de updates antigos com destilação data-free no servidor,
geração sintética e múltiplos professores. M4 não é uma reprodução de FedRevive:
usa um professor local, dados privados já disponíveis e uma janela de retorno.
Não herda as garantias ou os resultados daquele método.

**Contribuição candidata.** Investigar essa política estritamente local, com
orçamento limitado, quando a indisponibilidade se associa à concentração de
classes. Medir se a retenção alcança também os participantes continuamente online
e quanto custa. A composição é uma hipótese de contribuição; sua superioridade
e seu ineditismo não são pressupostos.

## 7. Idade, publicação e custos

`X-Training-Age` usa unidade `X-Training-Age-Unit: example_visits_v1`.
O runtime publica modelo, idade e versão sob o mesmo lock. A idade inicial é zero;
cada minibatch completo acrescenta seu tamanho real. Trabalho privado descartado
permanece nos contadores de custo, mas não na idade publicada. A destilação não
soma a idade do professor à do estudante: este herda a idade do vizinho copiado
e acrescenta somente seus próprios exemplos processados.

M1–M4 não usam metadados de idade ausentes ou inválidos. M0 aceita peers antigos
para compatibilidade operacional; se algum snapshot incorporado não possui idade
válida, sua idade herdada fica desconhecida e os headers de idade são omitidos.
Isso impede inventar uma linhagem completa. Experimentos entre métodos usam a
mesma versão nova do runtime, sem misturar imagens antigas.

Há registros separados de passos/exemplos totais e de reintegração, inferências,
exemplos, tempo e bytes de parâmetros do professor, espera por vizinhos, trabalho
descartado, snapshots diagnósticos, comunicação e pico RSS quando disponível.
O tempo de treino inclui a inferência do professor: não somar `teacher_seconds`
novamente ao total. Bytes do professor descrevem parâmetros/buffers, não todo o RSS.
Custos de inferência não representam uma medida completa de FLOPs.

Os diagnósticos são avaliados após o orçamento, sobre cópias CPU com instantes
registrados. Os retornantes guardam `returned.pt`, `after_reinitialization.pt` e
`after_reintegration.pt`; M0–M2 guardam também os diagnósticos de primeira mistura.
O snapshot antigo fica acessível até a publicação completa da intervenção.
Publicações provisórias não são expostas. Sem ausência, M3/M4 percorrem M0.

`effective_aggregation_weights` registra combinações de parâmetros M0–M2 e dos
ciclos normais de M3/M4, incluindo zeros. Na intervenção M3/M4 o campo é nulo:
treino supervisionado/destilação não possui atribuição convexa direta de parâmetros.
Os gráficos não tratam o professor como uma porcentagem de agregação.

## 8. Execução e calibração reproduzível

Os manifests fixam MNIST em 600 s/20 s e CIFAR-10 em 1800 s/60 s. Preservam
classes 3/8, R={0,4}, C={2,6}, topologia e seis cenários existentes. As identidades
não são permutadas entre sementes; essa restrição substitui a sugestão inicial
de permutação no protocolo e deve constar nas limitações.

Exemplos na raiz do repositório, PowerShell:

```powershell
$py = '.venv/Scripts/python.exe'
$runner = 'scripts/run_johann_experiments.py'
& $py $runner list --dataset all --methods M0 M1 M2 M3 M4
& $py $runner smoke --dataset all --methods M1 M2 M3 M4 --scenario concentrated-r --run-id verificacao-metodos
& $py $runner pilot --dataset mnist --methods M2 --merge-limit-multiplier 5 --run-id piloto-l5n
& $py $runner calibrate --dataset all --run-id busca-validacao
& $py $runner select-calibration results/johann --run-id busca-validacao
& $py $runner pilot --dataset all --methods M0 M1 M2 M3 M4 --scenario concentrated-r --seeds 8 9 --selected-config results/johann/busca-validacao-selection.json --run-id confirmacao-validacao
```

`calibrate --render-only` gera os onze candidatos por dataset sem iniciar Docker;
esses diretórios não podem depois ser sobrescritos por uma execução. Cada
tentativa precisa de identificador próprio. Redes são executadas sequencialmente.
Novos caminhos: `results/johann/<dataset>/<method>/<cenario>/<run_id>-s<seed>/`.
Os resumos continuam aceitando os caminhos antigos de M0.

Quando Docker não estiver disponível, `smoke --backend local` executa oito
processos Python independentes, com os mesmos treinadores e trocas HTTP reais.
É um fallback apenas funcional, limitado a smoke/validação: registra
`execution_backend=local_smoke`, usa dados em `results/johann/local_data` e não
substitui a validação dos contêineres nem uma calibração científica. O pareamento
verifica o backend. Portas já ocupadas são recusadas e somente processos criados
pelo executor são encerrados.

```powershell
& $py $runner smoke --dataset all --methods M1 M2 M3 M4 --rejoin-batches 2 --duration 45 --backend local --run-id verificacao-local
```

A busca predeclarada usa exclusivamente validação, `concentrated-r` e semente 7:
M2 com L={N,5N,10N}; M3 com K={16,64}; M4 com ambos os K, lambda={0.1,0.5,1}
e T=2. N é a maior capacidade local: 6750 no MNIST completo, 5625 no CIFAR-10.
Na seleção, conferir N contra os artefatos de distribuição efetivamente medidos.

Selecionar maior AUC de M3 para definir K; depois maior AUC de M4 naquele K para
lambda. Selecionar L pela maior AUC de M2. Desempates: menor K, menor lambda e
maior L. Não eliminar candidatos falhos ou incompletos para escolher entre os
restantes: o seletor exige toda a grade e a compatibilidade de dados/código/imagem.
A configuração sugerida não congela o protocolo automaticamente.

Confirmar nas sementes 8/9 sem reajuste automático. Uma revisão exige nova
calibração identificada. Somente após conferência e congelamento, iniciar `run`,
que usa o teste oficial e sementes 42–46. Os smokes não selecionam parâmetros.
A confirmação acima usa `concentrated-r`; a matriz completa pertence à coleta
posterior. Nos dois datasets, a grade soma 7 h 20 min e a confirmação 6 h 40 min
de orçamento de treino sequencial, antes de preparação e reserva computacional.

## 9. Análise e verificação

As repetições são identificadas por dataset, modo, método, configuração, cenário
e semente. Configurações diferentes não são agrupadas como réplicas. Dentro do
método, o pareamento permanece estrito. Entre métodos, permitir apenas diferenças
de método e parâmetros previstos, mantendo dados, tempo, treinamento comum,
fonte, imagem e dependências compatíveis.

Contraste principal: M4–M3 em `concentrated-r`, obrigatoriamente com K comum.
Comparações auxiliares: M1–M4 versus M0 e contrastes de cenário por método.
O desfecho é a AUC normalizada do recall acompanhado dos seis clientes online
na janela pós-retorno. Manter bootstrap pareado por rede completa e sementes,
sem tratar clientes ou pontos temporais como réplicas independentes. Relatar
cobertura incompleta, falhas e qualidade versus custo sob prazo total igual.

Validação funcional: regressões de fórmulas, fronteiras de L, contagens, publicação
atômica, retorno único, ausência de peers, vizinho de menor idade, descarte no fim,
professor congelado, lambda zero, equivalência sem ausência, configuração, gráficos,
pareamento e seleção da grade. Executar:

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m compileall -q app tests scripts
```

Verificação em 29/10/2026: **329 testes aprovados**, compilação Python concluída e
oito Compose validados com `docker compose ... config --quiet`. Uma advertência
de depreciação já existente no Starlette/AnyIO não impediu a suíte.

Smokes funcionais: coleção `metodos-local-20261029`, validação, semente 7,
800 exemplos globais antes do split, 45 segundos e K=2 para M3/M4. M1–M4
concluíram em **MNIST e CIFAR-10**: oito redes, 64 execuções de clientes. Cada
M3/M4 teve uma intervenção em cada um dos dois retornantes; M4 teve duas
inferências do professor por retornante, enquanto M3 não executou professor.
Conferiram-se eventos, idades, pesos, contadores e hashes das fontes contra os
arquivos desta entrega. Auditoria: `results/johann/verificacao_M1_M4_20261029.json`.

Na primeira tentativa, o Docker Desktop relatou falha no socket `dockerInference`;
os smokes foram realizados com processos locais. No reteste de **29/10/2026**, o
Docker Engine **29.6.2** respondeu normalmente. **M0–M4 passaram nos dois datasets
em contêineres**: dez redes, 80 execuções de clientes, validação, semente 7,
60 segundos e K=16. Conferiram-se idades, retorno único, passos do estudante,
inferências do professor, memória RSS Linux e hashes das fontes. Auditoria:
`results/johann/verificacao_docker_M0_M4_20261029.json`.

A grade completa de calibração em validação foi iniciada. A sequência executa
22 candidatos na semente 7, faz a seleção predeclarada, confirma vinte redes nas
sementes 8/9 e só então preserva fonte, dependências, configurações e imagens
Docker. Ela para na primeira falha, sem eliminar candidatos desfavoráveis ou
reajustar parâmetros automaticamente. O acompanhamento é atualizado em
`results/johann/desenvolvimento-docker-20261029/andamento.md` e `workflow_state.json`.
Até a conclusão desse ciclo, não considerar o protocolo congelado.
Não confundir smokes com validação das hipóteses ou coleta final; o teste oficial
e as sementes 42–46 continuam reservados.

## 10. Trabalhos relacionados e limites da alegação

- [FedNTD, Lee et al., NeurIPS 2022](https://arxiv.org/abs/2106.03097): preservação
  de conhecimento global durante treino local, com destilação nas classes não
  verdadeiras. M4 usa todas as classes e professor anterior local, não esse filtro.
- [Flashback, Aljahdali et al., 2024](https://arxiv.org/abs/2402.05558): esquecimento
  tanto no treino local quanto na agregação, com destilação dinâmica. Evita alegar
  que reconhecer ou mitigar esquecimento em FL seja novo.
- [CAFF, Martínez Beltrán et al., 2026](https://doi.org/10.1016/j.comnet.2026.112225):
  cache por vizinho, filtragem de atraso e limiar dinâmico em DFL. Não implementado
  como M5. A caracterização consultada foi limitada ao resumo/metadados acessíveis;
  não atribuir limitações específicas ao método sem ler seu texto integral.

O estudo é uma investigação de mecanismo em oito clientes e dois datasets, sem
garantia de superioridade geral. Pausa preserva memória; não há reinício real de
processo, concept drift, personalização ou recuperação de checkpoint. A ordem de
trocas assíncronas varia entre execuções, mesmo com sementes pareadas.
Sob tempo total igual, M4–M3 avalia a política inteira: o custo do professor pode
atrasar a publicação e alterar quais snapshots os vizinhos recebem. O contraste
não isola o termo KL de todos os efeitos do escalonamento assíncrono.

## 11. Referências em BibTeX

```bibtex
@article{hegedus2021decentralized,
  author = {Heged{\H u}s, Istv{\'a}n and Danner, G{\'a}bor and Jelasity, M{\'a}rk},
  title = {Decentralized learning works: An empirical comparison of gossip learning and federated learning},
  journal = {Journal of Parallel and Distributed Computing},
  volume = {148}, pages = {109--124}, year = {2021},
  doi = {10.1016/j.jpdc.2020.10.006}
}
@inproceedings{danner2023limited,
  author = {Danner, G{\'a}bor and Heged{\H u}s, Istv{\'a}n and Jelasity, M{\'a}rk},
  title = {Improving Gossip Learning via Limited Model Merging},
  booktitle = {Computational Collective Intelligence}, year = {2023},
  doi = {10.1007/978-3-031-41774-0_28},
  url = {https://www.inf.u-szeged.hu/~jelasity/cikkek/iccci23.pdf}
}
@misc{hinton2015distilling,
  author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  title = {Distilling the Knowledge in a Neural Network},
  year = {2015}, eprint = {1503.02531}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/1503.02531}
}
@inproceedings{li2016lwf,
  author = {Li, Zhizhong and Hoiem, Derek},
  title = {Learning without Forgetting},
  booktitle = {Computer Vision -- ECCV 2016}, pages = {614--629}, year = {2016},
  doi = {10.1007/978-3-319-46493-0_37},
  note = {Texto arXiv consultado: 1606.09282v3, versao estendida de 2017}
}
@misc{askin2026fedrevive,
  author = {Askin, Baris and Roth, Holger R. and Sun, Zhenyu and Joe-Wong, Carlee and Joshi, Gauri and Xu, Ziyue},
  title = {Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning},
  year = {2026}, eprint = {2511.00655}, archivePrefix = {arXiv},
  note = {Versao 3, 9 de outubro de 2026; primeira versao em 2025},
  url = {https://arxiv.org/html/2511.00655v3}
}
@inproceedings{lee2022fedntd,
  author = {Lee, Gihun and Jeong, Minchan and Shin, Yongjin and Bae, Sangmin and Yun, Se-Young},
  title = {Preservation of the Global Knowledge by Not-True Distillation in Federated Learning},
  booktitle = {Advances in Neural Information Processing Systems}, year = {2022},
  url = {https://arxiv.org/abs/2106.03097}
}
@misc{aljahdali2024flashback,
  author = {Aljahdali, Mohammed and Abdelmoniem, Ahmed M. and Canini, Marco and Horv{\'a}th, Samuel},
  title = {Flashback: Understanding and Mitigating Forgetting in Federated Learning},
  year = {2024}, eprint = {2402.05558}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2402.05558}
}
@article{martinez2026caff,
  author = {Mart{\'i}nez Beltr{\'a}n, Enrique Tom{\'a}s and Gash, Eduard and Bovet, G{\'e}r{\^o}me and Huertas Celdr{\'a}n, Alberto and Stiller, Burkhard},
  title = {Asynchronous Cache-based Aggregation with Fairness and Filtering for Decentralized Federated Learning},
  journal = {Computer Networks}, volume = {281}, pages = {112225}, year = {2026},
  doi = {10.1016/j.comnet.2026.112225}
}
```

## 12. Finalização do desenvolvimento Docker — execução de 29/10/2026

Estado final do pipeline: **completed**.

Artefatos e relatório: `results/johann/desenvolvimento-docker-20261029/andamento.md`.
Redes de calibração/confirmação concluídas: 42/42.

Fonte, imagens e configuração preservadas em `C:\Users\jjako\Documents\ChatGPT\asyncDFL\results\johann\desenvolvimento-docker-20261029\frozen`.
A grade foi selecionada na validação e confirmada nas sementes 8/9 sem reajuste automático. O teste oficial não foi usado.
