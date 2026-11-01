# TO-DO dos experimentos Johann

Checklist consolidado de 31/10/2026, separado em pendências técnicas,
experimentais, textuais e de organização:
[checklist do TCC](../../outputs/tcc/checklist_pendencias_TCC_2026-10-31.md).
O histórico abaixo permanece como registro; a fila principal atual tem quatro
cenários, com controles sem ausência definidos separadamente.

## Conferência E1 — concluída em 01/11/2026

- [x] `audit-m0`: inventário de arquivos existentes, compatibilidade histórica e
  com o congelamento, cobertura temporal, comparações e lista de lacunas para E2.
  Suíte completa: 392 testes; compilação e conferência do diff aprovadas.

Resultado: 40 registros M0, quatro confirmações recentes reaproveitáveis e
20 situações sem execução compatível com o congelamento (dez MNIST, dez CIFAR-10).
As matrizes antigas completas seguem válidas para seus próprios pilotos.
Nenhuma execução de E2 foi iniciada; T2 continua pendente.

[Relatório E1](../../outputs/tcc/auditoria_E1_M0_2026-11-01/relatorio.md).

## M1–M4 — implementação de 29/10/2026

- [x] Regras M1/M2 por pares, idade por visitas a exemplos e publicação atômica.
- [x] M3/M4, janela local comum, professor congelado e custos separados.
- [x] Executor por método, grade/seletor de calibração, análise M4–M3 e referências.
- [x] Suíte completa (329 testes), compilação, oito Compose validados e oito
  smokes com HTTP local real (M1–M4 em MNIST/CIFAR-10), sem teste oficial.
- [x] Verificar os smokes também em contêineres: dez redes M0–M4 em MNIST/CIFAR-10.
- [x] Executar a grade de validação (semente 7): 22 candidatos completos.
- [x] Confirmar nas sementes 8/9 sem reajuste automático: vinte redes completas.
- [x] Preservar fonte, imagens, dependências e parâmetros em
  `results/johann/desenvolvimento-docker-20261029/frozen/` (30/10/2026).
- [x] T1: executor exige `--frozen-bundle` na coleta final, fixa imagens por ID,
  carrega parâmetros preservados e confere os oito contêineres antes de `/start`.
  Entrega de 01/11: 374 testes, compilação e dez Compose validados estaticamente.
  Os defaults dos manifests continuam de desenvolvimento; E3 ainda está pendente.

Parâmetros selecionados: MNIST L=33750, K=64, lambda=0.1; CIFAR-10 L=56250,
K=64, lambda=0.5; T=2 em ambos. M3 usa o mesmo K de M4.
O congelamento é um snapshot externo: esta atualização documental não altera
os arquivos nem as imagens preservados nesse snapshot.

Referências e limites da contribuição:
`outputs/tcc/implementacao_M1_M4_referencias_e_contribuicoes.md`.

## MNIST — próxima etapa após os pilotos

Os seis cenários foram executados nas sementes 8 e 9, usando validação,
600 segundos por rede e avaliações a cada 20 segundos. As 12 redes tiveram
cobertura completa do desfecho. Coleção: `calibracao-intervalo20-20261023-182326`.

- [x] Preservar o protocolo selecionado, a fonte e as dependências antes da coleta final.
- [ ] Executar M0–M4 nos quatro cenários principais nas sementes **42–46**, usando
  **600 segundos** e **cadência de 20 segundos**, em sequência, mais os controles
  sem ausência definidos na fila. Preparar primeiro o caminho congelado do
  executor; a seleção de parâmetros isolada não fixa a imagem usada.
- [ ] Conferir conclusão, cobertura, eventos de pausa/retorno e compatibilidade
  dos pares antes da análise.
- [ ] Comparar cada ausência com seu controle sem ausência, usando os mesmos
  clientes observadores; calcular contrastes pareados e bootstrap por execução
  completa. Examinar também diferenças anteriores à saída e custos.
- [ ] Manter os pilotos de validação separados da coleta final de teste.

Os pilotos sugerem maior déficit nas classes acompanhadas quando R concentra
essas classes e fica offline, sobretudo durante a ausência. Duas sementes e
diferenças anteriores à saída ainda impedem uma conclusão confirmatória.
MNIST já se aproxima do teto antes da pausa; qualquer mudança de duração,
agenda ou hiperparâmetros exige uma nova calibração identificada antes da coleta.

Relatório dos pilotos:
`outputs/johann/analises/calibracao-intervalo20-20261023-182326-matriz-completa/analise.md`.
