# TO-DO dos experimentos Johann

## MNIST — próxima etapa após os pilotos

Os seis cenários foram executados nas sementes 8 e 9, usando validação,
600 segundos por rede e avaliações a cada 20 segundos. As 12 redes tiveram
cobertura completa do desfecho. Coleção: `calibracao-intervalo20-20261023-182326`.

- [ ] Congelar o protocolo, o código e as dependências antes da coleta final.
- [ ] Executar os seis cenários nas sementes **42, 43, 44, 45 e 46**, usando
  **600 segundos** e **cadência de 20 segundos**, em sequência. Usar o comando
  `run`, que avalia no **teste oficial**, e um novo identificador de coleção.
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
