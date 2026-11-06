# Alteração Vocal fica fora da Fusão

O classificador acústico é treinado em fala curta e controlada do Coswara e dispara em 17 de 18 Consultas do PriMock57, fala espontânea de pacientes simulados. Um Alerta que dispara em quase toda Consulta não confirma nada: mantido na regra "Sinal Vital (SpO2/FR) + áudio respiratório", quase todo Sinal Vital respiratório viraria Alerta Composto crítico. Por isso, só o Termo Crítico respiratório alimenta essa regra; a Alteração Vocal continua aparecendo como Alerta próprio, no Laboratório e no relatório.

## Considered Options

- **Manter a regra e declarar a limitação:** rejeitada, porque o Alerta Composto crítico, clímax da demonstração, ficaria quase garantido, sem mérito do modelo.
- **Recalibrar o limiar para o PriMock57:** rejeitada, porque o PriMock57 não tem rótulos de Alteração Vocal para validar a calibração.

## Consequences

- Para reabrir esta decisão, é preciso medir a taxa de disparo em fala espontânea rotulada.
- O Alerta Composto crítico depende de a Consulta de um Paciente de Demonstração conter um Termo Crítico respiratório.
