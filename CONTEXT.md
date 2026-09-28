# Monitoramento Multimodal — Tech Challenge Fase 4

Sistema do hospital que monitora pacientes a partir de vídeo, áudio e sinais vitais para detectar sinais precoces de risco e alertar a equipe médica.

## Language

### Detecção

**Analisador**:
Componente que processa um tipo de dado (vídeo, áudio, sinais vitais, prescrição) e decide se há uma Anomalia.
_Avoid_: detector, módulo, pipeline

**Anomalia**:
Desvio, detectado por um Analisador, em relação ao padrão esperado para aquele paciente ou procedimento.
_Avoid_: outlier, evento, problema

**Evidência**:
Trecho do dado de origem que justifica uma Anomalia (frame de vídeo, trecho de transcrição, janela da série temporal).
_Avoid_: prova, anexo

**Sessão de Fisioterapia**:
Vídeo de um paciente executando um exercício com amplitude de movimento esperada conhecida.
_Avoid_: vídeo clínico, gravação

**Desvio de Execução**:
Anomalia de vídeo em que a postura do paciente sai da faixa esperada para o exercício (amplitude, compensação, assimetria).
_Avoid_: erro de postura, falha

**Queda**:
Anomalia de vídeo em que o paciente internado passa de posição em pé ou sentada para o chão.
_Avoid_: tombo, acidente

**Relatório de Sessão**:
Documento gerado automaticamente por vídeo analisado, listando os Desvios de Execução com sua Evidência.
_Avoid_: laudo, report

**Consulta**:
Gravação de áudio de um diálogo entre médico e paciente.
_Avoid_: atendimento, conversa, sessão

**Termo Crítico**:
Entidade clínica (sintoma, condição, medicamento) extraída da transcrição de uma Consulta que exige atenção da equipe.
_Avoid_: palavra-chave, keyword

**Alteração Vocal**:
Anomalia de áudio em que características acústicas da voz do paciente (não o conteúdo dito) indicam fadiga ou dificuldade respiratória.
_Avoid_: problema de voz, sintoma vocal

**Sinal Vital**:
Medida fisiológica periódica de um paciente internado: frequência cardíaca, frequência respiratória, SpO2 ou pressão arterial.
_Avoid_: métrica, sensor, vital

**Monitoramento Contínuo**:
Análise dos Sinais Vitais janela a janela, à medida que chegam, em vez de todos de uma vez ao final.
_Avoid_: tempo real, streaming, batch

**Mudança Terapêutica Atípica**:
Anomalia na evolução das prescrições de um paciente (salto de dose, início de droga de alta vigilância, trocas excessivas, duplicidade terapêutica).
_Avoid_: erro de prescrição, alteração de medicação

### Fusão

**Paciente de Demonstração**:
Paciente fictício que reúne dados reais de fontes públicas distintas (sinais vitais, prescrições, consulta, vídeo) sob uma única identidade.
_Avoid_: paciente mock, paciente sintético, paciente real

**Fusão**:
Combinação de Alertas de modalidades diferentes do mesmo paciente, numa janela de tempo, para reavaliar a severidade.
_Avoid_: integração, agregação, merge

**Alerta Composto**:
Alerta gerado pela Fusão quando Anomalias de duas ou mais modalidades se confirmam mutuamente; tem severidade maior que suas origens.
_Avoid_: super-alerta, alerta combinado

### Resposta

**Alerta**:
Registro gerado a partir de uma Anomalia e endereçado à equipe médica, com paciente, origem, severidade, Evidência e status.
_Avoid_: notificação, aviso, alarme

**Central de Alertas**:
O único canal pelo qual a equipe médica vê e trata Alertas.
_Avoid_: dashboard, fila, inbox

**Reconhecimento**:
Ato de um membro da equipe médica declarar que viu um Alerta e assumiu o tratamento dele.
_Avoid_: ack, aprovação, validação
