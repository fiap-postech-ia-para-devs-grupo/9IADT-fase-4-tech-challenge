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
Entidade clínica (sintoma, condição, medicamento) extraída da transcrição de uma Consulta, afirmada sobre o próprio paciente, que exige atenção da equipe. Sintoma negado ou de um familiar não é Termo Crítico.
_Avoid_: palavra-chave, keyword

**Alteração Vocal**:
Anomalia de áudio em que características acústicas da voz do paciente (não o conteúdo dito) indicam fadiga ou dificuldade respiratória.
_Avoid_: problema de voz, sintoma vocal

**Sinal Vital**:
Medida fisiológica periódica de um paciente internado: frequência cardíaca, frequência respiratória, SpO2 ou pressão arterial.
_Avoid_: métrica, sensor, vital

**Evento Sonoro**:
Anomalia de áudio captada pelo microfone do leito: alarme de equipamento, grito, gemido de dor ou o paciente chamando a equipe. Diferente da Alteração Vocal (voz da Consulta) e do Termo Crítico (conteúdo da Consulta).
_Avoid_: ruído, barulho, som suspeito

**NEWS2 parcial**:
Escore de alerta precoce (NEWS2) calculado só com FR, SpO2, PAS e FC; é um piso do NEWS2 real. Sua piora (subida de 2+ pontos em até 1 h chegando a 5+) gera Alerta.
_Avoid_: score, índice de risco

**Monitoramento Contínuo**:
Análise dos Sinais Vitais janela a janela, à medida que chegam, em vez de todos de uma vez ao final.
_Avoid_: tempo real, streaming, batch

**Mudança Terapêutica Atípica**:
Anomalia na evolução das prescrições de um paciente (salto de dose, início de droga de alta vigilância, trocas excessivas, duplicidade terapêutica).
_Avoid_: erro de prescrição, alteração de medicação

### Unidade

**Leito**:
Lugar da unidade de internação onde um paciente é monitorado; livre ou ocupado.
_Avoid_: quarto, cama, vaga

**Admissão**:
Entrada de um paciente num Leito, ligando as fontes de monitoramento (monitor, prescrição, Consulta, câmera e microfone).
_Avoid_: cadastro de paciente, check-in

**Monitoramento do Leito**:
Vigilância contínua pela câmera (Queda) e pelo microfone (Evento Sonoro) do Leito; os dispositivos são simulados com webcam/microfone do computador ou gravações.
_Avoid_: câmera ao vivo, streaming

**Plantonista**:
Profissional da equipe que assume o plantão na sessão; reconhece Alertas e assina a Evolução.
_Avoid_: usuário, operador, responsável

**Evolução**:
Registro clínico no prontuário do paciente: anotação livre ou a conduta registrada ao reconhecer ou resolver um Alerta.
_Avoid_: nota, comentário, log

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
O único canal pelo qual a equipe médica vê e trata Alertas (o Acionamento só avisa).
_Avoid_: dashboard, fila, inbox

**Luz de Chamada**:
Sinal visual de que um Leito tem chamado aberto: Alerta crítico, Queda, piora do NEWS2, grito ou chamado sem Reconhecimento. Há uma luz por Leito e a sirene da unidade na barra superior.
_Avoid_: alarme, notificação

**Acionamento**:
Aviso ao profissional de plantão por WhatsApp quando a Luz de Chamada acende; um por Leito até o Reconhecimento.
_Avoid_: notificação push, mensagem, alerta externo

**Escalonamento**:
Acionamento da equipe médica quando um chamado fica 5 minutos (do plantão) sem Reconhecimento.
_Avoid_: reenvio, insistência

**Episódio**:
Grupo de Alertas repetidos do mesmo paciente, regra e sinal a menos de 30 minutos um do outro, mostrado como um só cartão na Central.
_Avoid_: duplicado, cluster

**Reconhecimento**:
Ato de um membro da equipe médica declarar que viu um Alerta e assumiu o tratamento dele.
_Avoid_: ack, aprovação, validação
