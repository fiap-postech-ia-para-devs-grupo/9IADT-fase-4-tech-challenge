# Monitoramento do Leito com câmera e microfone, e Evento Sonoro no Contrato do Alerta

O PDF pede vídeo e áudio para "identificar sinais precoces de risco" e "alertar a equipe médica em tempo real", e sugere o AudioSet como dataset. Até aqui, vídeo e áudio só eram analisados em lote ou por upload no Laboratório: nada ligava uma câmera ou um microfone a um Leito. Por isso, a Admissão passa a ligar ao paciente uma **câmera** (YOLOv8-pose vigiando Queda) e um **microfone** (AST treinado no AudioSet + Whisper tiny), e o Contrato do Alerta ganha o tipo `evento_sonoro` (origem `audio`) para alarme de equipamento, grito, gemido e chamado da equipe.

Os dispositivos são **simulados**: a webcam e o microfone do computador onde o app roda, ou gravações públicas tocadas em tempo real (UR Fall, Wikimedia Commons) e uma fala sintética pt-BR. A Fusão ganha a regra "Queda com o paciente gritando ou pedindo ajuda" (crítica).

## Considered Options

- **Reusar `termo_critico` ou `alteracao_vocal` para os sons:** rejeitada; misturaria conteúdo da Consulta e voz do paciente com o ambiente do quarto, e a Fusão (ADR 0002) depende dessa distinção.
- **Classificador próprio treinado em poucos exemplos:** rejeitada; sem dataset rotulado no prazo. O AST já é treinado no AudioSet, o dataset sugerido no PDF.
- **Streaming real (WebRTC/RTSP):** rejeitada para o escopo; a vigilância roda em thread no servidor, com a mesma interface que uma câmera IP teria.

## Consequences

- Bancos criados antes recebem os CHECKs novos por migração em `db.conectar` (a tabela é recriada com as linhas).
- Os modelos de som (~500 MB) são baixados do Hugging Face na primeira execução; a classificação leva ~2 s por janela de 5 s em CPU.
- Não há métrica formal dos sons: a validação é por gravações rotuladas pelo nome (alarme, grito, gemido, chamado). Fica nas Limitações do relatório.
- Cada Leito mantém um Alerta aberto por tipo de evento até o Reconhecimento, e o Acionamento por WhatsApp é um por Leito, para evitar fadiga de alarme.
