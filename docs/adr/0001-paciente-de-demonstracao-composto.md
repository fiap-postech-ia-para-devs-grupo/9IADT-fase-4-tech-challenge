# Pacientes de Demonstração compostos de datasets públicos distintos, com fusão tardia por regras

Não existe dataset público com o mesmo paciente tendo vídeo, áudio de consulta, sinais vitais e prescrições. Por isso, cada Paciente de Demonstração é um paciente fictício que liga um registro do MIMIC-III Waveform (sinais vitais), um paciente do MIMIC-IV Demo (prescrições), uma Consulta do PriMock57 e um clipe de vídeo de licença livre. Os offsets de tempo são escolhidos para alinhar tudo numa mesma internação. A Fusão é tardia: cada Analisador roda de forma independente e gera Alertas, e regras explícitas sobre Alertas do mesmo paciente, numa mesma janela de tempo, geram Alertas Compostos. O relatório declara que esses pacientes são compostos montados para a demonstração.

## Considered Options

- **Fusão precoce (um único modelo sobre features de todas as modalidades):** rejeitada, porque exige o mesmo paciente em todas as modalidades, e esse dado não existe publicamente.
- **Sem fusão (quatro Analisadores lado a lado + Central de Alertas única):** rejeitada, porque não atende à palavra "fusão" do objetivo do PDF, e o sistema pareceria quatro projetos independentes.
- **Dados 100% sintéticos para um paciente coerente:** rejeitada, porque perde o "dado real" que sustenta as métricas de cada Analisador.

## Consequences

- As coincidências entre modalidades num Paciente de Demonstração são **encenadas**, pois os offsets foram escolhidos. Portanto, a Fusão demonstra o mecanismo, e não uma correlação clínica real. Isso precisa constar nas Limitações do relatório.
- As métricas de cada Analisador (AUC, WER, precisão/recall) são medidas nos seus próprios datasets, e nunca no Paciente de Demonstração.
