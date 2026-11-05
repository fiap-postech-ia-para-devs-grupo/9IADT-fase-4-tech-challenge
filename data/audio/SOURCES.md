# Fontes dos dados de áudio versionados

## PriMock57

Consultas simuladas: clínicos e funcionários da Babylon atuando como pacientes a partir de fichas de caso. Não são pacientes reais.

- **Fonte:** <https://github.com/babylonhealth/primock57>
- **Autoria:** Papadopoulos Korfiatis, Moramarco, Sarac e Savkov. *PriMock57: A Dataset Of Primary Care Mock Consultations*, ACL 2022. <https://arxiv.org/abs/2204.00333>
- **Licença:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Material derivado neste repositório, com estas alterações:

- `teste_upload/teste_falta_de_ar.wav`: trecho de 35 s a 95 s da fala do paciente em `day3_consultation04`, convertido para WAV 16 kHz mono.
- `cache_primock57/*.json`: respostas do Azure Speech e do Azure Language para a fala do paciente de 18 Consultas, também convertida para WAV 16 kHz mono antes do envio.
