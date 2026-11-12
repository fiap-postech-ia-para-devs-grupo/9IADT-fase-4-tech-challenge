# Sons do leito (monitoramento de áudio)

Gravações tocadas em tempo real como se fossem o microfone do quarto, para demonstrar o
monitoramento de áudio do leito (`src/analyzers/audio/vigilancia_audio.py`). Os de terceiros não
vão para o git: `scripts/download_data.py sons` os reproduz.

| Arquivo | Evento | Fonte | Licença |
| --- | --- | --- | --- |
| `alarme_monitor.mp3` | alarme de equipamento | [Heart Monitor Beep (Wikimedia Commons / freesound)](https://commons.wikimedia.org/wiki/File:Heart_Monitor_Beep--freesound.org.mp3) | CC0 |
| `grito.ogg` | grito | [Wilhelm Scream (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Wilhelm_Scream.ogg) | CC0 |
| `gemido_choro.ogg` | gemido / choro | [Male crying and weeping (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Male_crying_and_weeping.ogg) | CC BY 3.0 |
| `chamado_enfermeira.wav` | chamado da equipe | gerado pelo time com a voz sintética pt-BR do Windows (“Enfermeira! Socorro! Alguém me ajuda, por favor!”) | do projeto (versionado) |

Classificação: AST treinado no AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`) e Whisper
tiny (`openai/whisper-tiny`) para as palavras de chamado; os pesos são baixados pelo Hugging Face
na primeira execução.
