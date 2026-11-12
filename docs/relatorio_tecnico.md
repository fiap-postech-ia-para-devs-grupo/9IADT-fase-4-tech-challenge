# Relatório Técnico — Tech Challenge Fase 4

> Esqueleto criado no #5. Cada dono escreve a sua seção até o feature freeze (04/11/2026); Renato revisa e costura.
> Vocabulário oficial em [CONTEXT.md](../CONTEXT.md). Estrutura definida em [ESTRATEGIA.md](../ESTRATEGIA.md), seção 13.

| Seção | Dono | Issue |
| --- | --- | --- |
| 1. Contexto | Renato | #17 |
| 2. Fluxo Multimodal | Antonio | #11 |
| 3.1 Vídeo | Marcelo | #6, #13 |
| 3.2 Áudio | V. Geizler | #9, #10, #14 |
| 3.3 Sinais Vitais | Antonio | #7 |
| 3.4 Prescrições | V. Blasque | #8 |
| 3.5 Monitoramento do Leito (câmera e microfone) | Marcelo | #6, #13 |
| 4. Fusão e Pacientes de Demonstração | V. Blasque | #12 |
| 5. Resultados e Exemplos de Anomalias Detectadas | cada dono, nas suas seções; Renato consolida | #17 |
| 6. Integração com Azure | V. Geizler | #14, #21 |
| 7. Limitações | Renato, com insumo de cada dono | #17 |
| 8. Conclusão e Trabalhos Futuros | Renato | #17 |

## 1. Contexto (continuidade com o assistente da Fase 3)

_Dono: Renato._ <!-- TODO -->

## 2. Fluxo Multimodal

_Dono: Antonio._

Cada modalidade tem o seu Analisador, que lê uma fonte pública e emite Alertas no mesmo Contrato do Alerta (`src/core/alerts.py`). Os Alertas vão para um único SQLite (`src/core/db.py`). A Fusão lê esse banco, cruza Alertas de modalidades diferentes do mesmo Paciente de Demonstração numa janela de 24 horas e grava os Alertas Compostos. A Central de Alertas é a única tela em que a equipe vê e reconhece os Alertas.

```mermaid
flowchart LR
    subgraph Fontes ligadas na Admissão
        S[Monitor multiparamétrico: MIMIC-III numerics]
        P[Prescrição eletrônica: MIMIC-IV Demo]
        A[Consulta gravada: PriMock57 / Coswara]
        V[Câmera do leito / sessão: Wikimedia Commons, UR Fall]
        M[Microfone do leito: AudioSet via AST, gravações]
    end
    subgraph Analisadores
        AS[vitals: regras, z-score, IF + piora do NEWS2]
        AP[prescriptions: regras]
        AA[audio: Azure STT + TA for Health + acústica]
        AV[video: YOLOv8-pose + YOLOv8 objetos]
        AM[audio do leito: AST + Whisper]
    end
    DB[(alertas.db)]
    F[Fusão: src/core/fusion.py]
    C[Central de Alertas, Mapa de Leitos, Prontuário]
    L[Luz de Chamada + Acionamento WhatsApp + Escalonamento]
    S --> AS
    P --> AP
    A --> AA
    V --> AV
    M --> AM
    AS --> DB
    AP --> DB
    AA --> DB
    AV --> DB
    AM --> DB
    DB --> F
    F -->|Alerta Composto| DB
    DB --> C
    DB --> L
    L --> C
```

**Monitoramento contínuo.** Os sinais vitais não são analisados em lote. O `src/analyzers/vitals/stream.py` reproduz o registro minuto a minuto (1 minuto de dado por segundo de parede, por padrão) e grava cada Alerta no banco no instante em que a Anomalia aparece pela primeira vez. Isso é o que o Mapa de Leitos e o Prontuário mostram ao vivo, conforme o relógio do plantão anda. O replay sem pausa (`scripts/replay_vitals.py`) grava o mesmo conjunto de Alertas, sem esperar.

**Identidade dos Alertas.** Os ids são determinísticos (UUID v5 a partir do conteúdo da Anomalia), então rodar de novo o mesmo registro não duplica Alertas. Isso faz do banco um resultado reprodutível, e não um acúmulo de execuções.

**Plantão simulado.** Um relógio único (barra lateral) toca o turno de todos os leitos em velocidade acelerada; a barra superior mostra o plantão de hoje (início → fim) e a hora atual. Cada tela mostra o hospital nesse instante, e os Alertas chegam à Central quando "acontecem". A câmera e o microfone do leito (seção 3.5) rodam em tempo real e emitem Alertas no horário do relógio.

**Limite da Fusão.** Os Pacientes de Demonstração são compostos (ver seção 4). Os Alertas de origens diferentes vêm de registros de bases públicas distintas, então a Fusão mostra a regra e a janela, não uma correlação clínica real.

## 3. Modelos por Tipo de Dado

### 3.1 Vídeo — YOLOv8-pose, Desvio de Execução, Queda

_Dono: Marcelo._

**Cenários.** O Analisador de Vídeo (`src/analyzers/video/`) cobre os dois cenários da ESTRATEGIA.md, seção 4. Na **Sessão de Fisioterapia**, o paciente executa um exercício com amplitude esperada conhecida, e o Analisador aponta o **Desvio de Execução**. No **quarto**, ele aponta a **Queda**: o paciente passa de em pé para o chão.

**Pipeline.**

1. `pose.py`: **YOLOv8n-pose** (Ultralytics, keypoints COCO-17) em CPU, 1 a cada 3 frames (~10 por segundo num clipe de 30 fps). Usa a pessoa de maior confiança no frame.
2. `angles.py`: ângulos de joelho, quadril, ombro e cotovelo nos dois lados, mais a inclinação do tronco em relação à vertical. Keypoint com confiança abaixo de 0,5 vira lacuna.
3. `exercise_rules.py`: segmenta as repetições pelo joelho mais flexionado (abaixo de 140° por pelo menos 0,4 s, tolerando 0,35 s de oclusão) e aplica as regras do exercício.
4. `fall.py`: regras sobre a caixa da pessoa (detalhadas abaixo).
5. `report.py`: o **Relatório de Sessão** em Markdown, com as repetições, os Desvios, os frames de Evidência com o esqueleto anotado e o gráfico dos ângulos ao longo do tempo.
6. `lab.py`: `analyze_video(arquivo, exercicio=..., vista=...)` amarra tudo, gera os Alertas no Contrato e é a mesma função usada pela Tela 3 e pelo lote (`scripts/processar_video.py`).

**Regras de Desvio de Execução.** Os limiares vêm de referências de reabilitação e estão documentados no código. Não foram calibrados nos clipes.

| Regra | Exercício | Limiar | Vista |
| --- | --- | --- | --- |
| Amplitude insuficiente | agachamento | joelho no fundo da repetição acima de 100° (coxa longe da horizontal) | qualquer |
| Compensação de tronco | agachamento | tronco inclinado mais de 50° durante a repetição | qualquer |
| Assimetria | agachamento | joelhos diferem mais de 20° no fundo | frontal |
| Valgo dinâmico | agachamento unipodal | joelho de apoio desvia mais de 10° para dentro da linha quadril–tornozelo (FPPA), por pelo menos 0,3 s | frontal |

Valgo e assimetria só são avaliados com câmera de frente ou de costas, porque de lado uma perna esconde a outra. No agachamento unipodal visto de frente, a flexão do joelho quase não aparece na imagem. Por isso, o valgo é contado por episódio, e não por repetição.

**Regra de Queda.** É uma Queda quando três condições se cumprem:

- a caixa da pessoa passa de vertical (largura/altura < 0,6) para deitada (≥ 1,0);
- a altura da caixa cai para no máximo 50% da altura em pé;
- isso acontece em até 2 s e se mantém por 1 s (mediana da janela).

Cada condição descarta um falso positivo visto nos clipes. A altura relativa descarta o exercício com a perna estendida, em que a caixa fica larga mas a pessoa continua alta. A transição curta descarta deitar-se devagar. A permanência descarta um frame isolado com a caixa errada.

**Dados.** Os clipes vêm de duas fontes públicas, listados com licença em `data/video/SOURCES.md` e com gabarito em `data/video/clipes.json`. Baixe com `scripts/download_data.py video`.

- **Fisioterapia:** quatro clipes do Wikimedia Commons. Dois são de agachamento, um de perfil e um de costas. Os outros dois são de agachamento unipodal, publicados por Ageberg et al. (2010) num estudo sobre a observação do valgo. Nesses dois, o gabarito é o rótulo dos próprios autores: "joelho sobre o pé" e "joelho medial ao pé".
- **Quarto:** oito sequências do **UR Fall Detection Dataset** (Kwolek & Kepski, 2014), cinco rotuladas como queda e três como atividades de vida diária sem queda. O vídeo do dataset junta profundidade e RGB lado a lado, e o Analisador usa só a metade RGB.

Não havia dataset público de fisioterapia rotulado com desvios (ESTRATEGIA.md, seção 4). Por isso, nos dois clipes de agachamento o gabarito é a inspeção dos frames.

**Resultados** (`results/video/avaliacao.csv`, Relatórios de Sessão em `results/video/<clipe>/`):

| Clipe | Cenário | Esperado | Detectado | CPU (s) / vídeo (s) |
| --- | --- | --- | --- | --- |
| `agachamento_lateral` | 6 agachamentos, perfil | 0 Desvios | 0 Desvios (6 repetições) | 17,8 / 28,2 |
| `agachamento_costas` | 2 agachamentos, costas | 0 Desvios | 0 Desvios (2 repetições) | 4,3 / 7,1 |
| `unipodal_alinhado` | unipodal, joelho sobre o pé | 0 Desvios | 0 Desvios | 8,2 / 16,7 |
| `unipodal_valgo` | unipodal, joelho medial ao pé | 5 valgos | 5 valgos (FPPA de 15° a 18°) | 7,9 / 16,2 |
| `queda_01` a `queda_05` | queda | 1 Queda cada | 1 Queda cada | 2,0–4,4 / 3,2–7,2 |
| `sem_queda_01` a `sem_queda_03` | atividade sem queda | 0 Quedas | 0 Quedas | 3,2–3,8 / 5,0–6,0 |

Todos os 12 clipes batem com o gabarito. Na Queda, são 5 verdadeiros positivos, 0 falsos positivos (nas 3 atividades sem queda e nos 4 clipes de exercício) e 0 falsos negativos. No valgo, são 5 de 5 no clipe rotulado com valgo e 0 no alinhado, com FPPA de no máximo 7° neste último. Em CPU, o processamento leva de 0,5 a 0,65 s por segundo de vídeo, então daria para analisar ao vivo.

**Leitura dos números e limitações.**

- A amostra é pequena: 12 clipes curtos. Os números mostram que as regras separam os casos rotulados, não estimam desempenho em produção.
- Os limiares de Queda foram escolhidos olhando estes mesmos clipes. Não há conjunto de teste separado.
- O valgo está perto do limiar. Ao processar 1 a cada 2 frames, em vez de 1 a cada 3, aparece um sexto episódio de 11° na transferência de peso inicial, antes do primeiro agachamento.
- A amplitude insuficiente e a compensação de tronco estão cobertas por testes com poses sintéticas, mas nenhum clipe público tinha esses erros com rótulo. Os dois agachamentos reais são bem executados e servem de teste de falso positivo.
- Repetições rasas demais, com o joelho sempre acima de 140°, nem chegam a ser segmentadas.
- A Queda depende da caixa da pessoa: a câmera precisa ver o paciente inteiro. Num quarto real, móveis e a cama ocluem, e cair na direção da câmera encurta menos a caixa. O UR Fall foi gravado numa sala de escritório, não num quarto de hospital.
- O vídeo entra no Paciente de Demonstração do mesmo jeito composto das outras modalidades (seção 4).

**Como rodar.**

```bash
uv run python scripts/download_data.py video
uv run python -m scripts.processar_video        # Relatórios em results/video/, Alertas no SQLite
uv run streamlit run app.py                     # Laboratório de Análise > Vídeo (upload)
```

### 3.2 Áudio — Azure STT + TA for Health + classificador acústico

_Dono: V. Geizler._

**Dados.** Canal do paciente de 18 Consultas do PriMock57 (4–5 por dia dos 5 dias de gravação, `data/audio/primock57_selecionadas.txt`) — áudio já em WAV 16 kHz mono e a transcrição manual (`.TextGrid`) usada como referência do WER. Subset do Coswara (`data/audio/coswara_pastas_selecionadas.txt`): 17 das ~43 pastas do repositório, escolhidas por densidade de rótulo (pessoas com `bd`/`ftg` por MB baixado), já que o dataset não tem um subset pequeno pronto e os rótulos de interesse estão espalhados de forma desigual entre as pastas. Rationale completo de cada escolha nos próprios arquivos de seleção.

**Camada semântica** (`src/analyzers/audio/semantic.py`, ESTRATEGIA.md seção 5):

1. `converter_wav`: WAV 16 kHz mono (`pydub`/`ffmpeg`), idempotente.
2. `transcrever`: Azure Speech to Text (`en-US`), reconhecimento contínuo com timestamp por trecho, **cacheado em JSON** (`src/analyzers/audio/cache.py`) — reprocessar a mesma Consulta lê o cache em vez de gastar cota.
3. `calcular_wer`: WER (`jiwer`) contra o `.TextGrid` manual, normalizando as duas transcrições (minúsculas, sem pontuação, sem as tags `<UNSURE>`/`<UNIN/>` do PriMock57).
4. `extrair_entidades_saude`: Text Analytics for Health, em lotes de até 25 trechos, cacheado.
5. `termos_criticos`: entidades `SymptomOrSign`/`Diagnosis` × lista curada (`src/analyzers/audio/termos.py`) → Alerta `termo_critico`, só para sintomas afirmados sobre o próprio paciente, com o trecho e o `inicio` (segundos desde o começo da Consulta) como Evidência; `categoria` (`respiratorio`/`dor`) só é preenchida nos termos que a Fusão usa (ESTRATEGIA.md, seção 8).
6. `analisar_sentimento`: Sentiment por trecho do paciente, cacheado.

**Camada acústica** (`src/analyzers/audio/acoustic.py`, ESTRATEGIA.md seção 6): features de `counting-normal.wav` e `vowel-a.wav` — MFCC (média/desvio de 13 coeficientes), F0, jitter, shimmer, HNR (`parselmouth`), proporção de pausas e taxa de fala (proxy por transições silêncio→fala sobre a energia RMS, `librosa`). As duas gravações entram como amostras independentes do mesmo rótulo (não concatenadas por pessoa): é o que permite aplicar o mesmo extrator depois sobre um áudio de fala livre, na inferência.

**Treino e avaliação** (`scripts/treinar_acustico.py`). Rótulo por pessoa a partir de `combined_data.csv`: 1 = `bd` (Breathing Difficulties) ou `ftg` (Fatigue), 0 = `healthy` sem nenhum dos dois. Das 1.163 pessoas rotuladas com áudio extraído, balanceei até 300 por classe (300 sintomáticas / 245 saudáveis — menos saudáveis disponíveis nas pastas escolhidas) → 1.067 gravações. Split 75/25 **por pessoa** (as duas gravações da mesma pessoa nunca ficam uma no treino e outra no teste, para não vazar informação do indivíduo entre os dois).

| Modelo | AUC | F1 | Limiar | VN | FP | FN | VP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0,705 | 0,715 | 0,535 | 64 | 55 | 36 | 114 |
| Random Forest | 0,721 | 0,719 | 0,573 | 80 | 39 | 44 | 106 |

Resultados em `results/audio/acoustic_metricas.csv` e `results/audio/acoustic_resumo.json`. O **limiar** de cada modelo é a estatística J de Youden (maximiza sensibilidade + especificidade − 1), calculada com previsões fora da dobra no treino (validação cruzada em 5 dobras, por pessoa), nunca no teste: escolhê-lo no teste inflaria o F1 e a matriz de confusão. O J equilibra os dois erros, já que não há custo clínico definido para favorecer um lado. O **Random Forest** (maior AUC no teste, o que torna essa escolha levemente otimista) é o modelo persistido (`data/audio/modelo_acustico.joblib`) e usado na inferência.

**Inferência no PriMock57** (mesmo script): rodar o classificador sobre a fala do paciente das 18 Consultas, sem precisar do Azure. Resultado em `results/audio/alteracao_vocal_primock57.csv` — probabilidade e se passou do limiar, por Consulta; quando passa, `src/analyzers/audio/acoustic.py::alerta_alteracao_vocal` grava um Alerta `alteracao_vocal`.

**Função do Laboratório** (`src/analyzers/audio/lab.py::analyze_audio`, #14): roda as duas camadas sobre um áudio qualquer e devolve transcrição, Termos Críticos, Sentimento, métricas acústicas e probabilidade de Alteração Vocal, com os Alertas já no Contrato — chamada pela Tela 3 (`src/ui/laboratorio.py`).

**Testes** (`uv run pytest tests/test_audio_*.py tests/test_laboratorio.py`): Azure **mockado** (nenhum teste chama a rede) — `_reconhecer_continuo`, `_extrair_entidades_azure` e `_analisar_sentimento_azure` são substituídos por funções falsas, e o cache é testado separadamente. `converter_wav` e a extração de features acústicas usam `ffmpeg`/`librosa`/`parselmouth` de verdade sobre áudio sintético gerado no teste (tom puro, silêncio), não dados reais. O parser de `.TextGrid`, o cálculo de WER, a busca de Termos Críticos e a geração de Alertas (determinismo, formato do Contrato) são testados com dados fixos.

**Leitura dos números e limitações** (ver também seção 7):

- **Desempenho moderado.** AUC ~0,7 era esperado: a ESTRATEGIA.md já previa "desempenho fraco (AUC ~0,6)" como risco aceito, e o valor está no pipeline reprodutível, não no número. O F1 precisa ser lido com cuidado: com 150 gravações positivas e 119 negativas no teste, marcar tudo como positivo já daria F1 0,716, praticamente igual ao dos dois modelos. Os falsos negativos (36–44) importariam clinicamente: o limiar de Youden não prioriza recall.
- **Treino e inferência vêm de tipos de fala diferentes.** O classificador aprende em `counting-normal`/`vowel-a` (fala controlada, curta) e infere sobre consulta espontânea (PriMock57); é a limitação explícita da ESTRATEGIA.md seção 5 ("as camadas são avaliadas em datasets diferentes"). Na prática, isso aparece como viés: 17 das 18 Consultas passam do limiar de Alteração Vocal (seção 5), evidência de desalinhamento de domínio, não de que quase todo paciente tenha o padrão.
- **Subset do Coswara reduzido por tempo/banda desta sessão.** 545 pessoas, não as 300–600 **por classe** da ESTRATEGIA.md (ficou 300 sintomáticas / 245 saudáveis). `data/audio/coswara_pastas_selecionadas.txt` documenta como escalar para mais pastas.
- **Falsos alertas de Termo Crítico.** Numa revisão manual dos 49 Alertas, sem medida formal de precisão, 6 são duvidosos ou errados (seção 5). Falta rótulo de referência para calcular precisão e recall da camada semântica.

**Como rodar.**

```bash
uv run python scripts/download_data.py primock57 coswara
uv run python -m scripts.treinar_acustico             # métricas + modelo + inferência no PriMock57
uv run python -m scripts.processar_audio_semantico    # requer .env (docs/azure-setup.md)
uv run pytest tests/test_audio_*.py tests/test_laboratorio.py
```

### 3.3 Sinais Vitais — regras × z-score × Isolation Forest

_Dono: Antonio._

**Dado.** 10 registros *numerics* do MIMIC-III Waveform Database (`mimic3wdb/1.0`, acesso aberto), lidos com `wfdb` e reamostrados para 1 minuto (`src/analyzers/vitals/load.py`). Cada registro tem entre 100 e 330 minutos. Foram escolhidos pelo critério: até 50% de lacunas em FC, FR, SpO2 e PAS, e pelo menos 3 janelas limpas de 40 minutos sem lacuna. Lacunas de até 5 minutos são interpoladas; as maiores ficam como ausência de dado. Os registros são curtos e irregulares, o que limita a amostra (ver seção 7).

**Técnicas** (`src/analyzers/vitals/detectors.py`, mesma interface `detect(janela) -> Alerta[]`):

- **Regras clínicas:** SpO2 < 90 (crítica), FC > 120 ou < 45, PAS > 180 ou < 90, FR > 25.
- **z-score móvel:** |x − média| > 3σ sobre os 60 minutos anteriores, com desvio mínimo de 5 bpm (FC), 3 rpm (FR), 2 pontos (SpO2) e 10 mmHg (PAS). O desvio mínimo evita que séries muito estáveis gerem falsos positivos.
- **Isolation Forest:** sobre médias móveis de 5 minutos das quatro séries, 200 árvores, contaminação de 1%.

**Avaliação** (`scripts/avaliar_vitals.py`). Em cada registro, injetamos uma Anomalia de cada tipo (pico de FC, queda gradual de SpO2, hipotensão súbita) com 20 minutos de duração, fora de trechos já marcados pelas regras e pelo z-score. Foram 23 injeções no total, pois nem todos os tipos couberam em todos os registros. Um Alerta cobre uma injeção quando a série bate e os intervalos se cruzam, com tolerância de 5 minutos.

| Técnica | Recall | Precisão | Alertas | Atraso mediano |
| --- | --- | --- | --- | --- |
| Regras clínicas | 1,00 (23/23) | 0,41 | 64 | 0 min |
| z-score móvel | 0,70 (16/23) | 0,19 | 90 | 0 min |
| Isolation Forest | 0,48 (11/23) | 0,61 | 18 | 18 min |

Resultados em `results/vitals/avaliacao.csv` (agregado) e `results/vitals/avaliacao_por_registro.csv`.

**Leitura dos números.** As regras acertam todas as injeções porque os eventos foram desenhados para cruzar exatamente os limites clínicos, então esse recall é otimista. A precisão baixa das regras não é só erro: os Alertas "falsos" contam Anomalias reais que já estavam nos registros (ex.: SpO2 de 76% no registro 3000100n). O z-score detecta mais tipos de desvio, mas gera muitos alertas sobre variações pequenas. O Isolation Forest é o mais preciso, com menos alertas, mas detecta menos e com atraso de cerca de 18 minutos, porque só marca um padrão conjunto depois que a média móvel se afasta da história.

No replay (`scripts/replay_vitals.py`), os 10 registros geraram 124 Alertas `sinal_vital` gravados no SQLite.

### 3.4 Prescrições — regras sobre MIMIC-IV Demo

_Dono: V. Blasque._

**Dado.** MIMIC-IV Clinical Database Demo 2.2 (ODbL, acesso aberto): `hosp/prescriptions` (18.087 linhas, 100 pacientes, 250 internações com prescrição) e `hosp/emar`, mais `admissions` para a alta de cada internação. Só entram prescrições de terapia (`drug_type = MAIN`), sem as soluções de manutenção de acesso (_flush_/_lock_); prescrição sem `stoptime` termina na alta. `scripts/download_data.py mimic4demo` baixa tudo; os dados brutos não vão para o git.

**Técnica.** Regras determinísticas, sem aprendizado: o MIMIC não traz rótulo de "prescrição atípica", então não há precisão/recall a medir. O valor está em regras auditáveis, cada Alerta citando a regra e os dados que o dispararam (`evidencia`). Isolation Forest ficou como trabalho futuro (ESTRATEGIA.md, seção 7).

| Regra | Dispara quando | Severidade |
| --- | --- | --- |
| `salto_de_dose` | mesma droga, via e unidade, dose nova > 1,5× a anterior com início em < 24h | alta se a droga é de alta vigilância, senão média |
| `alta_vigilancia` | primeira prescrição de cada classe de alta vigilância (anticoagulante, insulina, opioide, sedativo) na internação | alta para opioide e anticoagulante, média nas demais |
| `trocas_excessivas` | mais de 4 trocas de princípio ativo dentro da mesma classe em 48h (um Alerta por episódio) | baixa |
| `duplicidade` | dois princípios ativos diferentes da mesma classe com vigências sobrepostas (um Alerta por par e internação) | alta se envolve alta vigilância, senão média |

**Listas versionadas** (`src/analyzers/prescriptions/listas.py`, `VERSAO = "1.0.0"`; toda Evidência registra a versão): drogas de alta vigilância, 10 classes terapêuticas para duplicidade/trocas, lista de reposição de eletrólitos e lista de rotina. A unidade de comparação é o princípio ativo, então metoprolol _tartrate_ e _succinate_ não são duplicidade.

**Decisões de projeto** (todas declaradas aqui porque mudam a contagem):

- Reposição de eletrólitos e glicose (Mg, Ca, K, fosfato, NaCl, bicarbonato, dextrose) fica fora de `salto_de_dose`: é titulada por exame, então a dose oscila por rotina. Sem essa exclusão, a regra gerava 674 Alertas (93% dos pacientes), dominados por magnésio, cálcio e potássio; com ela, 444.
- Heparina + varfarina (ponte de anticoagulação), insulina basal + rápida e AAS + clopidogrel são esquemas padrão e não entram em `duplicidade`.
- "Suspensão" não entra em `trocas_excessivas`: o MIMIC só registra o `stoptime` prescrito, não o momento real da suspensão. Conta-se a troca de princípio ativo dentro da classe.
- O eMAR confirma `alta_vigilancia`: a Evidência traz `primeira_administracao`, a primeira dose de fato administrada.
- Os ids dos Alertas são determinísticos (UUIDv5), então reprocessar não duplica nada no SQLite.

**Resultado nos 100 pacientes** (`results/prescriptions/contagem_por_regra.csv`, listas v1.0.0):

| Regra | Alertas | Pacientes com ao menos 1 |
| --- | --- | --- |
| `salto_de_dose` | 444 | 81 |
| `alta_vigilancia` | 652 | 99 |
| `trocas_excessivas` | 16 | 11 |
| `duplicidade` | 243 | 69 |
| **Total** | **1.355** | |

**Linhas do tempo** (`results/prescriptions/linha_do_tempo_*.html`, gráfico interativo com as prescrições monitoradas e os Alertas sobre a droga de cada um): internações com mais regras distintas disparadas.

- Paciente 10014729, internação 28889419: os quatro tipos de Alerta, por exemplo "Duplicidade terapêutica (opioide): Meperidine e Oxycodone-Acetaminophen ativas ao mesmo tempo" e "5 trocas de medicação em 48h".
- Paciente 10032725, internação 25177949: "Aumento de dose > 50% em < 24h: HYDROmorphone (Dilaudid) 2 → 6 mg" e "Duplicidade terapêutica (opioide): HYDROmorphone e Oxycodone SR ativas ao mesmo tempo".

**Leitura dos números e limitações.** O demo é majoritariamente de UTI, então quase todo paciente inicia uma droga de alta vigilância (99%): a regra mostra o mecanismo, não discrimina risco, e numa enfermaria teria taxa bem menor. Os limiares (1,5×, 24h, 4 trocas, 48h) vêm da ESTRATEGIA.md ou foram escolhidos olhando a distribuição do próprio demo; não houve validação clínica. As listas de drogas são curadas à mão para o escopo do trabalho, não um formulário completo. As datas do MIMIC são deslocadas (anos 21xx); o alinhamento com as outras modalidades é feito pelos offsets dos Pacientes de Demonstração (seção 4).

**Como rodar.**

```bash
uv run python scripts/download_data.py mimic4demo
uv run python -m scripts.analisar_prescricoes --gravar   # contagem, linhas do tempo e Alertas no SQLite
uv run pytest tests/test_prescriptions.py
```

### 3.5 Monitoramento do Leito — câmera e microfone (YOLOv8, AST/AudioSet, Whisper)

_Dono: Marcelo._ Decisão em [ADR 0003](adr/0003-monitoramento-do-leito-e-evento-sonoro.md).

**Por quê.** Os Analisadores de vídeo e áudio das seções 3.1 e 3.2 rodam em lote ou sob demanda. Num monitoramento real, a câmera e o microfone ficam ligados ao leito, e o risco é detectado quando acontece. Na Admissão, cada paciente recebe esses dispositivos, e no Prontuário a seção **Câmera e microfone** liga vídeo e/ou áudio. A integração é **simulada**: a webcam e o microfone do computador onde o app roda, ou gravações públicas tocadas em tempo real.

**Câmera** (`src/analyzers/video/vigilancia.py`). Uma thread por leito lê os quadros, estima a pose (YOLOv8n-pose, 1 a cada 2 quadros) e roda o detector de Queda da seção 3.1 numa janela deslizante de 8 s. No frame da Queda, o **YOLOv8n de objetos** detecta as áreas críticas de transferência (cama, cadeira, sofá, vaso sanitário) junto ao paciente; elas entram na Evidência e na descrição ("… junto a cadeira"). Nos 5 clipes de queda do UR Fall, a área crítica apontada foi cadeira (e cadeira + sofá num deles). Cada leito mantém uma Queda aberta até o Reconhecimento.

**Score de Queda (webcam ao vivo).** Com a câmera perto do paciente o corpo inteiro não aparece, e o critério da caixa (seção 3.1) não se aplica. Para a webcam, cada quadro recebe um **Score de Queda de 0 a 100**, mostrado ao vivo no Prontuário com seus componentes:

| Componente | Medida | 0 → 1 | Peso |
| --- | --- | --- | --- |
| Inclinação | ângulo do eixo ombros → nariz com a vertical | 30° → 60° | 40% |
| Mudança brusca | graus ganhos nos últimos 3 s | 0° → 45° | 40% |
| Descida | quanto os ombros desceram em 3 s, em larguras de ombro | 0 → 1 | 20% |

O **ângulo que caracteriza a Queda é 60° ou mais** (90° = deitado de lado; acima de 120° o nariz está abaixo dos ombros, a pessoa se curvou e não conta). O Alerta sai com **score ≥ 70 e o ângulo mantido ≥ 60° por 1,5 s**. Uma segunda regra cobre a webcam na altura da mesa, que não enxerga o chão: ombros descendo na faixa de baixo da imagem e cabeça fora da imagem por 3 s (score 80). Score, ângulo, variação e descida vão para a Evidência. Calibrado com a trilha quadro a quadro de um teste real (`results/video/camera/<paciente>/trilha.csv`): as duas quedas do teste foram detectadas, e nenhum Alerta nos outros ~8 min (sentado, virando de perfil, levantando). É uma calibração de um teste, não uma métrica: o critério da caixa (12/12 nos clipes) continua sendo o validado.

**Processamento parcial.** A webcam roda em 640×480 com pose a 320 px a cada 2 quadros (os intermediários mostram o último esqueleto); o som usa metade dos núcleos da CPU e um único passe do AST por janela, e o Whisper só roda quando há indício de voz.

**Microfone** (`src/analyzers/audio/ambiente.py`, `vigilancia_audio.py`). A cada 2,5 s, os últimos 5 s de áudio passam pelo **AST** (Audio Spectrogram Transformer) treinado no **AudioSet** — o dataset de áudio sugerido no enunciado — e as 527 classes viram eventos clínicos:

| Evento Sonoro | Classes do AudioSet | Severidade |
| --- | --- | --- |
| alarme de equipamento | Beep/bleep, Alarm, Busy signal, Dial tone, Buzzer… | média |
| grito | Screaming, Shout, Yell… | alta (aciona) |
| gemido / choro | Groan, Whimper, Wail/moan, Crying… | média |
| chamado da equipe | fala + Whisper tiny (pt) com "enfermeira", "socorro", "ajuda", "doutor", "ai", "ui", "dói", "dor"… | alta (aciona) |

Nas gravações de teste (`data/audio/sons/`): o alarme do monitor saiu como alarme em 4 de 4 janelas (AudioSet "Busy signal"/"Beep"); o grito curto como grito (0,33) e gemido; o choro como gemido numa das janelas; e a frase sintética "Enfermeira! Socorro! Alguém me ajuda" como chamado, com o Whisper transcrevendo "Enfermeira. Socorro." e "alguém me ajuda por favor". O Whisper tiny erra a grafia de fala cortada ("infirmei"), por isso a janela é de 5 s e as palavras de chamado são radicais. Cada leito mantém um Alerta aberto por tipo de evento.

**Fusão.** Nova regra crítica: **Queda + grito ou chamado** do mesmo paciente ("Queda com o paciente gritando ou pedindo ajuda").

**Limitações.** Não há métrica formal para os sons (só gravações rotuladas pelo nome); em CPU, câmera e microfone juntos rendem ~3 quadros/s, e o carregamento dos modelos de som leva ~2 min na primeira vez.

## 4. Fusão e Pacientes de Demonstração

_Dono: V. Blasque._

**Os pacientes são compostos.** Não existe dataset público com o mesmo paciente em todas as modalidades. Cada Paciente de Demonstração é fictício e liga fontes reais de pessoas diferentes, com os horários escolhidos para caber numa mesma internação ([ADR 0001](adr/0001-paciente-de-demonstracao-composto.md)). As coincidências entre modalidades são **encenadas**: a Fusão demonstra o mecanismo, não uma correlação clínica. As métricas de cada Analisador vêm dos seus próprios datasets, nunca destes pacientes.

**Montagem** (`data/demo_patients.json`, `src/core/demo.py`). A internação dura o que dura o registro de Sinais Vitais. Cada fonte é deslocada para esse relógio: as prescrições por uma âncora (um instante do MIMIC-IV que cai num minuto da internação), a Consulta e o vídeo por um minuto de início. Alertas de prescrição fora da internação ficam de fora. A Consulta é lida do cache versionado do Azure, sem gastar cota.

| Paciente | Sinais Vitais (MIMIC-III) | Prescrições (MIMIC-IV Demo) | Consulta (PriMock57) | Alerta Composto |
| --- | --- | --- | --- | --- |
| `demo-01` | `3002751n` | paciente 10035631, oxicodona no minuto 60 | `day3_consultation04`, início no minuto 136,33 | **crítico**: possível insuficiência respiratória |
| `demo-02` | `3009891n` | paciente 10005348, metoprolol 12,5 → 50 mg no minuto 60 | `day5_consultation01`, início no minuto 150 | alto: possível efeito adverso de medicação |
| `demo-03` | `3007382n` | paciente 10021487, hidromorfona no minuto 90 | `day1_consultation03`, início no minuto 120 | alto: Queda (clipe `queda_02`, minuto 200) possivelmente associada a medicação |

No `demo-01`, o offset da Consulta faz "short of breath", dito aos 40 s, coincidir com a SpO2 de 85% do minuto 137: o Termo Crítico respiratório e o Sinal Vital caem no mesmo minuto e geram o Alerta Composto crítico. Essa Consulta está entre as únicas com Termo Crítico respiratório, que é o que a regra exige ([ADR 0002](adr/0002-alteracao-vocal-fora-da-fusao.md)).

**Fusão.** Tardia e por regras (`src/core/fusion.py`, seção 8 da ESTRATEGIA.md): dois Alertas do mesmo paciente, de modalidades diferentes, numa janela de 24 h. Como a internação simulada dura poucas horas, quase todo par coincide. Por isso, a demonstração mantém um Alerta Composto por regra, o do par mais próximo no tempo.

**Unidade, Admissão e Prontuário.** Os Pacientes de Demonstração entram como admissões já feitas em leitos da unidade (`src/core/cadastro.py`), com identidade fictícia e idade/sexo da fonte do MIMIC-IV. Novas admissões escolhem um leito livre e ligam as fontes (registro de Sinais Vitais, internação do MIMIC-IV, Consulta em cache e câmera); o paciente passa a ser monitorado a partir da hora do plantão. A equipe é cadastrada com função, registro e WhatsApp; o plantonista é escolhido dela e assina Reconhecimentos e a evolução.

**Prontuário (Tela 2).** O replay avança o relógio da internação e cada Alerta surge no minuto em que ficaria visível: os de Sinais Vitais quando o detector os emite, os demais no seu horário e o Composto quando a última origem aparece. Os Alertas surgidos vão para a Central de Alertas, onde a equipe os reconhece; reiniciar o replay não desfaz um Reconhecimento. A linha do tempo mostra prescrições monitoradas, Consulta e vídeo, com os marcadores de Alerta.

**Alertas inteligentes** (`src/core/news2.py`, `inteligencia.py`).

- **Piora do NEWS2 parcial:** o escore (FR, SpO2, PAS e FC; sem temperatura, consciência e O2, então é um piso do NEWS2 real) subir 2+ pontos em até 1 h e chegar a 5+ (alto se 7+) gera um Alerta de Sinal Vital, um por episódio. O monitor de cada leito mostra o escore e a conduta recomendada pelo NEWS2.
- **Episódios:** Alertas repetidos do mesmo paciente, regra e sinal a menos de 30 min viram um cartão só na Central, contra a fadiga de alarme.
- **Escalonamento:** chamado sem Reconhecimento em 5 min do plantão é marcado como escalonado e aciona a equipe médica.

**Do Alerta à equipe médica (fluxo final).**

1. O Analisador grava o Alerta no SQLite (Contrato único). A Fusão grava o Composto quando modalidades se confirmam.
2. Se o Alerta pede chamada — crítico, Queda, piora do NEWS2, grito ou chamado —, a **luz do leito** acende (vermelha, piscando) no Mapa de Leitos e no Prontuário, a **sirene** da unidade acende na barra superior e a faixa de chamada de emergência lista os leitos.
3. A equipe de plantão com WhatsApp cadastrado é **acionada** pela Evolution API (`src/core/notificacao.py`), com o leito, o paciente, o Alerta e o link da Central — **um disparo por leito** até o Reconhecimento. Sem configuração, o acionamento é simulado e registrado.
4. Sem Reconhecimento em 5 min, o chamado **escala** para a equipe médica.
5. Na **Central de Alertas**, o plantonista abre a Evidência (frame anotado, trecho de áudio, janela do sinal), **reconhece** e registra a **conduta**, que vai para a **evolução** do paciente; depois **resolve**. A luz do leito apaga.

**Limitações.**

- Cada paciente tem um clipe de vídeo (`demo-01`: `unipodal_valgo`; `demo-02`: `agachamento_lateral`; `demo-03`: `queda_02`), e os Alertas de vídeo entram na tela e na Fusão depois de `scripts/processar_video.py`. Como nas outras modalidades, a coincidência entre a Queda e o opioide do `demo-03` é encenada pelo minuto de início do clipe.
- Os trechos de prescrição foram escolhidos a dedo para a história de cada paciente, o que torna a contagem de Alertas destes pacientes não representativa; a contagem por regra está na seção 3.4.
- Os registros de Sinais Vitais têm só 4 a 5 horas, então a internação simulada é curta.

**Como rodar.**

```bash
uv run python scripts/download_data.py mimic3wdb mimic4demo
uv run streamlit run app.py                     # barra lateral > ▶ Iniciar; Mapa de Leitos e Prontuário
uv run python -m scripts.gerar_pacientes_demo   # opcional: grava tudo no SQLite sem tocar o replay
```

## 5. Resultados e Exemplos de Anomalias Detectadas

_Dono: cada um nas suas seções; Renato consolida._

### Sinais vitais (Antonio)

Exemplos reais, nos registros do MIMIC-III listados em `data/vitals/selecionados.txt`. Horários são simulados, a partir de 2150-01-01 00:00, como no `load.py`.

- **Dessaturação grave (regra clínica, crítica).** Registro `3000100n`: SpO2 de 79,4% e 76,6% em dois minutos consecutivos, entre 01:22 e 01:23. Alerta: "SpO2 < 90 por 2 min", severidade crítica. É a Anomalia real mais clara dos registros, e a equipe precisaria ser avisada.
- **Hipertensão sustentada (regra clínica, alta).** Registro `3000100n`: PAS de 186 mmHg por quatro minutos, entre 02:11 e 02:14. Alerta: "PAS > 180 por 4 min", severidade alta.
- **Bradicardia isolada (regra clínica, alta).** Registro `3000100n`: FC de 44,9 bpm durante um minuto, às 02:35. Alerta: "FC < 45 por 1 min". Por ser de um minuto, pode ser artefato de medida; a equipe confirma no leito.
- **Desvio em relação à própria história (z-score, média).** Registro `3002751n`: FC 3,6σ acima da média dos 60 minutos anteriores, entre 01:46 e 01:47. O z-score pega variações que não cruzam nenhum limite clínico, mas também gera Alertas sobre oscilações pequenas, por isso a precisão dele é a mais baixa (ver seção 3.3).

Para os números da avaliação com anomalias injetadas (recall, precisão e atraso por técnica), ver a seção 3.3 e `results/vitals/`.

### Vídeo (Marcelo)

Exemplos reais, com o frame de Evidência e o Relatório de Sessão em `results/video/<clipe>/`:

- **Valgo dinâmico (Desvio de Execução, média).** No clipe `unipodal_valgo`, o joelho direito de apoio desvia 18° para dentro aos 8,8 s (limite: 10°). Os 5 mini agachamentos geram um Desvio cada, como no rótulo dos autores. No `demo-01`, esses Desvios entram na internação no minuto 210.
- **Queda (alta).** No clipe `queda_02`, a pessoa vai de em pé ao chão em 0,6 s: a caixa fica 1,8 vez mais larga que alta, com 48% da altura em pé. No `demo-03`, a Queda cai no minuto 200, depois do início de hidromorfona no minuto 90, e gera o Alerta Composto "Queda possivelmente associada a medicação".
- **Sem falso positivo.** Os 6 agachamentos bem executados do `agachamento_lateral`, com o joelho chegando a 50–54°, não geram Desvio nem Queda.

### Áudio (V. Geizler)

Inferência real do classificador acústico (Random Forest, AUC 0,721, limiar 0,573 — seção 3.2) sobre a fala do paciente nas 18 Consultas do PriMock57, sem depender do Azure (`results/audio/alteracao_vocal_primock57.csv`):

- **17 de 18 Consultas** passaram do limiar (probabilidades entre 0,60 e 0,87) — ex.: `day5_consultation02` com 0,867, `day1_consultation01` com 0,667.
- Só `day3_consultation01` (0,493) ficou abaixo.

Esse resultado **não deve ser lido como "quase todo paciente do PriMock57 tem Alteração Vocal"**: é evidência do desalinhamento de domínio já apontado na seção 3.2 — o classificador, treinado em gravações curtas e controladas (contagem, vogal sustentada) do Coswara, generaliza mal para fala espontânea de consulta e tende a marcar positivo. Fica registrado como achado honesto, não como resultado clínico (ver seção 7). Por isso, a Alteração Vocal fica fora da Fusão ([ADR 0002](adr/0002-alteracao-vocal-fora-da-fusao.md)): um Alerta que dispara em quase toda Consulta não confirma nada.

**Camada semântica**, executada de verdade no Azure sobre as mesmas 18 Consultas (`results/audio/wer.csv` e `results/audio/termos_criticos.csv`):

| Medida | Resultado |
| --- | --- |
| WER médio, bruto | 0,218 (0,106 a 0,326 por Consulta) |
| WER médio, sem hesitações | 0,171 (0,092 a 0,292 por Consulta) |
| Termos Críticos | 49, em 15 Consultas: 12 respiratórios, 11 de dor, 26 sem categoria da Fusão |
| Consultas com Termo Crítico respiratório | 7 |

A transcrição manual registra hesitações ("um", "uh") que o Azure quase sempre omite: em `day1_consultation01`, são 52 contra 3. Elas não afetam nenhum Termo Crítico, então o WER sem hesitações é a leitura principal, com o bruto ao lado para transparência. Números escritos em dígitos pelo Azure ("6" contra "six") ainda contam como erro nas duas medidas.

Sintomas negados ou atribuídos a terceiros, conforme marcados pela TA for Health, não geram Alerta ("No fever at all"). A marcação não é perfeita: numa revisão manual dos 49 Alertas, sem rótulo de referência, 6 são duvidosos ou errados, por exemplo "one child was vomiting", "I don't feel like I'm feverish" e "Cold chest pain out of pain", este último um erro de transcrição. Erros de reconhecimento também causam perdas: "shortness of breath" foi transcrito como "assurance of breath" e "sure of breath". Todos os 12 respiratórios revisados têm base na fala do paciente, ainda que "difficult breathing" (`day3_consultation01`) se refira a um episódio passado.

## 6. Integração com Azure (serviços, custos, cotas)

_Dono: V. Geizler._ Setup completo, cotas F0 e Plano B em [azure-setup.md](azure-setup.md) (#4); conta e recursos criados no #21, com o smoke test passando nos três serviços em 05/11/2026.

**Serviços.** **Speech** (Speech to Text, `en-US`, reconhecimento contínuo com timestamp) e **Language** (Text Analytics for Health + Sentiment), ambos tier **F0** (gratuito): 5h/mês de STT e 5.000 registros/mês de Language, compartilhados pelo time inteiro numa única chave.

**Orçamento de cota usado pelo Analisador de Áudio:**

| Uso | Custo estimado |
| --- | --- |
| Batch de 18 Consultas (canal do paciente) | 2,57 h de STT (a cota conta a duração do arquivo, não só a fala); 1.043 trechos: ~1.043 registros de Sentiment e ~2.086 de TA for Health, que precisou rodar duas vezes para guardar negação e sujeito |
| Smoke test (`scripts/azure_smoke_test.py`) | ~10 s de STT + 2 registros de Language |
| Upload no Laboratório (por análise) | 1 arquivo de STT + 1 registro de Language por trecho transcrito |

Isso consome cerca de metade das 5 h de STT e cerca de 3.100 dos 5.000 registros mensais. Sobra margem para a demo e a gravação do vídeo, mas sem espaço para repetir o batch sem o cache (ESTRATEGIA.md, seção 15). O F0 aceita uma transcrição por vez: duas pessoas não transcrevem ao mesmo tempo com a mesma chave.

**Protegendo a cota.** Toda resposta do Azure (STT, TA for Health, Sentiment) é cacheada em JSON por `src/analyzers/audio/cache.py` (`obter_ou_computar`), em `data/audio/cache_primock57/` (batch) ou `data/audio/cache_laboratorio/` (upload): reprocessar o mesmo áudio lê o cache e não chama o Azure de novo. Isso é o que permite rodar `uv run pytest` e `scripts/treinar_acustico.py` (camada acústica) sem tocar a cota nenhuma vez — só `scripts/processar_audio_semantico.py` e o upload na Tela 3 chamam o Azure de fato, e só na primeira vez por arquivo. O cache do batch é versionado no git, sem os WAVs, para que o time e os avaliadores reproduzam os resultados sem gastar cota.

**Estado:** executado de verdade contra o Azure nas 18 Consultas; os testes automatizados seguem com Azure mockado. Rodar com o Azure real expôs o que os mocks escondiam, como o limite de 10 documentos por chamada de Sentiment.

## 7. Limitações (datasets distintos, pacientes compostos, dados em inglês)

_Dono: Renato, com insumo de cada dono._

**Sinais vitais (Antonio):**

- **Amostra pequena.** São 10 registros de 100 a 330 minutos, com até 50% de lacunas, e uma injeção de cada tipo por registro. Os números da avaliação têm pouca base estatística; servem para comparar técnicas, não para afirmar desempenho clínico.
- **Critério de seleção próprio.** Os registros foram escolhidos por lacunas e janelas limpas, critério definido pelo time porque o PDF não especifica nenhum. Isso pode favorecer registros mais "limpos" do que a população real de UTI.
- **Anomalias injetadas são artificiais.** Os eventos injetados cruzam exatamente os limites clínicos, o que favorece as regras. O recall das regras é otimista e não deve ser lido como capacidade de detecção em produção.
- **Anomalias naturais contam como falso positivo.** A precisão baixa das regras inclui Anomalias reais que já estavam nos registros. Sem rótulo clínico, não dá para separar erro de detecção de anomalia verdadeira.
- **Registros de 1 minuto.** A resolução é de um ponto por minuto. Eventos de poucos segundos, como uma arritmia breve, não são capturados.

**Vídeo (Marcelo):** detalhes na seção 3.1.

- **Poucos clipes e limiares vistos nos mesmos clipes.** Os resultados mostram que as regras separam os casos rotulados; não são uma estimativa de desempenho clínico.
- **Gabarito de fisioterapia parcial.** Só o valgo tem rótulo externo, dos autores do estudo. Amplitude insuficiente e compensação de tronco foram validadas apenas com poses sintéticas.
- **Queda depende de ver o corpo inteiro.** Num quarto real, oclusão por móveis e cama e a queda na direção da câmera são os casos difíceis. O UR Fall não é um quarto de hospital.

**Monitoramento do Leito, NEWS2 e acionamento (Marcelo):**

- **Dispositivos simulados.** A câmera e o microfone do leito são a webcam/microfone do computador do app ou gravações; não há integração com câmera IP nem central de áudio hospitalar.
- **Sons sem métrica formal.** O AST/AudioSet e o Whisper foram verificados só com gravações rotuladas pelo nome; o Whisper tiny erra grafia em fala cortada.
- **NEWS2 parcial.** Sem temperatura, nível de consciência e O2 suplementar, o escore é um piso do NEWS2 real; não substitui a avaliação clínica.
- **WhatsApp depende de serviço externo.** O acionamento usa a Evolution API de um integrante; num hospital, o canal oficial seria o sistema de chamada de enfermagem.

**Áudio (V. Geizler):**

- **Termos Críticos sem medida formal de precisão e recall.** Não há rótulo de referência no PriMock57: a revisão manual achou 6 Alertas duvidosos ou errados em 49 (seção 5). A lista curada também limita o recall: variantes como "pain in my chest" não casam.
- **Treino e inferência acústica vêm de tipos de fala diferentes.** O classificador é treinado em `counting-normal`/`vowel-a` do Coswara (fala controlada, curta) e infere sobre a fala espontânea de uma Consulta (PriMock57) ou de um upload qualquer; é uma limitação de desenho, não um bug — já prevista na ESTRATEGIA.md, seção 5. Na inferência real sobre as 18 Consultas do PriMock57, 17 passaram do limiar (seção 5): é desalinhamento de domínio, não um sinal clínico real.
- **AUC moderado (~0,7) e F1 próximo do trivial.** O AUC era esperado desde a ESTRATEGIA.md (seção 15, risco aceito); o F1 de 0,719 fica perto dos 0,716 de marcar tudo como positivo. O valor do classificador está em ter um pipeline reprodutível com métrica real, não em alta acurácia; os falsos negativos do limiar de Youden importariam clinicamente numa aplicação real.
- **Subset do Coswara menor que o alvo.** 545 pessoas (300 sintomáticas / 245 saudáveis), não as 300–600 **por classe** da ESTRATEGIA.md: o dataset distribui os rótulos de interesse de forma muito desigual entre as pastas (ver rationale em `data/audio/coswara_pastas_selecionadas.txt`), e baixar mais pastas custaria bandwidth/tempo além do orçado nesta frente.
- **Dados em inglês, de dois contextos diferentes.** PriMock57 é consulta simulada (ator, não paciente real); Coswara é rastreamento de COVID-19, não triagem geral de dificuldade respiratória/fadiga — a Alteração Vocal detectada não tem validação clínica própria.

## 8. Conclusão e Trabalhos Futuros

_Dono: Renato._ AWS Textract/Comprehend, OpenAI, TORGO, notificação externa. <!-- TODO -->

## Apêndice A — Cobertura do enunciado

Rastreabilidade de cada item do `docs/IADT - Fase 4 - Tech challenge.pdf` até o que foi entregue.

| Enunciado | Onde está | Evidência |
| --- | --- | --- |
| Vídeo: processar sessões de fisioterapia | `src/analyzers/video/`, Laboratório (clipe, upload, webcam) | seção 3.1, `results/video/` |
| Vídeo: OpenPose/postura | YOLOv8-pose (keypoints COCO-17, ângulos articulares) | seção 3.1 |
| Vídeo: YOLOv8 para objetos e áreas críticas | YOLOv8n de objetos no frame da Queda (cama, cadeira, sofá, vaso) | seção 3.5, `results/video/queda_0*/` |
| Vídeo: relatórios automáticos de desvios | Relatório de Sessão (`report.py`) | `results/video/<clipe>/relatorio.md` |
| Áudio: processar consultas | PriMock57, upload/microfone no Laboratório | seção 3.2 |
| Áudio: alterações vocais (cansaço, dificuldade respiratória) | classificador acústico (Coswara) | seção 3.2 |
| Áudio: Azure Speech to Text | `semantic.py` | seção 3.2, seção 6, WER |
| Áudio: termos críticos e sentimentos com Azure Text Analytics | TA for Health + Sentiment | seção 3.2 |
| Anomalias em sinais vitais | regras, z-score, Isolation Forest, piora do NEWS2 | seções 3.3 e 4 |
| Anomalias em prescrições | regras de Mudança Terapêutica Atípica | seção 3.4 |
| Padrões de movimentação durante a internação | Queda pela câmera do leito (contínua) | seções 3.1 e 3.5 |
| Alertas automáticos à equipe médica | Central, luz de chamada, WhatsApp, escalonamento | seção 4 (fluxo final) |
| Fusão de dados (texto, áudio, vídeo) | Fusão tardia por regras (5 regras, entre elas Queda + grito/chamado) | seção 4, ADRs 0001–0003 |
| Tempo real / monitoramento preventivo | relógio do plantão, câmera e microfone do leito | seções 2, 3.5 |
| Serviços em nuvem (Azure) | Speech + Language (F0) | seção 6 |
| Sugestão de datasets: PhysioNet, AudioSet | MIMIC-III/IV (PhysioNet), AST treinado no AudioSet | seções 3.3, 3.4, 3.5 |
| Relatório: fluxo, modelos, resultados, exemplos | este documento | seções 2, 3, 5 |
| Vídeo de até 15 min | roteiro na ESTRATEGIA.md, seção 13 | entrega final (#18) |
