# Tech Challenge Fase 4 — Estratégia de Entrega

> Projeto: Monitoramento Multimodal de Pacientes (vídeo + áudio + sinais vitais + prescrições → Alertas)
> Time: 5 integrantes · 28/09/2026 → **17/11/2026** (50 dias) · ~1h/dia por pessoa · **~215h efetivas totais**
> Vocabulário oficial: ver [CONTEXT.md](CONTEXT.md). Usar esses termos no código, no relatório e no vídeo.

---

## 1. Decisões Fechadas (não reabrir)

| Decisão | Escolha | Motivo |
| --- | --- | --- |
| Repositório | Novo, do zero; reaproveita **padrões** da Fase 3 (fila com status, auditoria SQLite, Streamlit), não código | Stack da Fase 3 (Llama/QLoRA/GPU) não tem relação com a Fase 4 |
| Time | Mesmas 5 pessoas; Renato só a partir de 05/11 (férias em outubro) | Renato assume vídeo + entrega final; Vinicius Blasque vai para construção |
| Feature freeze | **04/11/2026** | A partir de 05/11 só integração final, polimento, relatório e vídeo |
| Nuvem | **Somente Azure** (Azure for Students, tier F0): Speech to Text + AI Language (Text Analytics for Health + Sentiment) | Requisito literal do PDF; uma nuvem só = metade do custo de setup. AWS/OpenAI ficam como trabalhos futuros |
| Conceito central | **Alerta** único para todas as modalidades, persistido em SQLite | O PDF termina em "fluxo final do alerta à equipe médica" |
| Canal do alerta | **Central de Alertas** no Streamlit, com Reconhecimento | Sem bot de Telegram/Teams |
| Vídeo — cenário | Sessões de Fisioterapia (Desvio de Execução) + vídeo de quarto (Queda) | Cobre "fisioterapia" e "movimentação durante internação"; cirurgia exigiria dataset rotulado de instrumentos |
| Vídeo — modelo | **YOLOv8-pose** (Ultralytics, keypoints COCO-17) | Cobre "YOLOv8" e análise postural estilo OpenPose com uma lib só, em CPU; OpenPose tem instalação inviável |
| Vídeo — dados | Clipes com licença livre (Pexels/Pixabay) | Não há dataset público de fisioterapia rotulado no escopo |
| Áudio — semântica | **PriMock57** (57 consultas simuladas, ~8,6h, transcrição manual) → Azure STT `en-US` + TA for Health + Sentiment | Único dataset público de consulta médica real em áudio; transcrição manual permite medir WER |
| Áudio — acústica | **Coswara** (2.635 pessoas, rótulos de *breathing difficulty* / *fatigue*) → features `librosa`/`parselmouth` + classificador | Modelo com métrica real (AUC/F1), não limiar inventado |
| Áudio — fora | Gravação própria do time, TORGO (disartria), SPIRA (sem download público) | Dataset pequeno/irrelevante; terceira frente de dados sem retorno |
| Sinais vitais | **MIMIC-III Waveform Database** (numerics: FC, FR, SpO2, PA) — acesso aberto, lido via `wfdb` | Dado real de UTI; MIMIC Clinical exige credenciamento (semanas) |
| Técnica — sinais vitais | Regras clínicas + z-score em janela móvel + Isolation Forest, comparadas com **anomalias injetadas** | Sem rótulo real de anomalia; injeção dá precisão/recall mensuráveis |
| "Tempo real" | **Replay em streaming** (simulador toca a série acelerada, janela a janela) | Monitoramento Contínuo sem infra de streaming |
| Prescrições | **MIMIC-IV Clinical Database Demo** (100 pacientes, acesso aberto, ODbL) + regras | Dado real, custo baixo; Isolation Forest só se sobrar tempo |
| Fusão | **Fusão tardia por regras** sobre Alertas do mesmo **Paciente de Demonstração** | Não existe paciente público com todas as modalidades; fusão precoce é inviável ([ADR 0001](docs/adr/0001-paciente-de-demonstracao-composto.md)) |
| Execução | Sinais vitais ao vivo; prescrições batch; vídeo/áudio batch + upload sob demanda | Demo visual sem estourar cota do Azure |
| Interface | Streamlit, 3 telas | Familiaridade do time (Fases 2 e 3) |
| Relatório e testes | **Cada dono escreve sua seção e seus testes** até 04/11 | Renato recebe um documento para revisar, não para escrever |

---

## 2. Arquitetura da Solução

```text
 FONTES PÚBLICAS                 ANALISADORES (src/analyzers/)            NÚCLEO (src/core/)
 ──────────────                  ──────────────────────────────           ──────────────────
 Clipes Pexels/Pixabay ───────►  video/   YOLOv8-pose                ┐
   (fisioterapia, quarto)          • ângulos articulares → Desvio    │
                                   • pose em pé→chão → Queda         │
                                   • Relatório de Sessão (HTML/MD)   │
                                                                     │
 PriMock57 (consultas) ───────►  audio/   camada semântica (Azure)   │
                                   • Speech to Text + WER            │
                                   • TA for Health → Termos Críticos │     ┌──────────────┐
                                   • Sentiment                       ├───► │  alertas     │
 Coswara (voz rotulada) ──────►           camada acústica (local)    │     │  (SQLite)    │
                                   • MFCC/pitch/jitter/shimmer/pausas│     └──────┬───────┘
                                   • classificador → Alteração Vocal │            │
                                                                     │            ▼
 MIMIC-III Waveform ──────────►  vitals/  Monitoramento Contínuo     │     ┌──────────────┐
   (numerics FC/FR/SpO2/PA)        • replay em streaming             │     │  Fusão       │
                                   • regras + z-score + IsolForest   │     │  (regras,    │
                                                                     │     │  janela de   │
 MIMIC-IV Demo ───────────────►  prescriptions/  regras              │     │  tempo)      │
   (prescriptions/emar)            • Mudança Terapêutica Atípica     ┘     └──────┬───────┘
                                                                                  │ Alerta Composto
 Pacientes de Demonstração (data/demo_patients.json) ──── liga as fontes ────────┤
                                                                                  ▼
                                               ┌─────────────────────────────────────────────┐
                                               │ app.py (Streamlit)                          │
                                               │  1. Central de Alertas  (Reconhecimento)    │
                                               │  2. Painel do Paciente  (replay ao vivo)    │
                                               │  3. Laboratório de Análise (upload)         │
                                               └─────────────────────────────────────────────┘
```

---

## 3. Contrato do Alerta (definido pelo Antonio até 05/10)

Todo Analisador grava Alertas **exatamente** neste formato (`src/core/alerts.py`). O Streamlit consome esse formato desde o dia 1 com dados mock.

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `id` | str (uuid) | |
| `paciente_id` | str | id do Paciente de Demonstração (ou `avulso` no Laboratório) |
| `origem` | enum | `video` · `audio` · `sinais_vitais` · `prescricao` · `fusao` |
| `tipo` | str | ex.: `desvio_execucao`, `queda`, `termo_critico`, `alteracao_vocal`, `sinal_vital`, `mudanca_terapeutica`, `composto` |
| `severidade` | enum | `baixa` · `media` · `alta` · `critica` |
| `descricao` | str | frase legível para a equipe médica |
| `evidencia` | JSON | ex.: `{"frame": "path.jpg", "t": 12.4, "angulo": 62}`, `{"trecho": "...", "inicio": 83.2}`, `{"serie": "SpO2", "janela": [t0, t1], "valores": [...]}` |
| `detectado_em` | datetime | momento (simulado) da detecção |
| `alertas_origem` | JSON list | só em Alerta Composto: ids dos Alertas fundidos |
| `status` | enum | `novo` → `reconhecido` → `resolvido` |
| `reconhecido_por`, `reconhecido_em` | str, datetime | preenchidos no Reconhecimento |

---

## 4. Analisador de Vídeo — Especificação

**Dados:** 6–10 clipes curtos (10–40s) com licença livre: exercícios de fisioterapia (agachamento, elevação lateral de braço, flexão de joelho) + 2–3 clipes de queda. Salvar a licença/URL de cada um em `data/video/SOURCES.md`.

**Pipeline (`src/analyzers/video/`):**

1. `pose.py`: YOLOv8n-pose frame a frame → 17 keypoints por pessoa.
2. `angles.py`: ângulos articulares (joelho, quadril, ombro, cotovelo) por frame.
3. `exercise_rules.py`: por exercício, faixa esperada de amplitude + simetria esquerda/direita → **Desvio de Execução** (amplitude insuficiente, compensação de tronco, assimetria > X°).
4. `fall.py`: razão altura/largura do bounding box + queda brusca do centro de massa (keypoints de quadril) → **Queda**.
5. `report.py`: **Relatório de Sessão** (Markdown/HTML) com os Desvios, o frame de Evidência anotado com o esqueleto e o gráfico de ângulo ao longo do tempo.

**Resultados para o relatório:** tabela por clipe (desvios esperados × detectados), frames anotados, tempo de processamento em CPU. Opcional: comparação com MediaPipe (aula 03).

---

## 5. Analisador de Áudio — Especificação

**Setup Azure (primeira semana, Vinicius Geizler):** conta Azure for Students; recursos **Speech** e **Language** em tier F0 (Speech: 5h/mês; Language: 5.000 registros/mês). Chaves no `.env` de cada um, **nunca no git**. Orçamento de cota: processar ~15–20 Consultas do PriMock57 em batch (não as 57) e reservar cota para as demos de upload.

**Camada semântica (`src/analyzers/audio/semantic.py`):**

1. Converter o áudio do paciente/consulta para WAV 16 kHz mono (`pydub`/ffmpeg).
2. Azure Speech to Text (`en-US`, com timestamps por palavra).
3. **WER** contra a transcrição manual do PriMock57 (`jiwer`).
4. Text Analytics for Health → entidades (`SymptomOrSign`, `MedicationName`, `Diagnosis`…) → **Termos Críticos** (lista curada: dyspnea, chest pain, fatigue, wheezing…).
5. Sentiment por fala do paciente.

**Camada acústica (`src/analyzers/audio/acoustic.py`):**

1. Subset do Coswara: gravações `counting-normal` + `vowel-a` de pessoas com `breathing difficulty`/`fatigue` vs. assintomáticas saudáveis (balancear classes, ~300–600 por classe).
2. Features: MFCC (média/desvio), pitch (F0), jitter, shimmer, HNR (`parselmouth`), proporção de pausas, taxa de fala.
3. Classificador: Regressão Logística e Random Forest (scikit-learn), split estratificado → **AUC, F1, matriz de confusão**.
4. Na demo: roda sobre a fala do paciente no PriMock57 como inferência → **Alteração Vocal** se a probabilidade passar do limiar.

**Resultados para o relatório:** WER médio do Azure, exemplos de Termos Críticos extraídos, métricas do classificador acústico, e a limitação explícita (as camadas são avaliadas em datasets diferentes).

---

## 6. Analisador de Sinais Vitais — Especificação

**Dados:** `wfdb` lendo ~10–20 registros *numerics* do MIMIC-III Waveform Database (escolher registros com FC, SpO2 e PA presentes e poucas lacunas).

**Pipeline (`src/analyzers/vitals/`):**

1. `load.py`: leitura, reamostragem para 1/min, tratamento de lacunas.
2. `detectors.py`: três técnicas com a mesma interface `detect(janela) -> list[Anomalia]`:
   - **Regras clínicas:** SpO2 < 90, FC > 120 ou < 45, PAS > 180 ou < 90, FR > 25.
   - **z-score em janela móvel:** desvio > 3σ da própria história recente do paciente.
   - **Isolation Forest:** sobre janelas multivariadas (FC, SpO2, PAS, FR).
3. `inject.py` + `evaluate.py`: injeta anomalias conhecidas (pico de FC, queda gradual de SpO2, hipotensão súbita) em trechos limpos → **precisão / recall / atraso de detecção** por técnica.
4. `stream.py`: **replay** acelerado (ex.: 1 min de dado = 1 s de parede), alimentando os detectores janela a janela e gravando Alertas conforme ocorrem.

---

## 7. Analisador de Prescrições — Especificação

**Dados:** MIMIC-IV Clinical Database Demo (`hosp/prescriptions`, `hosp/emar`).

**Regras (`src/analyzers/prescriptions/rules.py`) → Mudança Terapêutica Atípica:**

- Aumento de dose > 50% da mesma droga em < 24h.
- Início de droga de alta vigilância (anticoagulantes, insulina, opioides, sedativos).
- Mais de N trocas ou suspensões em 48h.
- Duplicidade terapêutica (duas drogas da mesma classe ativas ao mesmo tempo).

**Resultado para o relatório:** linha do tempo de prescrições de 1–2 pacientes com os Alertas marcados; contagem de Alertas por regra nos 100 pacientes.

---

## 8. Fusão e Pacientes de Demonstração

**Pacientes de Demonstração (`data/demo_patients.json`, Vinicius Blasque):** 3 pacientes fictícios, cada um ligando um registro do MIMIC-III (sinais vitais), um paciente do MIMIC-IV Demo (prescrições), uma Consulta do PriMock57 e um clipe de vídeo, com offsets de tempo alinhando tudo numa mesma internação. No relatório, declarar explicitamente que são **compostos para demonstração**.

**Regras de Fusão (`src/core/fusion.py`, Antonio):** mesma janela (ex.: 24h simuladas) e mesmo paciente:

| Alertas de origem | Alerta Composto | Severidade |
| --- | --- | --- |
| Sinal Vital (SpO2/FR) + Alteração Vocal ou Termo Crítico respiratório | Possível insuficiência respiratória | crítica |
| Queda + Mudança Terapêutica (opioide/sedativo iniciado) | Queda possivelmente associada a medicação | alta |
| Sinal Vital (PA/FC) + Mudança Terapêutica (anti-hipertensivo/antiarrítmico) | Possível efeito adverso de medicação | alta |
| Desvio de Execução recorrente + Termo Crítico de dor | Possível agravamento em reabilitação | média |

Pelo menos **um** Paciente de Demonstração precisa disparar um Alerta Composto crítico: é o clímax do vídeo.

---

## 9. Interface Streamlit — 3 Telas

### Tela 1: Central de Alertas

- Lista de Alertas e Alertas Compostos, ordenada por severidade e horário; filtros por origem, status e paciente.
- Ao abrir: Evidência renderizada (frame anotado, player do trecho de áudio + transcrição, gráfico da janela do sinal), e para um Composto, os Alertas de origem.
- Botão **Reconhecer** (grava `reconhecido_por` e `reconhecido_em`) e **Resolver**.

### Tela 2: Painel do Paciente

- Seletor de Paciente de Demonstração.
- **Replay ao vivo** dos sinais vitais com os Alertas surgindo sobre o gráfico.
- Linha do tempo de prescrições, Consultas e vídeos com marcadores de Alerta.

### Tela 3: Laboratório de Análise

- Upload de vídeo → esqueleto sobreposto, ângulos, Relatório de Sessão.
- Upload de áudio → transcrição do Azure, Termos Críticos, sentimento, métricas acústicas, probabilidade de Alteração Vocal.

---

## 10. Divisão de Responsabilidades e Cronograma

Marcos globais:

| Data | Marco |
| --- | --- |
| **05/10** | Contrato do Alerta + schema SQLite prontos; datasets baixados; Azure funcionando; esqueleto do relatório criado |
| **19/10** | Cada Analisador gera Alertas reais a partir dos seus dados (sem interface) |
| **26/10** | Streamlit consumindo Alertas reais; Pacientes de Demonstração montados; Fusão funcionando |
| **04/11** | **Feature freeze.** Cada dono entrega: código + testes passando + sua seção do relatório com resultados reais |
| **11/11** | Relatório consolidado; roteiro do vídeo aprovado pelo time |
| **15/11** | Vídeo gravado e publicado (não listado) |
| **17/11** | **Entrega** |

### Marcelo Costa — Analisador de Vídeo (~38h até 04/11)

**28/09–05/10:** Curadoria de 6–10 clipes + `SOURCES.md`; setup Ultralytics; `pose.py` rodando num clipe.
**06/10–19/10:** `angles.py`, `exercise_rules.py`, `fall.py` gerando Alertas no contrato.
**20/10–04/11:** `report.py` (Relatório de Sessão), função chamável pela Tela 3, testes (`tests/test_video_*.py`), seção 3.1 do relatório com resultados.
**05/11–17/11:** Correções pós-integração; apoio ao Renato no trecho de vídeo da demo.

**Entregável:** `src/analyzers/video/` + clipes processados + Relatórios de Sessão em `results/video/`.

### Vinicius Geizler — Analisador de Áudio (~38h até 04/11)

**28/09–05/10:** Conta Azure for Students + recursos Speech e Language; `.env.example`; download PriMock57 (Git LFS) e subset do Coswara.
**06/10–19/10:** Camada semântica completa (STT, WER, TA for Health, Sentiment) em ~15–20 Consultas; extração de features do Coswara.
**20/10–04/11:** Classificador acústico + métricas; Alertas de Termo Crítico e Alteração Vocal no contrato; função chamável pela Tela 3; testes; seção 3.2 do relatório.
**05/11–17/11:** Correções; garantir cota do Azure disponível no dia da gravação.

**Entregável:** `src/analyzers/audio/` + `results/audio/` (WER, métricas, exemplos).

### Antonio Bazo — Sinais Vitais + Núcleo de Alertas + Fusão (~38h até 04/11)

**28/09–05/10:** **Contrato do Alerta** + `src/core/db.py` (schema SQLite) + dados mock para o Streamlit; selecionar e baixar registros MIMIC-III.
**06/10–19/10:** Três detectores + `inject.py`/`evaluate.py` com precisão/recall.
**20/10–04/11:** `stream.py` (replay); `fusion.py` com as regras da seção 8; testes; seção 3.3 do relatório + diagrama do fluxo multimodal.
**05/11–17/11:** Correções; ajustar velocidade do replay para a gravação.

**Entregável:** `src/core/` + `src/analyzers/vitals/` + `results/vitals/`.

### Vinicius Blasque — Interface + Prescrições + Pacientes de Demonstração (~38h até 04/11)

**28/09–05/10:** `app.py` com navegação das 3 telas; Tela 1 sobre dados mock do contrato; **criar esqueleto de `docs/relatorio_tecnico.md`** (seção 12).
**06/10–19/10:** Analisador de Prescrições (MIMIC-IV Demo) gerando Alertas; Tela 2 com replay (integrando `stream.py` quando pronto).
**20/10–04/11:** `demo_patients.json`; Tela 3 integrando vídeo e áudio; substituir mocks por dados reais; testes; seção 3.4 e 4 do relatório.
**05/11–17/11:** Correções de interface durante a gravação.

**Entregável:** `app.py` + `src/analyzers/prescriptions/` + `data/demo_patients.json`.

### Renato Mattos — Entrega Final (05/11–17/11, ~13h)

**05/11–08/11:** Rodar o sistema do zero seguindo o README (teste de reprodutibilidade); abrir issues para o que quebrar; revisar e costurar o relatório.
**09/11–11/11:** Roteiro do vídeo (seção 13) validado com o time; ensaio da demo.
**12/11–15/11:** Gravação, edição mínima, upload no YouTube (não listado).
**16/11–17/11:** Checklist da seção 14, link do vídeo no README, entrega.

**Entregável:** vídeo publicado + `docs/relatorio_tecnico.md` final + README + entrega na plataforma.

---

## 11. Estrutura de Arquivos

```text
monitoramento-multimodal-fase4/
├── app.py                               ← Streamlit, 3 telas (V. Blasque)
├── src/
│   ├── core/
│   │   ├── alerts.py                    ← contrato do Alerta (Antonio)
│   │   ├── db.py                        ← SQLite (Antonio)
│   │   └── fusion.py                    ← Fusão / Alerta Composto (Antonio)
│   └── analyzers/
│       ├── video/                       ← pose, angles, exercise_rules, fall, report (Marcelo)
│       ├── audio/                       ← azure (config), semantic (Azure), acoustic (Coswara) (V. Geizler)
│       ├── vitals/                      ← load, detectors, inject, evaluate, stream (Antonio)
│       └── prescriptions/               ← rules (V. Blasque)
├── scripts/
│   ├── download_data.py                 ← baixa subsets de PhysioNet/Coswara/PriMock57
│   ├── azure_smoke_test.py              ← valida chaves/cota do Azure (STT, TA for Health, Sentiment)
│   └── run_batch.py                     ← processa tudo e popula o SQLite
├── data/
│   ├── video/ (+ SOURCES.md)
│   ├── audio/  primock57/ coswara/
│   ├── vitals/ mimic3wdb/
│   ├── prescriptions/ mimic4demo/
│   └── demo_patients.json
├── results/                             ← métricas e artefatos por modalidade
├── tests/
├── docs/
│   ├── azure-setup.md                   ← conta, região, cotas F0 e Plano B do Azure
│   └── relatorio_tecnico.md
├── pyproject.toml + uv.lock            ← dependências (uv); requirements.txt exportado para quem não usa uv
├── .devcontainer/                       ← ambiente padrão do time (Python 3.12, uv, ffmpeg, git-lfs, gh, torch CPU)
├── .env.example
└── README.md
```

> Datasets brutos **não** vão para o git (`data/*` no `.gitignore`, exceto `demo_patients.json` e `SOURCES.md`). `scripts/download_data.py` reproduz tudo.

---

## 12. Dependências (`pyproject.toml` + `uv.lock`)

Gerenciadas com **uv**; `uv.lock` versionado. Adicionar dependência: `uv add <pacote>` (ou `uv add --group dev <pacote>`) e commitar `pyproject.toml` + `uv.lock` juntos.

```text
ultralytics>=8.2          # YOLOv8-pose
torch, torchvision        # wheels CPU no Linux (índice pytorch-cpu); sem GPU no escopo
opencv-python>=4.9
azure-cognitiveservices-speech>=1.37
azure-ai-textanalytics>=5.3   # TA for Health + Sentiment
librosa>=0.10
praat-parselmouth>=0.4
pydub>=0.25               # + ffmpeg instalado no sistema
jiwer>=3.0                # WER
wfdb>=4.1                 # PhysioNet
pandas>=2.2
scikit-learn>=1.4
plotly>=5.20
streamlit>=1.35
python-dotenv>=1.0
# grupo dev: pytest, ruff, pyright, ipykernel
```

**Checagens:** `uv run pytest`, `uv run ruff check .`, `uv run pyright` (mesma configuração do Pylance, em `[tool.pyright]` do `pyproject.toml`).

**Ambiente:** o devcontainer (`.devcontainer/`) é o padrão do time: já traz ffmpeg, git-lfs, gh (autenticado via `GITHUB_TOKEN`), libs do OpenCV/Azure Speech e roda `uv sync --frozen` na criação. Fora do container: instalar ffmpeg + git-lfs e rodar `uv sync`. Para avaliadores sem uv: `uv export --no-dev --no-hashes -o requirements.txt` e `pip install -r requirements.txt`.

`.env`: `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT`. `.env` (na raiz, lido via `python-dotenv`) no `.gitignore`; `.devcontainer/.env` guarda só credenciais de git do container.

---

## 13. Relatório Técnico e Roteiro do Vídeo

### Relatório (`docs/relatorio_tecnico.md`)

```markdown
# Relatório Técnico — Tech Challenge Fase 4
## 1. Contexto (continuidade com o assistente da Fase 3)
## 2. Fluxo Multimodal (diagrama: fontes → Analisadores → Alertas → Fusão → Central)
## 3. Modelos por Tipo de Dado
### 3.1 Vídeo — YOLOv8-pose, Desvio de Execução, Queda            (Marcelo)
### 3.2 Áudio — Azure STT + TA for Health + classificador acústico (V. Geizler)
### 3.3 Sinais Vitais — regras × z-score × Isolation Forest        (Antonio)
### 3.4 Prescrições — regras sobre MIMIC-IV Demo                   (V. Blasque)
## 4. Fusão e Pacientes de Demonstração
## 5. Resultados e Exemplos de Anomalias Detectadas
## 6. Integração com Azure (serviços, custos, cotas)
## 7. Limitações (datasets distintos, pacientes compostos, dados em inglês)
## 8. Conclusão e Trabalhos Futuros (AWS Textract/Comprehend, OpenAI, TORGO, notificação externa)
```

### Vídeo (≤ 15 min, Renato)

| Tempo | Conteúdo | Requisito do PDF |
| --- | --- | --- |
| 0–1:30 | Contexto: hospital, monitoramento multimodal, arquitetura | — |
| 1:30–4:30 | Laboratório: upload de vídeo de fisioterapia → esqueleto, Desvio, Relatório de Sessão; clipe de Queda | análise de vídeo |
| 4:30–7:30 | Laboratório: upload de Consulta → Azure STT, Termos Críticos, sentimento, Alteração Vocal | análise de áudio + Azure |
| 7:30–8:30 | Portal Azure: recursos Speech e Language, chamadas registradas | integração dos serviços Azure |
| 8:30–12:00 | Painel do Paciente: replay dos sinais vitais ao vivo, Alertas surgindo, prescrição atípica, **Alerta Composto crítico** | detecção e resposta a anomalias |
| 12:00–14:00 | Central de Alertas: abrir Evidência, Reconhecer, Resolver | fluxo final do alerta à equipe |
| 14:00–15:00 | Resultados (métricas) e conclusão | — |

---

## 14. Critérios Mínimos para Aprovação

- [ ] Vídeo: YOLOv8-pose detecta Desvio de Execução e Queda, com Relatório de Sessão gerado automaticamente
- [ ] Áudio: Azure Speech to Text transcreve Consultas; WER medido contra PriMock57
- [ ] Áudio: Azure Text Analytics (for Health + Sentiment) extrai Termos Críticos e sentimento
- [ ] Áudio: classificador acústico treinado no Coswara, com AUC/F1 reportados
- [ ] Sinais vitais: três técnicas comparadas com anomalias injetadas (precisão/recall)
- [ ] Prescrições: Mudanças Terapêuticas Atípicas detectadas no MIMIC-IV Demo
- [ ] Monitoramento Contínuo (replay) gerando Alertas durante a execução
- [ ] Fusão gera pelo menos um Alerta Composto crítico num Paciente de Demonstração
- [ ] Central de Alertas com Evidência + Reconhecimento funcionando
- [ ] `pytest` passa
- [ ] Relatório com fluxo multimodal, modelos por tipo de dado, resultados e exemplos de anomalias
- [ ] Vídeo ≤15 min no YouTube/Vimeo (não listado) cobrindo áudio, vídeo, anomalias, Azure e fluxo do alerta
- [ ] Nenhuma chave do Azure nem dataset bruto no git
- [ ] README permite rodar do zero (validado pelo Renato em 05/11)

---

## 15. Riscos e Plano B

| Risco | Probabilidade | Plano B |
| --- | --- | --- |
| Azure for Students não liberado / cartão exigido | Média | Criar na primeira semana, em paralelo, por 2 pessoas; fallback: conta free com cartão de um integrante (F0 não cobra) |
| Cota F0 do Azure estoura antes da gravação | Média | Processar em batch só 15–20 Consultas; cachear respostas do Azure em JSON; reservar cota para o dia da gravação |
| YOLOv8-pose lento em CPU | Baixa | Usar `yolov8n-pose`, reduzir resolução/FPS (processar 1 a cada 2–3 frames) |
| Classificador do Coswara com desempenho fraco (AUC ~0,6) | Média | Reportar honestamente como resultado; o valor está no pipeline e na análise, não no número |
| Registros MIMIC-III com muitas lacunas | Média | Selecionar registros com cobertura de FC+SpO2+PA antes de 05/10 |
| Fusão não encontra coincidências naturais | Média | Offsets do Paciente de Demonstração são escolhidos para alinhar as Anomalias (declarado no relatório) |
| Integrante atrasa sua frente até 04/11 | Média | Streamlit roda com dados mock do contrato; relatório documenta o que ficou pronto |
| Renato encontra o sistema quebrado em 05/11 | Média | Marco de 26/10 exige Streamlit rodando com dados reais; teste de reprodutibilidade é a primeira tarefa do Renato |

---

## 16. Comunicação do Time

- **Check-in semanal** (domingos): ✅ feito / 🔴 bloqueios, alinhado aos marcos da seção 10.
- **Dependências críticas, avisar imediatamente:** contrato do Alerta (Antonio, 05/10), Azure funcionando (V. Geizler, 05/10), `stream.py` (Antonio), `demo_patients.json` (V. Blasque).
- **Branches:** `feature/video`, `feature/audio`, `feature/vitals-core`, `feature/app-prescriptions`, `feature/docs`. PR para `main` quando estável.
- **Handoff para o Renato (04/11):** cada dono deixa no PR final um parágrafo "como rodar minha parte" e "o que mostrar no vídeo".
- **Não reabrir escopo:** ideias novas vão para "Trabalhos Futuros".

---

*Gerado a partir de sessão de grilling em 2026-09-28, com base no PDF do Tech Challenge Fase 4, nas aulas da fase e na estrutura da estratégia da Fase 3.*
