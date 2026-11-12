# Monitoramento Multimodal de Pacientes

Tech Challenge da Fase 4 (FIAP, pós-graduação em IA para Devs). O sistema monitora pacientes a partir de vídeo, áudio, sinais vitais e prescrições, gera **Alertas** para a equipe médica e os funde em **Alertas Compostos** quando sinais de diferentes origens se confirmam. Vocabulário e decisões em [CONTEXT.md](CONTEXT.md) e [ESTRATEGIA.md](ESTRATEGIA.md).

> Os pacientes da demonstração são **compostos**: cada um junta registros reais de fontes públicas diferentes sob uma única identidade (ver [ADR 0001](docs/adr/0001-paciente-de-demonstracao-composto.md)).

## Pré-requisitos

Escolha uma das duas opções:

- **Devcontainer (recomendado):** VS Code ou outro editor com suporte a devcontainers e Docker. O container já traz Python 3.12, uv, ffmpeg, git-lfs e as bibliotecas do OpenCV e do Azure Speech.
- **Local:** Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg e git-lfs instalados no sistema.

Para a etapa de áudio, é preciso uma conta Azure for Students com os recursos Speech e Language em tier F0. O passo a passo está em [docs/azure-setup.md](docs/azure-setup.md).

## Passo a passo do zero

**1. Ambiente**

Com devcontainer, abra o repositório no container: o `post-create.sh` roda `uv sync --frozen` sozinho.

Local:

```bash
uv sync --frozen
```

Sem uv, use o `requirements.txt` exportado do lock:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

O `--extra-index-url` é necessário no Linux, porque o `torch` CPU vem do índice do PyTorch.

**2. Credenciais do Azure**

Copie `.env.example` para `.env` e preencha as variáveis do Azure. O `.env` não vai para o git. Sem as chaves, o lote de áudio roda só do cache versionado (o upload de áudio no Laboratório precisa delas).

Opcional: para a equipe receber as chamadas de emergência no WhatsApp, preencha `EVOLUTION_URL`, `EVOLUTION_APIKEY` e `EVOLUTION_INSTANCE` (Evolution API) e `APP_URL` (link da Central na mensagem). Sem elas, o acionamento fica simulado e só é registrado.

```bash
cp .env.example .env
```

**3. Dados**

Os datasets brutos não vão para o git. O script baixa cada um para `data/`:

```bash
uv run python scripts/download_data.py mimic3wdb mimic4demo video sons
```

Os nomes disponíveis são `video`, `sons`, `primock57`, `coswara`, `mimic3wdb` e `mimic4demo`. Rodar sem argumentos baixa todos. Os pesos do YOLOv8 e dos modelos de som (AST/AudioSet e Whisper) são baixados na primeira execução.

**4. Processar tudo e popular o banco**

```bash
uv run python scripts/run_batch.py
```

O script roda cada modalidade em sequência (sinais vitais, prescrições, áudio, vídeo e Pacientes de Demonstração) e termina com um resumo: o que deu certo, o que está pendente e o que falhou. Uma etapa que falha não impede as outras.

**5. Abrir a interface**

```bash
uv run streamlit run app.py
```

O menu segue a rotina de um plantão:

| Seção | Tela | Para quê |
| --- | --- | --- |
| — | **Início** | Painel do plantão (ocupação, chamados, alertas por severidade, origem e tipo, NEWS2 por leito, equipe) e como o sistema funciona |
| Monitoramento | **Mapa de Leitos** | Um cartão por leito: sinais atuais, NEWS2, luz de chamada; leitos livres |
| Monitoramento | **Central de Alertas** | Fila única (episódios agrupados, escalonamento), Evidência, Reconhecimento com conduta |
| Paciente | **Admissão** | Dar entrada no paciente e ligar as fontes: monitor, prescrição, consulta, câmera |
| Paciente | **Prontuário** | Monitor de beira de leito, tendências, linha do tempo, medicações, alertas, **câmera e microfone do leito**, evolução e alta |
| Análise | **Laboratório de Análise** | Vídeo (clipe, upload ou webcam) e áudio (upload, microfone ou exemplo) sob demanda |
| Administração | **Equipe e Leitos** | Profissionais com WhatsApp, leitos, histórico de acionamentos |

**Como demonstrar em 5 minutos:** escolha o plantonista na barra lateral, clique em **▶ Iniciar** no relógio do plantão e acompanhe o Mapa de Leitos. Num Prontuário, abra **Câmera e microfone**, ligue vídeo e áudio (clipe `queda_01` e a gravação "Enfermeira! Socorro!") e veja a luz do leito acender, o Alerta chegar à Central e a equipe ser acionada. O tour (🧭 na barra) percorre o sistema todo.

## Testes

```bash
uv run pytest
uv run ruff check .
```

Os testes de áudio usam o Azure mockado, então rodam sem chave.

## Estrutura

```
app.py                 interface Streamlit (menu por seções)
src/core/              contrato do Alerta, SQLite, Fusão, Pacientes de Demonstração, cadastros
                       (leitos, equipe, admissões, evolução), NEWS2, triagem e acionamento WhatsApp
src/analyzers/         um Analisador por modalidade: video, audio, vitals, prescriptions
src/ui/                telas do Streamlit
scripts/               download, processamento em lote e avaliações
data/                  datasets brutos (fora do git), lista de registros e Pacientes de Demonstração
results/               métricas e artefatos de cada modalidade
docs/                  relatório técnico, decisões (ADRs) e setup do Azure
```

## Relatório e resultados

O relatório técnico está em [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md). Os resultados de cada modalidade estão em `results/`.
