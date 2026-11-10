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
pip install -r requirements.txt
```

**2. Credenciais do Azure**

Copie `.env.example` para `.env` e preencha as quatro variáveis. O `.env` não vai para o git.

```bash
cp .env.example .env
```

**3. Dados**

Os datasets brutos não vão para o git. O script baixa cada um para `data/`:

```bash
uv run python scripts/download_data.py mimic3wdb mimic4demo
```

Os nomes disponíveis são `video`, `primock57`, `coswara`, `mimic3wdb` e `mimic4demo`. Rodar sem argumentos baixa todos.

**4. Processar tudo e popular o banco**

```bash
uv run python scripts/run_batch.py
```

O script roda cada modalidade em sequência (sinais vitais, prescrições, áudio, vídeo e Pacientes de Demonstração) e termina com um resumo: o que deu certo, o que está pendente e o que falhou. Uma etapa que falha não impede as outras.

**5. Abrir a interface**

```bash
uv run streamlit run app.py
```

A interface tem três telas: a **Central de Alertas** (com Reconhecimento e Resolução), o **Painel do Paciente** (replay ao vivo dos sinais vitais) e o **Laboratório de Análise** (upload de vídeo e de áudio).

## Testes

```bash
uv run pytest
uv run ruff check .
```

Os testes de áudio usam o Azure mockado, então rodam sem chave.

## Estrutura

```
app.py                 interface Streamlit (3 telas)
src/core/              contrato do Alerta, SQLite, Fusão e Pacientes de Demonstração
src/analyzers/         um Analisador por modalidade: video, audio, vitals, prescriptions
src/ui/                telas do Streamlit
scripts/               download, processamento em lote e avaliações
data/                  datasets brutos (fora do git), lista de registros e Pacientes de Demonstração
results/               métricas e artefatos de cada modalidade
docs/                  relatório técnico, decisões (ADRs) e setup do Azure
```

## Relatório e resultados

O relatório técnico está em [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md). Os resultados de cada modalidade estão em `results/`.
