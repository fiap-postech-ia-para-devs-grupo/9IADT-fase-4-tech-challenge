# Imagem única do projeto: dependências, dados públicos, lote processado e modelos já baixados.
#
#   docker build -t monitoramento .                      # ~15 min na primeira vez (baixa tudo)
#   docker run --rm -p 8501:8501 monitoramento           # abre em http://localhost:8501
#   docker run --rm -p 8501:8501 --env-file .env monitoramento   # com Azure/Telegram (opcional)
#
# As chaves nunca entram na imagem (.env está no .dockerignore): vão só no `docker run`.

FROM python:3.12-slim-bookworm

# ffmpeg: conversão de áudio; libgl1/libglib2.0-0: OpenCV; libasound2/libssl3: Azure Speech SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libasound2 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependências antes do código: mudar o código não reinstala o torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# Dados da demonstração (~40 MB), lote de todos os Analisadores e pesos do YOLOv8.
RUN python scripts/download_data.py mimic3wdb mimic4demo video sons \
    && python scripts/run_batch.py

# Modelos de som do leito (AST/AudioSet e Whisper): baixados aqui, o app abre sem esperar.
RUN python -c "from src.analyzers.audio.ambiente import Classificador; Classificador()"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
