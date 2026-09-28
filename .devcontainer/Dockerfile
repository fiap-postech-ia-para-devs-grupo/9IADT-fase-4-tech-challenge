FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# ffmpeg: pydub (conversão de áudio para WAV 16 kHz)
# git-lfs: download do PriMock57
# libgl1, libglib2.0-0: OpenCV (dependência do ultralytics)
# libasound2, libssl3: Azure Speech SDK no Linux
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libasound2 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -eu' \
    'prompt=${1:-}' \
    'case "$prompt" in' \
    '  *Username*) printf "%s\n" "x-access-token" ;;' \
    '  *Password*) printf "%s\n" "${GITHUB_TOKEN:-}" ;;' \
    '  *) printf "\n" ;;' \
    'esac' \
    > /usr/local/bin/github-askpass \
    && chmod +x /usr/local/bin/github-askpass

# Named volume montado aqui (cache do uv: rebuild não baixa tudo de novo); cria com o dono certo
RUN mkdir -p /home/vscode/.cache && chown vscode:vscode /home/vscode/.cache

WORKDIR /workspaces/9IADT-fase-4-tech-challenge
