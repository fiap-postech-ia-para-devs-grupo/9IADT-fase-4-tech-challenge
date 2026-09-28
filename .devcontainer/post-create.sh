#!/usr/bin/env bash

set -e

bash .devcontainer/scripts/configure-git.sh

git lfs install

uv sync --frozen

uv run python -m ipykernel install --user --name monitoramento-fase4 --display-name "Python (monitoramento-fase4)"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Criado .env a partir de .env.example — preencha as chaves do Azure."
fi

# .claude/skills -> .agents/skills (link relativo: funciona no host e no container)
mkdir -p .agents/skills .claude
if [[ -e .claude/skills && ! -L .claude/skills ]]; then
  echo "Aviso: .claude/skills é um diretório real; não substituído pelo link para .agents/skills."
else
  ln -sfn ../.agents/skills .claude/skills
fi
