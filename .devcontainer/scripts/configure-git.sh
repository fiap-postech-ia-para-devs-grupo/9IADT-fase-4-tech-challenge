#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config --global user.name "$GIT_USER_NAME"
fi

if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config --global user.email "$GIT_USER_EMAIL"
fi

git config --global core.askPass /usr/local/bin/github-askpass
git config --global credential.helper ""
git config --global credential.useHttpPath true
git config --global --unset-all url."https://github.com/".insteadOf 2>/dev/null || true
git config --global --add url."https://github.com/".insteadOf "git@github.com:"
git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/"
