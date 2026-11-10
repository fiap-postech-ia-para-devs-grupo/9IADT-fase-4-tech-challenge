"""Processa todas as modalidades e popula o SQLite com um comando (ESTRATEGIA.md, seção 11).

    uv run python scripts/run_batch.py

Roda, nesta ordem, os scripts de cada modalidade e depois a Fusão dos Pacientes de Demonstração.
Cada etapa roda em outro processo: se uma falhar, as outras continuam e o resumo no fim mostra o
que deu certo, o que está pendente e o que falhou. Os dados precisam estar baixados antes
(`scripts/download_data.py`), e as chaves do Azure no `.env` para a etapa de áudio.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]

# (nome, script ou None, motivo se não houver script ainda)
ETAPAS: tuple[tuple[str, str | None, str], ...] = (
    ("sinais vitais", "scripts/replay_vitals.py", ""),
    ("prescrições", "scripts/analisar_prescricoes.py", ""),
    ("áudio (Azure)", "scripts/processar_audio_semantico.py", ""),
    ("vídeo", None, "Analisador de Vídeo ainda não entregue (#6)"),
    ("pacientes de demonstração", "scripts/gerar_pacientes_demo.py", ""),
)


def rodar(script: str) -> tuple[bool, str]:
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ)}
    processo = subprocess.run(
        [sys.executable, str(RAIZ / script)],
        cwd=RAIZ,
        env=ambiente,
        capture_output=True,
        text=True,
    )
    if processo.returncode == 0:
        return True, ""
    ultima = (processo.stderr or processo.stdout).strip().splitlines()[-1:] or [""]
    return False, ultima[0]


def main() -> int:
    resultados: list[tuple[str, str]] = []
    for nome, script, motivo in ETAPAS:
        if script is None:
            resultados.append((nome, f"pendente: {motivo}"))
            continue
        print(f"-> {nome}", flush=True)
        ok, erro = rodar(script)
        resultados.append((nome, "ok" if ok else f"falhou: {erro}"))

    print("\nResumo:")
    for nome, situacao in resultados:
        print(f"  {nome}: {situacao}")
    falhas = [n for n, s in resultados if s.startswith("falhou")]
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
