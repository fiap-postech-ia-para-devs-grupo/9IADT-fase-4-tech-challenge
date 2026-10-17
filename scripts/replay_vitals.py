"""Replay dos Sinais Vitais reais gravando os Alertas no SQLite (ESTRATEGIA.md, seção 6).

    uv run python scripts/replay_vitals.py

Para cada registro em `data/vitals/selecionados.txt`, roda `stream.replay` sem pausa e grava cada
Alerta no banco (`data/alertas.db`, ou `ALERTAS_DB`). Reexecutar não duplica: os ids são
determinísticos e `db.salvar` sobrescreve.
"""

from __future__ import annotations

import pathlib

from src.analyzers.vitals import load, stream
from src.core import db

RAIZ = pathlib.Path("data/vitals/mimic3wdb")


def main() -> None:
    conn = db.conectar()
    registros = [
        linha.strip()
        for linha in (RAIZ.parent / "selecionados.txt").read_text().splitlines()
        if linha.strip()
    ]
    total = 0
    for rel in registros:
        paciente = rel.rsplit("/", 1)[-1]
        janela = load.carregar(RAIZ / rel)
        gravados = sum(
            1
            for _ in stream.replay(
                janela,
                paciente,
                minutos_por_segundo=0,
                ao_detectar=lambda alerta: db.salvar(conn, alerta),
            )
        )
        total += gravados
        print(paciente, gravados, flush=True)
    print("alertas gravados", total, flush=True)


if __name__ == "__main__":
    main()
