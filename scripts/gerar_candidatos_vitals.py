"""Filtra os candidatos da varredura por duração e grava a lista para a seleção.

    uv run python scripts/gerar_candidatos_vitals.py

Lê `data/vitals/scan2_log.txt` (saída de `escanear_registros_vitals.py`) e grava em
`data/vitals/candidatos2.txt` os registros com 1 dia a 20 mil minutos de dados (1440 a 20000
amostras, já que o sinal é de 1 amostra por minuto). Depois `selecionar_registros_vitals.py` baixa
e mede cada um.
"""

from __future__ import annotations

import pathlib

LOG = pathlib.Path("data/vitals/scan2_log.txt")
SAIDA = pathlib.Path("data/vitals/candidatos2.txt")
MIN_AMOSTRAS, MAX_AMOSTRAS = 1440, 20000
# Os 10 registros de `data/vitals/selecionados.txt` saíram dos 38 primeiros candidatos.
# Manter esse limite reproduz exatamente a mesma seleção.
LIMITE = 38


def main() -> None:
    candidatos = []
    for linha in LOG.read_text().splitlines():
        if not linha.startswith("OK "):
            continue
        partes = linha.split()
        registro, amostras = partes[1], int(partes[2])
        if MIN_AMOSTRAS <= amostras <= MAX_AMOSTRAS:
            candidatos.append(registro)
    SAIDA.write_text("\n".join(candidatos[:LIMITE]) + "\n")
    print(
        f"candidatos com 1 dia ou mais: {len(candidatos)}; gravados: {min(len(candidatos), LIMITE)}"
    )


if __name__ == "__main__":
    main()
