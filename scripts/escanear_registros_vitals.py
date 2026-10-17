"""Varre os cabeçalhos dos registros numerics do MIMIC-III e lista os candidatos.

    uv run python scripts/escanear_registros_vitals.py > data/vitals/scan2_log.txt

Lê `data/vitals/RECORDS-numerics.txt` (índice do PhysioNet, baixado de
https://physionet.org/files/mimic3wdb/1.0/RECORDS-numerics) e imprime uma linha `OK <registro>
<amostras> <sinais>` para cada registro com FC, SpO2 e RESP e com pressão arterial (ABP, ART ou
NBP). Lê só o cabeçalho, sem baixar o sinal.
"""

from __future__ import annotations

import pathlib

import wfdb

INDICE = pathlib.Path("data/vitals/RECORDS-numerics.txt")
PA = {"ABP Sys", "ART Sys", "NBP Sys"}
BASICOS = {"HR", "SpO2", "RESP"}


def main() -> None:
    registros = [linha.strip() for linha in INDICE.read_text().splitlines() if linha.strip()]
    print("total registros", len(registros), flush=True)
    erros = achados = 0
    for i, registro in enumerate(registros):
        pasta, nome = registro.rsplit("/", 1)
        try:
            cabecalho = wfdb.rdheader(nome, pn_dir="mimic3wdb/1.0/" + pasta)
        except Exception:
            erros += 1
            continue
        sinais = set(cabecalho.sig_name)
        if BASICOS <= sinais and sinais & PA and cabecalho.sig_len and cabecalho.sig_len >= 360:
            achados += 1
            print("OK", registro, cabecalho.sig_len, sorted(sinais & (PA | BASICOS)), flush=True)
        if i % 200 == 0:
            print("progresso", i, "achados", achados, "erros", erros, flush=True)
    print("fim achados", achados, "erros", erros, flush=True)


if __name__ == "__main__":
    main()
