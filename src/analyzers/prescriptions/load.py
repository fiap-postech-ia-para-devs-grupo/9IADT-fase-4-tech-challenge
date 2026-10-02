"""Leitura do MIMIC-IV Clinical Database Demo (hosp/prescriptions, hosp/emar, admissions)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PASTA_PADRAO = Path(__file__).resolve().parents[3] / "data" / "prescriptions" / "mimic4demo"

_NUMERO = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Dados:
    prescricoes: pd.DataFrame
    emar: pd.DataFrame


def dose_numerica(valor: object) -> float | None:
    """Dose como número: "25,000" → 25000; faixa "5-10" → 10 (o teto da faixa prescrita)."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    numeros = _NUMERO.findall(str(valor).replace(",", ""))
    return max(float(n) for n in numeros) if numeros else None


def carregar(pasta: str | Path = PASTA_PADRAO) -> Dados:
    """Prescrições de terapia (drug_type MAIN) e administrações do eMAR, prontas para as regras.

    Prescrições sem `stoptime` terminam na alta da internação (`admissions.dischtime`).
    """
    hosp = Path(pasta) / "hosp"
    prescricoes = pd.read_csv(
        hosp / "prescriptions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "starttime",
            "stoptime",
            "drug_type",
            "drug",
            "dose_val_rx",
            "dose_unit_rx",
            "route",
        ],
        parse_dates=["starttime", "stoptime"],
    )
    admissoes = pd.read_csv(hosp / "admissions.csv.gz", usecols=["hadm_id", "dischtime"])
    admissoes["dischtime"] = pd.to_datetime(admissoes["dischtime"])

    prescricoes = prescricoes[prescricoes["drug_type"] == "MAIN"].copy()
    prescricoes = prescricoes.merge(admissoes, on="hadm_id", how="left")
    prescricoes["stoptime"] = prescricoes["stoptime"].fillna(prescricoes["dischtime"])
    prescricoes["dose"] = prescricoes["dose_val_rx"].map(dose_numerica)
    prescricoes = prescricoes.drop(columns=["dischtime", "dose_val_rx", "drug_type"])
    prescricoes = prescricoes.sort_values(["subject_id", "hadm_id", "starttime"], ignore_index=True)

    emar = pd.read_csv(
        hosp / "emar.csv.gz",
        usecols=["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
        parse_dates=["charttime"],
    )
    return Dados(prescricoes=prescricoes, emar=emar)
