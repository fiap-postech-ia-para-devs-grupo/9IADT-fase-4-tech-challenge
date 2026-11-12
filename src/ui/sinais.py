"""Leitura dos Sinais Vitais da internação no instante do relógio, para monitor e cartões."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.analyzers.vitals.detectors import SERIES
from src.core.demo import Internacao

UNIDADE = {"FC": "bpm", "FR": "irpm", "SpO2": "%", "PAS": "mmHg"}
NOME_SINAL = {
    "FC": "Frequência cardíaca",
    "FR": "Frequência respiratória",
    "SpO2": "Saturação (SpO2)",
    "PAS": "Pressão sistólica",
}
NOME_CURTO = {"FC": "FC", "FR": "FR", "SpO2": "SpO2", "PAS": "PAS"}


def ultimos_sinais(internacao: Internacao, agora: datetime) -> dict[str, float]:
    """Último valor medido de cada Sinal Vital até `agora`: uma lacuna não apaga o monitor."""
    ate_agora = internacao.sinais.loc[: pd.Timestamp(agora), list(SERIES)].ffill()
    return ate_agora.iloc[-1].to_dict() if not ate_agora.empty else {}
