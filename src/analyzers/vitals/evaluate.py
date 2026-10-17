"""Avaliação dos detectores contra as Anomalias injetadas (ESTRATEGIA.md, seção 6).

Por técnica, sobre o gabarito de `inject.aplicar`:

    recall           injeções com ao menos um Alerta cobrindo seu trecho / injeções
    precisao         Alertas que cobrem alguma injeção / Alertas emitidos
    atraso_mediano   minutos entre o início da injeção e o início do primeiro Alerta que a cobre

Um Alerta cobre uma injeção quando a série dele inclui a série afetada e os intervalos se
cruzam, com tolerância de `TOLERANCIA_MIN` minutos nas bordas.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.analyzers.vitals.inject import Injecao
from src.core.alerts import Alerta

TOLERANCIA_MIN = 5


def _cobre(alerta: Alerta, injecao: Injecao, inicio_janela: datetime) -> bool:
    if injecao.serie not in alerta.evidencia["serie"].split(","):
        return False
    a_ini, a_fim = alerta.evidencia["janela"]
    i_ini = (injecao.inicio - inicio_janela).total_seconds()
    i_fim = (injecao.fim - inicio_janela).total_seconds()
    folga = TOLERANCIA_MIN * 60
    return a_ini <= i_fim + folga and a_fim >= i_ini - folga


def avaliar(
    alertas_por_tecnica: dict[str, list[Alerta]],
    gabarito: list[Injecao],
    inicio_janela: datetime,
) -> pd.DataFrame:
    linhas = []
    for tecnica, alertas in alertas_por_tecnica.items():
        cobertos: set[int] = set()
        atrasos: list[float] = []
        alertas_uteis = 0

        for alerta in alertas:
            acertou = False
            for k, injecao in enumerate(gabarito):
                if _cobre(alerta, injecao, inicio_janela):
                    acertou = True
                    cobertos.add(k)
            alertas_uteis += acertou

        for k, injecao in enumerate(gabarito):
            if k not in cobertos:
                continue
            i_ini = (injecao.inicio - inicio_janela).total_seconds()
            primeiros = [
                a.evidencia["janela"][0] for a in alertas if _cobre(a, injecao, inicio_janela)
            ]
            atrasos.append(max(0.0, (min(primeiros) - i_ini) / 60))

        linhas.append(
            {
                "tecnica": tecnica,
                "injecoes": len(gabarito),
                "detectadas": len(cobertos),
                "recall": len(cobertos) / len(gabarito) if gabarito else np.nan,
                "precisao": alertas_uteis / len(alertas) if alertas else np.nan,
                "alertas": len(alertas),
                "atraso_mediano_min": float(np.median(atrasos)) if atrasos else np.nan,
            }
        )
    return pd.DataFrame(linhas)
