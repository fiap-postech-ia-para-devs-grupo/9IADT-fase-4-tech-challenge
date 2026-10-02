"""Linha do tempo de prescrições de uma internação, com os Alertas marcados."""

# pyright: reportAttributeAccessIssue=false
# (as stubs do pandas tipam as linhas de `itertuples()` como tuple)

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.analyzers.prescriptions import listas
from src.core.alerts import Alerta

COR_REGRA = {
    "salto_de_dose": "#d62728",
    "alta_vigilancia": "#ff7f0e",
    "trocas_excessivas": "#9467bd",
    "duplicidade": "#1f77b4",
}


def _monitorada(droga: str) -> bool:
    return listas.alta_vigilancia(droga) is not None or listas.classe_terapeutica(droga) is not None


def figura(prescricoes: pd.DataFrame, alertas: list[Alerta], hadm_id: int) -> go.Figure:
    """Barras de vigência das drogas monitoradas (alta vigilância ou de classe conhecida) mais
    as de qualquer Alerta, um marcador por Alerta na linha da sua droga."""
    da_internacao = prescricoes[prescricoes["hadm_id"] == hadm_id]
    do_paciente = [a for a in alertas if a.evidencia.get("hadm_id") == hadm_id]
    drogas_alertadas = {a.evidencia["droga"] for a in do_paciente if "droga" in a.evidencia}
    drogas_alertadas |= {d for a in do_paciente for d in a.evidencia.get("drogas", [])}
    mascara = da_internacao["drug"].map(_monitorada).astype(bool) | da_internacao["drug"].isin(
        list(drogas_alertadas)
    )
    relevantes = da_internacao.loc[mascara]
    ordem = list(relevantes.sort_values("starttime")["drug"].drop_duplicates())

    fig = go.Figure()
    for linha in relevantes.itertuples():
        duracao_ms = (linha.stoptime - linha.starttime).total_seconds() * 1000
        fig.add_trace(
            go.Bar(
                base=[linha.starttime],
                x=[duracao_ms],
                y=[linha.drug],
                orientation="h",
                marker={"color": "#c7d3df"},
                hovertemplate=f"{linha.drug}<br>dose {linha.dose} {linha.dose_unit_rx}"
                f"<br>{linha.route}<extra></extra>",
                showlegend=False,
            )
        )
    for regra, cor in COR_REGRA.items():
        da_regra = [a for a in do_paciente if a.evidencia["regra"] == regra]
        if not da_regra:
            continue
        fig.add_trace(
            go.Scatter(
                x=[a.detectado_em for a in da_regra],
                y=[a.evidencia.get("droga") for a in da_regra],
                mode="markers",
                name=regra,
                marker={"color": cor, "size": 13, "symbol": "diamond", "line": {"width": 1}},
                text=[a.descricao for a in da_regra],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Prescrições e Alertas, internação {hadm_id}",
        barmode="overlay",
        height=max(380, 28 * len(ordem) + 160),
        yaxis={"categoryorder": "array", "categoryarray": ordem[::-1]},
        xaxis={"type": "date"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig
