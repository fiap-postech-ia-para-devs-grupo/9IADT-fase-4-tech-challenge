"""Gráficos do Prontuário (ESTRATEGIA.md, seção 9): tendências dos Sinais Vitais com os
Alertas surgidos e o NEWS2 parcial, e a linha do tempo de prescrições, Consulta e vídeo."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analyzers.prescriptions import listas
from src.analyzers.vitals.detectors import REGRAS_CLINICAS, SERIES
from src.core import news2
from src.core.alerts import Origem
from src.core.demo import EventoAlerta, Internacao
from src.ui.sinais import NOME_SINAL, UNIDADE

COR_ORIGEM = {
    Origem.SINAIS_VITAIS: "#1f77b4",
    Origem.PRESCRICAO: "#ff7f0e",
    Origem.AUDIO: "#2ca02c",
    Origem.VIDEO: "#9467bd",
    Origem.FUSAO: "#d62728",
}
# Cores de traçado como num monitor: FC verde, FR amarela, SpO2 azul, PA vermelha.
COR_SINAL = {"FC": "#2e7d32", "FR": "#b28704", "SpO2": "#0277bd", "PAS": "#c62828"}
LINHA_SINAIS = "Sinais Vitais"
LINHA_CONSULTA = "Consulta"
LINHA_VIDEO = "Vídeo"


def figura_sinais(
    internacao: Internacao, eventos: list[EventoAlerta], agora: datetime
) -> go.Figure:
    """Uma linha por Sinal Vital até `agora`, com os Alertas já surgidos."""
    sinais = internacao.sinais.loc[: pd.Timestamp(agora)]
    titulos = [f"{NOME_SINAL[s]} ({UNIDADE[s]})" for s in SERIES] + ["NEWS2 parcial"]
    figura = make_subplots(
        rows=len(SERIES) + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=titulos,
        row_heights=[1] * len(SERIES) + [0.7],
    )
    for linha, serie in enumerate(SERIES, start=1):
        figura.add_trace(
            go.Scatter(
                x=sinais.index,
                y=sinais[serie],
                mode="lines",
                line={"color": COR_SINAL[serie], "width": 1.6},
                showlegend=False,
                hovertemplate=f"{serie} %{{y:.0f}}<extra></extra>",
            ),
            row=linha,
            col=1,
        )
        for nome, _, limite, _ in REGRAS_CLINICAS:
            if nome == serie:
                figura.add_hline(
                    y=limite,
                    line={"dash": "dot", "color": "#999", "width": 1},
                    row=linha,  # pyright: ignore[reportArgumentType]
                    col=1,  # pyright: ignore[reportArgumentType]
                )
        do_sinal = [
            e.alerta
            for e in eventos
            if e.alerta.origem is Origem.SINAIS_VITAIS
            and serie in e.alerta.evidencia["serie"].split(",")
        ]
        pontos = [(a, pd.Timestamp(a.detectado_em)) for a in do_sinal]
        pontos = [(a, t) for a, t in pontos if t in internacao.sinais.index]
        figura.add_trace(
            go.Scatter(
                x=[t for _, t in pontos],
                y=[internacao.sinais.at[t, serie] for _, t in pontos],
                mode="markers",
                marker={"color": "#d62728", "size": 10, "symbol": "x"},
                text=[a.descricao for a, _ in pontos],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=linha,
            col=1,
        )
    figura.add_trace(
        go.Scatter(
            x=sinais.index,
            y=news2.serie(sinais) if not sinais.empty else [],
            mode="lines",
            line={"color": "#8e44ad", "width": 2, "shape": "hv"},
            fill="tozeroy",
            showlegend=False,
            hovertemplate="NEWS2 %{y}<extra></extra>",
        ),
        row=len(SERIES) + 1,
        col=1,
    )
    for limite in (5, 7):  # risco médio e alto
        figura.add_hline(
            y=limite,
            line={"dash": "dot", "color": "#c0392b" if limite == 7 else "#e67e22", "width": 1},
            row=len(SERIES) + 1,  # pyright: ignore[reportArgumentType]
            col=1,  # pyright: ignore[reportArgumentType]
        )
    # Só o Alerta Composto vira marco vertical: os demais eventos estão na Linha do tempo, e uma
    # linha por Termo Crítico esconderia o traçado.
    for evento in eventos:
        if evento.alerta.origem is Origem.FUSAO:
            figura.add_vline(
                x=evento.aparece_em, line={"color": COR_ORIGEM[Origem.FUSAO], "width": 2.5}
            )
    figura.update_xaxes(
        range=[internacao.inicio, internacao.fim], showgrid=True, gridcolor="#eef2f6"
    )
    figura.update_yaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False)
    figura.update_annotations(font={"size": 12, "color": "#0b2545"}, xanchor="left", x=0)
    figura.update_layout(
        height=720,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
    )
    return figura


def linhas_de_prescricao(internacao: Internacao) -> pd.DataFrame:
    """Prescrições monitoradas (alta vigilância ou classe terapêutica) ativas na internação."""
    prescricoes = internacao.prescricoes
    ativas = (prescricoes["starttime"] <= pd.Timestamp(internacao.fim)) & (
        prescricoes["stoptime"] >= pd.Timestamp(internacao.inicio)
    )
    monitoradas = prescricoes["drug"].map(
        lambda droga: (
            listas.alta_vigilancia(droga) is not None
            or listas.classe_terapeutica(droga) is not None
        )
    )
    return prescricoes.loc[ativas & monitoradas.astype(bool)]


def _linha_do_alerta(evento: EventoAlerta) -> str:
    alerta = evento.alerta
    if alerta.origem is Origem.SINAIS_VITAIS:
        return LINHA_SINAIS
    if alerta.origem is Origem.AUDIO:
        return LINHA_CONSULTA
    if alerta.origem is Origem.VIDEO:
        return LINHA_VIDEO
    return str(alerta.evidencia.get("droga", LINHA_SINAIS))


def figura_linha_do_tempo(
    internacao: Internacao, eventos: list[EventoAlerta], agora: datetime
) -> go.Figure:
    """Prescrições, Consulta e vídeo da internação, com um marcador por Alerta já surgido."""
    paciente = internacao.paciente
    inicio, fim = internacao.inicio, internacao.fim
    barras: list[tuple[str, datetime, datetime]] = [
        (LINHA_SINAIS, inicio, fim),
        (LINHA_CONSULTA, paciente.em(paciente.consulta.inicio_min), internacao.consulta_fim),
    ]
    video_inicio = paciente.em(paciente.video.inicio_min)
    barras.append((LINHA_VIDEO, video_inicio, video_inicio + timedelta(minutes=2)))
    ativas = linhas_de_prescricao(internacao)
    barras += zip(
        ativas["drug"].astype(str),
        ativas["starttime"].clip(lower=pd.Timestamp(inicio)),
        ativas["stoptime"].clip(upper=pd.Timestamp(fim)),
        strict=True,
    )

    marcadores = [e for e in eventos if e.alerta.origem is not Origem.FUSAO]
    ordem = list(
        dict.fromkeys([nome for nome, _, _ in barras] + [_linha_do_alerta(e) for e in marcadores])
    )

    figura = go.Figure()
    for nome, comeco, termino in barras:
        figura.add_trace(
            go.Bar(
                base=[comeco],
                x=[(termino - comeco).total_seconds() * 1000],
                y=[nome],
                orientation="h",
                marker={"color": "#c7d3df"},
                hoverinfo="y",
                showlegend=False,
            )
        )
    for origem, cor in COR_ORIGEM.items():
        da_origem = [e for e in marcadores if e.alerta.origem is origem]
        if da_origem:
            figura.add_trace(
                go.Scatter(
                    x=[e.aparece_em for e in da_origem],
                    y=[_linha_do_alerta(e) for e in da_origem],
                    mode="markers",
                    name=str(origem),
                    marker={"color": cor, "size": 11, "symbol": "diamond"},
                    text=[e.alerta.descricao for e in da_origem],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    figura.add_vline(x=agora, line={"color": "#444", "width": 1})
    for evento in eventos:
        if evento.alerta.origem is Origem.FUSAO:
            figura.add_vline(
                x=evento.aparece_em, line={"color": COR_ORIGEM[Origem.FUSAO], "width": 3}
            )
    figura.update_layout(
        barmode="overlay",
        height=max(280, 30 * len(ordem) + 120),
        xaxis={"type": "date", "range": [inicio, fim]},
        yaxis={"categoryorder": "array", "categoryarray": ordem[::-1]},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h"},
    )
    return figura
