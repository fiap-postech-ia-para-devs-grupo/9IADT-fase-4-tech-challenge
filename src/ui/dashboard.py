"""Painel do plantão (Início): indicadores e gráficos da unidade no instante do relógio.

Ocupação dos leitos, chamados abertos, Alertas por severidade, origem e tipo, Alertas ao longo
do turno, NEWS2 de cada leito, tempo até o Reconhecimento e a equipe de plantão.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core import cadastro, db, news2, notificacao
from src.core.alerts import Alerta, Severidade, Status
from src.core.ficha import ROTULO_ORIGEM, ROTULO_SEVERIDADE, ROTULO_TIPO
from src.ui import plantao_estado as plantao
from src.ui.estilo import COR_RISCO, COR_SEVERIDADE
from src.ui.sinais import ultimos_sinais

ORDEM = [Severidade.CRITICA, Severidade.ALTA, Severidade.MEDIA, Severidade.BAIXA]
_LAYOUT = {
    "height": 280,
    "margin": {"l": 10, "r": 10, "t": 36, "b": 10},
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "font": {"size": 12, "color": "#1b2b3a"},
    "legend": {"orientation": "h", "y": -0.2},
}


def alertas_do_plantao(agora: datetime) -> list[Alerta]:
    internados = set(plantao.internacoes())
    return [
        a
        for a in db.listar(plantao.conexao(), paciente_id=internados)
        if plantao.inicio_plantao() <= a.detectado_em <= agora
    ]


def _minutos_ate_reconhecimento(alertas: list[Alerta]) -> float | None:
    tempos = [
        (a.reconhecido_em - a.detectado_em).total_seconds() / 60
        for a in alertas
        if a.reconhecido_em is not None and a.reconhecido_em >= a.detectado_em
    ]
    return sum(tempos) / len(tempos) if tempos else None


def _indicadores(alertas: list[Alerta], agora: datetime) -> None:
    conn = plantao.conexao()
    leitos = cadastro.leitos(conn)
    ocupados = len(cadastro.internacoes(conn))
    escores = {
        pid: news2.calcular(ultimos_sinais(i, min(max(agora, i.inicio), i.fim)))
        for pid, i in plantao.internacoes().items()
    }
    pendentes = [a for a in alertas if a.status is Status.NOVO]
    tempo = _minutos_ate_reconhecimento(alertas)
    graves = sum(e.total >= 5 for e in escores.values())
    chamados = len(plantao.chamadas(agora))
    cartoes = [
        (
            "🛏️",
            "Ocupação",
            f"{ocupados}/{len(leitos)}",
            f"{ocupados / len(leitos):.0%} dos leitos" if leitos else "",
            "#0b6e99",
        ),
        (
            "🚨",
            "Chamados abertos",
            str(chamados),
            "luz de chamada acesa",
            "#d32f2f" if chamados else "#2e7d32",
        ),
        ("🔔", "Alertas no plantão", str(len(alertas)), f"{len(pendentes)} pendentes", "#ef6c00"),
        (
            "📈",
            "NEWS2 ≥ 5",
            f"{graves}/{len(escores)}",
            "leitos em risco médio ou alto",
            "#d32f2f" if graves else "#2e7d32",
        ),
        (
            "⏱️",
            "Até o Reconhecimento",
            f"{tempo:.0f} min" if tempo is not None else "—",
            "média no relógio do plantão",
            "#0b2545",
        ),
    ]
    cartoes += [
        (
            "",
            f"{ROTULO_SEVERIDADE[sev]} pendentes",
            str(sum(a.severidade is sev for a in pendentes)),
            "sem Reconhecimento",
            COR_SEVERIDADE[sev],
        )
        for sev in ORDEM
    ]
    st.html(
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi" style="--cor:{cor}"><div class="kpi-rot">{icone} {rotulo}</div>'
            f'<div class="kpi-val">{valor}</div><div class="kpi-sub">{sub}</div></div>'
            for icone, rotulo, valor, sub, cor in cartoes
        )
        + "</div>"
    )


def _grafico_turno(alertas: list[Alerta], agora: datetime) -> go.Figure:
    inicio = plantao.inicio_plantao()
    figura = go.Figure()
    for severidade in ORDEM[::-1]:
        horas = [
            pd.Timestamp(a.detectado_em).floor("30min")
            for a in alertas
            if a.severidade is severidade
        ]
        contagem = Counter(horas)
        figura.add_bar(
            x=list(contagem),
            y=list(contagem.values()),
            name=ROTULO_SEVERIDADE[severidade],
            marker_color=COR_SEVERIDADE[severidade],
        )
    figura.update_layout(
        **_LAYOUT,
        barmode="stack",
        title="Alertas ao longo do turno (a cada 30 min)",
        xaxis={"range": [inicio, plantao.fim_plantao()], "gridcolor": "#eef2f6"},
        yaxis={"gridcolor": "#eef2f6"},
    )
    figura.add_vline(x=agora, line={"color": "#0b2545", "dash": "dot"})
    return figura


def _grafico_origens(alertas: list[Alerta]) -> go.Figure:
    contagem = Counter(ROTULO_ORIGEM[a.origem] for a in alertas)
    figura = go.Figure(
        go.Pie(
            labels=list(contagem),
            values=list(contagem.values()),
            hole=0.55,
            marker={"colors": ["#0b6e99", "#ef6c00", "#2e7d32", "#8e24aa", "#d32f2f"]},
            textinfo="label+value",
        )
    )
    figura.update_layout(**{**_LAYOUT, "legend": {"visible": False}}, title="Por origem")
    return figura


def _grafico_tipos(alertas: list[Alerta]) -> go.Figure:
    contagem = Counter(ROTULO_TIPO[a.tipo] for a in alertas).most_common()
    figura = go.Figure(
        go.Bar(
            x=[n for _, n in contagem][::-1],
            y=[t for t, _ in contagem][::-1],
            orientation="h",
            marker_color="#0b6e99",
            text=[n for _, n in contagem][::-1],
            textposition="outside",
        )
    )
    figura.update_layout(**_LAYOUT, title="Por tipo de Alerta", xaxis={"visible": False})
    return figura


def _grafico_news2(agora: datetime) -> go.Figure:
    leitos, totais, cores = [], [], []
    for pid, internacao in plantao.internacoes().items():
        escore = news2.calcular(
            ultimos_sinais(internacao, min(max(agora, internacao.inicio), internacao.fim))
        )
        leitos.append(internacao.paciente.leito or pid)
        totais.append(escore.total)
        cores.append(COR_RISCO[escore.risco])
    figura = go.Figure(
        go.Bar(x=leitos, y=totais, marker_color=cores, text=totais, textposition="outside")
    )
    figura.add_hline(y=5, line={"dash": "dot", "color": "#ef6c00"})
    figura.add_hline(y=7, line={"dash": "dot", "color": "#d32f2f"})
    figura.update_layout(
        **_LAYOUT, title="NEWS2 parcial por leito (agora)", yaxis={"range": [0, 10]}
    )
    return figura


def _equipe() -> None:
    conn = plantao.conexao()
    plantonista = st.session_state.get("responsavel") or "não identificado"
    cfg = notificacao.config()
    recebem = [p.rotulo for p in cadastro.equipe(conn) if p.recebe_alertas and p.telefone]
    with st.container(border=True):
        st.markdown(f"**👤 Plantonista:** {plantonista}")
        st.markdown(
            f"**🕒 Turno:** hoje ({date.today():%d/%m}), {plantao.inicio_plantao():%H:%M} → "
            f"{plantao.fim_plantao():%H:%M} · agora {plantao.agora():%H:%M}"
        )
        st.markdown(
            "**📲 Recebem chamadas no WhatsApp:** " + (", ".join(recebem) or "ninguém cadastrado")
        )
        st.caption(
            f"WhatsApp {'conectado (instância ' + cfg.instancia + ')' if cfg.ativo else 'simulado'}"
            " · equipe e números em Administração › Equipe e Leitos"
        )


def painel() -> None:
    agora = plantao.agora()
    if not plantao.internacoes():
        st.info("Nenhum paciente internado: comece pela Admissão.")
        return
    alertas = alertas_do_plantao(agora)
    esquerda, direita = st.columns([3, 1.2], gap="large")
    with esquerda:
        _indicadores(alertas, agora)
    with direita:
        _equipe()
    turno, origem = st.columns([2, 1])
    turno.plotly_chart(_grafico_turno(alertas, agora), width="stretch", key="dash_turno")
    origem.plotly_chart(_grafico_origens(alertas), width="stretch", key="dash_origem")
    tipos, leitos = st.columns(2)
    tipos.plotly_chart(_grafico_tipos(alertas), width="stretch", key="dash_tipos")
    leitos.plotly_chart(_grafico_news2(agora), width="stretch", key="dash_news2")
