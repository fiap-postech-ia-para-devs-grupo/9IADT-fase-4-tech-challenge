"""Tela 2: Painel do Paciente (ESTRATEGIA.md, seção 9).

Replay ao vivo dos Sinais Vitais de um Paciente de Demonstração, com os Alertas surgindo sobre o
gráfico e gravados na Central de Alertas, e linha do tempo de prescrições, Consulta e vídeo.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analyzers.prescriptions import listas
from src.analyzers.vitals.detectors import REGRAS_CLINICAS, SERIES
from src.core import db, demo, news2
from src.core.alerts import Origem, Severidade
from src.core.demo import EventoAlerta, Internacao, PacienteDemo
from src.core.ficha import ROTULO_ORIGEM, ROTULO_SEVERIDADE, Ficha, carregar_fichas
from src.ui.central import ICONE_SEVERIDADE

COR_ORIGEM = {
    Origem.SINAIS_VITAIS: "#1f77b4",
    Origem.PRESCRICAO: "#ff7f0e",
    Origem.AUDIO: "#2ca02c",
    Origem.VIDEO: "#9467bd",
    Origem.FUSAO: "#d62728",
}
LINHA_SINAIS = "Sinais Vitais"
LINHA_CONSULTA = "Consulta"
LINHA_VIDEO = "Vídeo"
SEGUNDOS_POR_QUADRO = 1.0
UNIDADE = {"FC": "bpm", "FR": "irpm", "SpO2": "%", "PAS": "mmHg"}
NOME_SINAL = {
    "FC": "Frequência cardíaca",
    "FR": "Frequência respiratória",
    "SpO2": "Saturação (SpO2)",
    "PAS": "Pressão sistólica",
}
COR_RISCO = {"baixo": "🟢", "baixo-médio": "🟡", "médio": "🟠", "alto": "🔴"}
COR_PONTOS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
MINUTOS_TENDENCIA = 15


@st.cache_resource
def _conexao() -> sqlite3.Connection:
    return db.conectar(check_same_thread=False)


@st.cache_resource(show_spinner="Montando a internação...")
def _internacao(paciente_id: str) -> Internacao:
    pacientes = {paciente.id: paciente for paciente in demo.carregar_pacientes()}
    return demo.montar(pacientes[paciente_id])


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
                line={"color": COR_ORIGEM[Origem.SINAIS_VITAIS], "width": 1.5},
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
    for evento in eventos:
        if evento.alerta.origem is Origem.SINAIS_VITAIS:
            continue
        figura.add_vline(
            x=evento.aparece_em,
            line={
                "color": COR_ORIGEM[evento.alerta.origem],
                "width": 3 if evento.alerta.origem is Origem.FUSAO else 1,
                "dash": "solid" if evento.alerta.origem is Origem.FUSAO else "dash",
            },
        )
    figura.update_xaxes(range=[internacao.inicio, internacao.fim])
    figura.update_layout(height=760, margin={"l": 10, "r": 10, "t": 30, "b": 10})
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


def _mostrar_alertas(eventos: list[EventoAlerta]) -> None:
    for evento in eventos:
        if evento.alerta.origem is Origem.FUSAO:
            aviso = st.error if evento.alerta.severidade is Severidade.CRITICA else st.warning
            aviso(
                f"🚨 Alerta Composto {ROTULO_SEVERIDADE[evento.alerta.severidade].lower()}: "
                f"{evento.alerta.descricao} ({evento.aparece_em:%H:%M})"
            )
    st.subheader(f"Alertas surgidos ({len(eventos)})")
    if not eventos:
        st.caption("Nenhum Alerta até este momento.")
    else:
        st.caption("Reconhecimento e Resolução na Central de Alertas.")
    recentes = sorted(eventos, key=lambda e: e.aparece_em, reverse=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Horário": e.aparece_em.strftime("%H:%M"),
                    "Severidade": f"{ICONE_SEVERIDADE[e.alerta.severidade]} "
                    f"{ROTULO_SEVERIDADE[e.alerta.severidade]}",
                    "Origem": ROTULO_ORIGEM[e.alerta.origem],
                    "Descrição": e.alerta.descricao,
                }
                for e in recentes
            ],
            columns=["Horário", "Severidade", "Origem", "Descrição"],
        ),
        hide_index=True,
        width="stretch",
        height=240,
    )


def ultimos_sinais(internacao: Internacao, agora: datetime) -> dict[str, float]:
    """Último valor medido de cada Sinal Vital até `agora`: uma lacuna não apaga o monitor."""
    ate_agora = internacao.sinais.loc[: pd.Timestamp(agora), list(SERIES)].ffill()
    return ate_agora.iloc[-1].to_dict() if not ate_agora.empty else {}


def _monitor(internacao: Internacao, agora: datetime) -> None:
    """Monitor de beira de leito: sinais atuais, tendência e NEWS2 parcial."""
    atuais = ultimos_sinais(internacao, agora)
    antes = ultimos_sinais(internacao, agora - timedelta(minutes=MINUTOS_TENDENCIA))
    escore = news2.calcular(atuais)

    colunas = st.columns(5)
    for coluna, serie in zip(colunas, SERIES, strict=False):
        valor = atuais.get(serie)
        if valor is None or pd.isna(valor):
            coluna.metric(NOME_SINAL[serie], "—")
            continue
        anterior = antes.get(serie)
        delta = None if anterior is None or pd.isna(anterior) else round(valor - anterior)
        coluna.metric(
            f"{COR_PONTOS[escore.pontos.get(serie, 0)]} {NOME_SINAL[serie]}",
            f"{valor:.0f} {UNIDADE[serie]}",
            delta=f"{delta:+d} em {MINUTOS_TENDENCIA} min" if delta else None,
            delta_color="off",
        )
    colunas[4].metric(
        f"{COR_RISCO[escore.risco]} NEWS2 parcial",
        f"{escore.total} · {escore.risco}",
        help=(
            "Escore de alerta precoce (Royal College of Physicians, 2017) com FR, SpO2, PAS e "
            f"FC. Sem {', '.join(news2.AUSENTES)}: é um piso do NEWS2 real."
        ),
    )
    if escore.risco != "baixo":
        st.caption(f"Conduta sugerida pelo NEWS2 ({escore.risco}): {escore.resposta}.")


def _medicacoes(internacao: Internacao, agora: datetime) -> None:
    prescricoes = internacao.prescricoes
    instante = pd.Timestamp(agora)
    ativas = prescricoes[
        (prescricoes["starttime"] <= instante) & (prescricoes["stoptime"] >= instante)
    ]
    if ativas.empty:
        st.caption("Nenhuma prescrição ativa neste momento.")
        return
    monitoradas = set(linhas_de_prescricao(internacao).index)
    tabela = pd.DataFrame(
        {
            "Vigilância": ["⚠️" if i in monitoradas else "" for i in ativas.index],
            "Medicamento": ativas["drug"].astype(str),
            "Dose": [
                f"{dose:g} {unidade}" if pd.notna(dose) else "—"
                for dose, unidade in zip(
                    ativas["dose"], ativas["dose_unit_rx"].fillna(""), strict=True
                )
            ],
            "Via": ativas["route"].fillna("—"),
            "Desde": ativas["starttime"].dt.strftime("%d/%m %H:%M"),
        }
    ).sort_values("Vigilância", ascending=False)
    st.caption("⚠️ alta vigilância ou classe monitorada pelas regras de prescrição")
    st.dataframe(tabela, hide_index=True, width="stretch", height=min(35 * len(tabela) + 40, 280))


def _ficha(paciente: PacienteDemo, ficha: Ficha | None) -> None:
    with st.container(border=True):
        cabecalho, info = st.columns([2, 3])
        cabecalho.markdown(f"### 🛏️ {paciente.leito or paciente.id}")
        cabecalho.caption(f"{paciente.id} · paciente composto (ADR 0001)")
        if ficha is not None:
            dados = [
                f"**Idade:** {ficha.idade} anos" if ficha.idade else None,
                f"**Sexo:** {ficha.sexo}" if ficha.sexo else None,
                f"**Admissão:** {ficha.admissao}" if ficha.admissao else None,
                f"**Internação:** desde {paciente.internacao_inicio:%d/%m %H:%M}",
            ]
            info.markdown("  \n".join(d for d in dados if d))
        info.caption(paciente.resumo)


def _quadro(internacao: Internacao, eventos: list[EventoAlerta], velocidade: int) -> None:
    """Um quadro do replay; quando o replay está tocando, avança `velocidade` minutos."""
    total = len(internacao.sinais) - 1
    if st.session_state["replay_tocando"]:
        st.session_state["replay_minuto"] = min(
            st.session_state["replay_minuto"] + velocidade, total
        )

    minuto = st.session_state["replay_minuto"]
    agora = internacao.inicio + timedelta(minutes=minuto)
    surgidos = demo.visiveis(eventos, agora)
    demo.gravar(_conexao(), (e.alerta for e in surgidos))

    st.progress(minuto / total, text=f"{agora:%H:%M} (minuto {minuto} de {total})")
    _monitor(internacao, agora)
    _mostrar_alertas(surgidos)
    st.subheader("Sinais Vitais")
    st.plotly_chart(
        figura_sinais(internacao, surgidos, agora), width="stretch", key="grafico_sinais"
    )
    st.subheader("Linha do tempo")
    st.plotly_chart(
        figura_linha_do_tempo(internacao, surgidos, agora), width="stretch", key="linha_do_tempo"
    )
    st.subheader("Medicações em uso")
    _medicacoes(internacao, agora)

    if st.session_state["replay_tocando"] and minuto >= total:
        st.session_state["replay_tocando"] = False
        st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def _fichas() -> dict[str, Ficha]:
    return carregar_fichas(demo.carregar_pacientes())


def pagina() -> None:
    st.title("Painel do Paciente")
    st.caption(
        "Pacientes fictícios, compostos de datasets públicos distintos; os horários foram "
        "alinhados para a demonstração (ADR 0001)."
    )

    pacientes = demo.carregar_pacientes()
    paciente = st.selectbox(
        "Paciente de Demonstração",
        pacientes,
        format_func=lambda p: f"{p.leito} · {p.nome}" if p.leito else p.nome,
    )
    _ficha(paciente, _fichas().get(paciente.id))
    if paciente.video.clipe is None:
        st.caption("Paciente sem clipe de vídeo: nenhum Alerta de vídeo nesta internação.")

    if st.session_state.get("replay_paciente") != paciente.id:
        st.session_state.update(replay_paciente=paciente.id, replay_minuto=0, replay_tocando=False)

    try:
        internacao = _internacao(paciente.id)
    except demo.DadosAusentes as erro:
        st.error(f"Dados ausentes. {erro}")
        return

    tocar, reiniciar, velocidade = st.columns([1, 1, 3])
    rotulo = "Pausar" if st.session_state["replay_tocando"] else "Tocar replay"
    if tocar.button(rotulo, type="primary", key="replay_tocar"):
        st.session_state["replay_tocando"] = not st.session_state["replay_tocando"]
        st.rerun()
    if reiniciar.button("Reiniciar", key="replay_reiniciar"):
        st.session_state.update(replay_minuto=0, replay_tocando=False)
        st.rerun()
    minutos_por_quadro = velocidade.slider(
        "Minutos de dado por segundo", 1, 30, 5, key="replay_vel"
    )

    eventos = demo.fundir_eventos(
        internacao.eventos, demo.alertas_de_video(_conexao(), paciente.id)
    )
    intervalo = SEGUNDOS_POR_QUADRO if st.session_state["replay_tocando"] else None
    st.fragment(_quadro, run_every=intervalo)(internacao, eventos, minutos_por_quadro)
