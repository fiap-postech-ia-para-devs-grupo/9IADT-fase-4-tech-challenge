"""Tela 1: Central de Alertas (ESTRATEGIA.md, seção 9).

Única interface pela qual a equipe médica vê e trata Alertas: lista ordenada por severidade e
horário, Evidência renderizada por tipo e Reconhecimento/Resolução gravados no SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core import db, mock
from src.core.alerts import Alerta, Origem, Severidade, Status, TransicaoInvalida

RAIZ = Path(__file__).resolve().parents[2]

ICONE_SEVERIDADE = {
    Severidade.CRITICA: "🔴",
    Severidade.ALTA: "🟠",
    Severidade.MEDIA: "🟡",
    Severidade.BAIXA: "🟢",
}


def tabela_alertas(alertas: list[Alerta]) -> pd.DataFrame:
    """Uma linha por Alerta, na ordem recebida (já ordenada por `db.listar`)."""
    return pd.DataFrame(
        [
            {
                "Severidade": f"{ICONE_SEVERIDADE[a.severidade]} {a.severidade}",
                "Horário": a.detectado_em.strftime("%d/%m %H:%M"),
                "Paciente": a.paciente_id,
                "Origem": str(a.origem),
                "Tipo": str(a.tipo),
                "Status": str(a.status),
                "Descrição": a.descricao,
            }
            for a in alertas
        ]
    )


def _caminho(relativo: str) -> Path:
    caminho = Path(relativo)
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _grafico_janela(evidencia: dict) -> go.Figure:
    valores = evidencia["valores"]
    t0, t1 = evidencia.get("janela", [0, len(valores) - 1])
    passo = (t1 - t0) / max(len(valores) - 1, 1)
    figura = go.Figure(
        go.Scatter(
            x=[t0 + i * passo for i in range(len(valores))],
            y=valores,
            mode="lines+markers",
            name=evidencia.get("serie", "série"),
        )
    )
    figura.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=f"{evidencia.get('serie', 'Sinal')} na janela do Alerta",
        xaxis_title="tempo simulado (s)",
    )
    return figura


def renderizar_evidencia(alerta: Alerta) -> None:
    """Evidência por origem: frame, trecho de áudio + transcrição ou gráfico da janela."""
    ev = alerta.evidencia
    match alerta.origem:
        case Origem.VIDEO:
            frame = _caminho(ev["frame"]) if "frame" in ev else None
            if frame is not None and frame.is_file():
                st.image(str(frame), caption=f"t = {ev.get('t', '?')} s")
            else:
                st.caption(f"Frame de Evidência indisponível: `{ev.get('frame', '—')}`")
            detalhes = {k: v for k, v in ev.items() if k != "frame"}
            st.json(detalhes, expanded=False)
        case Origem.AUDIO:
            arquivo = _caminho(ev["arquivo"]) if "arquivo" in ev else None
            if arquivo is not None and arquivo.is_file():
                st.audio(str(arquivo), start_time=int(ev.get("inicio", 0)))
            elif "arquivo" in ev:
                st.caption(f"Áudio indisponível: `{ev['arquivo']}`")
            if "trecho" in ev:
                st.markdown(f"> {ev['trecho']}")
            st.caption(f"início em {ev.get('inicio', '?')} s")
            extras = {k: v for k, v in ev.items() if k not in {"trecho", "arquivo", "inicio"}}
            if extras:
                st.json(extras, expanded=False)
        case Origem.SINAIS_VITAIS:
            if ev.get("valores"):
                st.plotly_chart(_grafico_janela(ev), width="stretch")
            st.json({k: v for k, v in ev.items() if k != "valores"}, expanded=False)
        case _:
            st.json(ev, expanded=False)


def _renderizar_origens(conn: sqlite3.Connection, composto: Alerta) -> None:
    st.markdown("**Alertas de origem da Fusão**")
    for origem_id in composto.alertas_origem:
        origem = db.obter(conn, origem_id)
        if origem is None:
            st.caption(f"Alerta {origem_id} não encontrado")
            continue
        titulo = f"{ICONE_SEVERIDADE[origem.severidade]} {origem.origem} · {origem.descricao}"
        with st.expander(titulo):
            renderizar_evidencia(origem)


def _acoes(conn: sqlite3.Connection, alerta: Alerta, responsavel: str) -> None:
    coluna_a, coluna_b = st.columns(2)
    reconhecer = coluna_a.button(
        "Reconhecer",
        disabled=alerta.status is not Status.NOVO,
        type="primary",
        key=f"reconhecer-{alerta.id}",
    )
    resolver = coluna_b.button(
        "Resolver",
        disabled=alerta.status is not Status.RECONHECIDO,
        key=f"resolver-{alerta.id}",
    )
    if not (reconhecer or resolver):
        return
    try:
        if reconhecer:
            if not responsavel.strip():
                st.error("Informe quem está reconhecendo (campo na barra lateral).")
                return
            db.reconhecer(conn, alerta.id, por=responsavel.strip())
        else:
            db.resolver(conn, alerta.id)
    except TransicaoInvalida as erro:
        st.error(str(erro))
        return
    st.rerun()


def _detalhe(conn: sqlite3.Connection, alerta: Alerta, responsavel: str) -> None:
    st.subheader(f"{ICONE_SEVERIDADE[alerta.severidade]} {alerta.descricao}")
    st.caption(
        f"{alerta.paciente_id} · {alerta.origem}/{alerta.tipo} · "
        f"{alerta.detectado_em:%d/%m/%Y %H:%M} · status: **{alerta.status}**"
    )
    if alerta.reconhecido_por and alerta.reconhecido_em:
        st.caption(
            f"Reconhecido por {alerta.reconhecido_por} em {alerta.reconhecido_em:%d/%m %H:%M}"
        )
    _acoes(conn, alerta, responsavel)
    st.markdown("**Evidência**")
    renderizar_evidencia(alerta)
    if alerta.origem is Origem.FUSAO:
        _renderizar_origens(conn, alerta)


def pagina(conn: sqlite3.Connection) -> None:
    st.title("Central de Alertas")

    if not db.listar(conn):
        mock.popular_mock(conn)
        st.info("Banco vazio: Alertas mock carregados (substituídos pelos reais na integração).")

    todos = db.listar(conn)
    with st.sidebar:
        st.header("Filtros")
        responsavel = st.text_input("Responsável", placeholder="ex.: enf. Ana", key="responsavel")
        origens = st.multiselect("Origem", [str(o) for o in Origem])
        status = st.multiselect("Status", [str(s) for s in Status])
        pacientes = st.multiselect("Paciente", sorted({a.paciente_id for a in todos}))

    alertas = db.listar(
        conn,
        origem=origens or None,
        status=status or None,
        paciente_id=pacientes or None,
    )
    novos = sum(a.status is Status.NOVO for a in alertas)
    st.caption(f"{len(alertas)} Alertas ({novos} novos)")
    if not alertas:
        st.warning("Nenhum Alerta para os filtros escolhidos.")
        return

    selecao = st.dataframe(
        tabela_alertas(alertas),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="tabela_alertas",
    )
    linhas = selecao.selection.rows
    if not linhas:
        st.caption("Selecione um Alerta para ver a Evidência.")
        return
    # relê do banco para refletir o status mais recente
    escolhido = db.obter(conn, alertas[linhas[0]].id)
    if escolhido is not None:
        st.divider()
        _detalhe(conn, escolhido, responsavel)
