"""Estado compartilhado do plantão simulado: relógio único, dados em cache e chegada de Alertas.

Os três Pacientes de Demonstração começam a internação no mesmo instante, então um só relógio
(`relogio_min`, minutos desde o início do plantão) dirige todas as telas: o Mapa de Leitos, o
Prontuário e a Central mostram o hospital nesse instante. Enquanto o relógio anda, os Alertas
que já "aconteceram" são gravados no SQLite e chegam à Central, como num plantão real.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import streamlit as st

from src.core import db, demo
from src.core.demo import EventoAlerta, Internacao, PacienteDemo
from src.core.ficha import Ficha, carregar_fichas

SEGUNDOS_POR_QUADRO = 1.0
VELOCIDADES = {"1×": 1, "5×": 5, "15×": 15, "30×": 30}


def conexao() -> sqlite3.Connection:
    """Uma conexão por sessão: o SQLite não aceita a mesma conexão em threads concorrentes, e
    cada sessão do Streamlit (e cada fragmento com timer) roda na sua thread."""
    if "conexao_sqlite" not in st.session_state:
        st.session_state["conexao_sqlite"] = db.conectar(check_same_thread=False)
    return st.session_state["conexao_sqlite"]


@st.cache_data(ttl=300, show_spinner=False)
def pacientes() -> list[PacienteDemo]:
    return demo.carregar_pacientes()


@st.cache_data(ttl=300, show_spinner=False)
def fichas() -> dict[str, Ficha]:
    return carregar_fichas(pacientes())


@st.cache_resource(show_spinner="Montando a internação...")
def internacao(paciente_id: str) -> Internacao:
    por_id = {p.id: p for p in demo.carregar_pacientes()}
    return demo.montar(por_id[paciente_id])


def internacoes() -> dict[str, Internacao]:
    """Internações que têm os dados baixados; as outras ficam de fora do plantão."""
    montadas = {}
    for paciente in pacientes():
        try:
            montadas[paciente.id] = internacao(paciente.id)
        except demo.DadosAusentes:
            continue
    return montadas


def eventos(paciente_id: str) -> list[EventoAlerta]:
    """Alertas de todas as fontes do paciente, já com os Compostos da Fusão."""
    return demo.fundir_eventos(
        internacao(paciente_id).eventos, demo.alertas_de_video(conexao(), paciente_id)
    )


def inicio_plantao() -> datetime:
    return min((p.internacao_inicio for p in pacientes()), default=datetime(2026, 10, 1, 8))


def duracao_min() -> int:
    montadas = internacoes()
    if not montadas:
        return 0
    inicio = inicio_plantao()
    return max(int((i.fim - inicio).total_seconds() // 60) for i in montadas.values())


def _iniciar_estado() -> None:
    st.session_state.setdefault("relogio_min", 0)
    st.session_state.setdefault("relogio_tocando", False)
    st.session_state.setdefault("relogio_vel", "15×")


def minuto() -> int:
    _iniciar_estado()
    return int(st.session_state["relogio_min"])


def agora() -> datetime:
    return inicio_plantao() + timedelta(minutes=minuto())


def tocando() -> bool:
    _iniciar_estado()
    return bool(st.session_state["relogio_tocando"])


def intervalo() -> float | None:
    """`run_every` dos fragmentos que acompanham o relógio."""
    return SEGUNDOS_POR_QUADRO if tocando() else None


def avancar() -> None:
    """Um passo do relógio (chamado no início de cada fragmento) e a chegada dos Alertas."""
    _iniciar_estado()
    total = duracao_min()
    if st.session_state["relogio_tocando"]:
        passo = VELOCIDADES[st.session_state["relogio_vel"]]
        st.session_state["relogio_min"] = min(st.session_state["relogio_min"] + passo, total)
        if st.session_state["relogio_min"] >= total:
            st.session_state["relogio_tocando"] = False
            sincronizar()
            st.rerun()  # a tela inteira: a barra lateral volta a mostrar "Iniciar"
    sincronizar()


def sincronizar() -> int:
    """Grava os Alertas dos pacientes internados já surgidos até o relógio."""
    instante = agora()
    novos = 0
    for paciente_id in internacoes():
        visiveis = demo.visiveis(eventos(paciente_id), instante)
        novos += demo.gravar(conexao(), (e.alerta for e in visiveis))
    return novos


def controles() -> None:
    """Controles do relógio na barra lateral (mesmos em todas as telas)."""
    _iniciar_estado()
    total = duracao_min()
    st.markdown("**Relógio do plantão**")
    tocar, voltar = st.columns(2)
    if tocar.button(
        "⏸ Pausar" if tocando() else "▶ Iniciar",
        type="primary",
        key="relogio_tocar",
        width="stretch",
    ):
        if st.session_state["relogio_min"] >= total:
            st.session_state["relogio_min"] = 0
        st.session_state["relogio_tocando"] = not st.session_state["relogio_tocando"]
        st.rerun()
    if voltar.button("⟲ Reiniciar", key="relogio_reiniciar", width="stretch"):
        st.session_state.update(relogio_min=0, relogio_tocando=False)
        st.rerun()
    st.segmented_control(
        "Velocidade (min de plantão por segundo)",
        list(VELOCIDADES),
        key="relogio_vel",
        label_visibility="collapsed",
    )
    if not tocando() and total:
        # Chave própria: um widget com a chave `relogio_min` perderia o valor nas execuções em
        # que não é desenhado (relógio tocando).
        st.slider(
            "Ir para o minuto",
            0,
            total,
            value=minuto(),
            key="relogio_slider",
            format="%d min",
            on_change=lambda: st.session_state.update(
                relogio_min=st.session_state["relogio_slider"]
            ),
        )
