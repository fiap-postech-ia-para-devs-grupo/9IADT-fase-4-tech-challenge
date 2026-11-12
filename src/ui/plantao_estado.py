"""Estado compartilhado do plantão simulado: relógio único, dados em cache e chegada de Alertas.

Os pacientes internados vêm das admissões da unidade (`src/core/cadastro.py`). Um só relógio
(`relogio_min`, minutos desde o início do plantão) dirige todas as telas: o Mapa de Leitos, o
Prontuário e a Central mostram o hospital nesse instante. Enquanto o relógio anda, os Alertas
que já "aconteceram" são gravados no SQLite e chegam à Central, como num plantão real.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta

import streamlit as st

from src.core import cadastro, db, demo, news2, notificacao
from src.core.alerts import Alerta
from src.core.demo import EventoAlerta, Internacao, PacienteDemo
from src.core.ficha import Ficha

SEGUNDOS_POR_QUADRO = 1.0
VELOCIDADES = {"1×": 1, "5×": 5, "15×": 15, "30×": 30}
_ENVIANDO = threading.Lock()


def conexao() -> sqlite3.Connection:
    """Uma conexão por sessão: o SQLite não aceita a mesma conexão em threads concorrentes, e
    cada sessão do Streamlit (e cada fragmento com timer) roda na sua thread."""
    if "conexao_sqlite" not in st.session_state:
        conn = db.conectar(check_same_thread=False)
        cadastro.preparar(conn)
        st.session_state["conexao_sqlite"] = conn
    return st.session_state["conexao_sqlite"]


def admissoes() -> list[cadastro.Internacao]:
    return cadastro.internacoes(conexao())


def pacientes() -> list[PacienteDemo]:
    """Pacientes internados agora, no formato dos Analisadores (admissões da unidade)."""
    return [i.como_paciente() for i in admissoes()]


def fichas() -> dict[str, Ficha]:
    return {i.id: Ficha(i.id, i.nome, i.leito_id, i.idade, i.sexo, i.motivo) for i in admissoes()}


@st.cache_resource(show_spinner="Conectando as fontes de monitoramento do paciente...")
def _montar(paciente: PacienteDemo) -> Internacao:
    return demo.montar(paciente)


@st.cache_resource(show_spinner=False)
def _tendencias(paciente: PacienteDemo) -> list[EventoAlerta]:
    sinais = _montar(paciente).sinais
    return [
        EventoAlerta(a, a.detectado_em) for a in news2.alertas_de_tendencia(sinais, paciente.id)
    ]


def internacao(paciente_id: str) -> Internacao:
    por_id = {p.id: p for p in pacientes()}
    return _montar(por_id[paciente_id])


def internacoes() -> dict[str, Internacao]:
    """Internações que têm os dados baixados; as outras ficam de fora do plantão."""
    montadas = {}
    for paciente in pacientes():
        try:
            montadas[paciente.id] = _montar(paciente)
        except demo.DadosAusentes:
            continue
    return montadas


def eventos(paciente_id: str) -> list[EventoAlerta]:
    """Alertas de todas as fontes do paciente (com a piora do NEWS2) e os Compostos da Fusão."""
    paciente = next(p for p in pacientes() if p.id == paciente_id)
    return demo.fundir_eventos(
        [*_montar(paciente).eventos, *_tendencias(paciente)],
        demo.alertas_de_video(conexao(), paciente_id),
    )


def rotulos_pacientes() -> dict[str, str]:
    return {i.id: f"Leito {i.leito_id} · {i.nome}" for i in admissoes()}


def inicio_plantao() -> datetime:
    return min((p.internacao_inicio for p in pacientes()), default=datetime(2026, 10, 1, 8))


def duracao_min() -> int:
    montadas = internacoes()
    if not montadas:
        return 0
    inicio = inicio_plantao()
    return max(int((i.fim - inicio).total_seconds() // 60) for i in montadas.values())


def fim_plantao() -> datetime:
    return inicio_plantao() + timedelta(minutes=duracao_min())


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
    """Grava os Alertas surgidos até o relógio e aciona a equipe para os que pedem chamada."""
    instante = agora()
    novos = 0
    for paciente_id in internacoes():
        visiveis = demo.visiveis(eventos(paciente_id), instante)
        novos += demo.gravar(conexao(), (e.alerta for e in visiveis))
    acionar_equipe(instante)
    return novos


def chamadas(instante: datetime | None = None) -> list[Alerta]:
    """Luz de chamada acesa: Alertas críticos ou Quedas de internados, ainda sem Reconhecimento."""
    instante = instante or agora()
    internados = set(internacoes())
    return [
        a
        for a in db.listar(conexao(), status="novo", paciente_id=internados)
        if notificacao.exige_chamada(a) and a.detectado_em <= instante
    ]


def acionar_equipe(instante: datetime) -> None:
    """WhatsApp em segundo plano, com conexão própria, para não travar a tela no envio."""
    pendentes = chamadas(instante)
    if not pendentes or not st.session_state.get("whatsapp_ativo", True):
        return
    rotulos = rotulos_pacientes()

    def enviar() -> None:
        # Um envio por vez no processo: dois ticks seguidos não mandam a mesma chamada duas vezes.
        if not _ENVIANDO.acquire(blocking=False):
            return
        conn = db.conectar(db.CAMINHO_PADRAO)
        try:
            notificacao.acionar(conn, pendentes, instante, rotulos)
        finally:
            conn.close()
            _ENVIANDO.release()

    threading.Thread(target=enviar, daemon=True).start()


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
    inicio, fim = inicio_plantao(), fim_plantao()
    # A hora atual fica na barra superior (que se atualiza); aqui só o turno.
    st.caption(f"Plantão de hoje ({date.today():%d/%m}): **{inicio:%H:%M} → {fim:%H:%M}**")
    if not tocando() and total:
        # Chave própria: um widget com a chave `relogio_min` perderia o valor nas execuções em
        # que não é desenhado (relógio tocando).
        st.slider(
            "Ir para (min desde o início do turno)",
            0,
            total,
            value=minuto(),
            key="relogio_slider",
            format="%d min",
            on_change=lambda: st.session_state.update(
                relogio_min=st.session_state["relogio_slider"]
            ),
        )
