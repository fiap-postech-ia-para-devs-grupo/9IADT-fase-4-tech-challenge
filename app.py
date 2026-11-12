"""Monitoramento Multimodal: app Streamlit (ESTRATEGIA.md, seção 9).

uv run streamlit run app.py

Menu por seções, como num sistema hospitalar:
- Início: propósito, fluxo multimodal e tour guiado;
- Monitoramento: Mapa de Leitos e Central de Alertas (luz de chamada, WhatsApp);
- Paciente: Admissão e Prontuário (monitor, câmera do leito, evolução);
- Análise: Laboratório (vídeo e áudio sob demanda);
- Administração: Equipe e Leitos (WhatsApp da equipe, histórico de chamadas).
A barra superior, a luz de chamada e o relógio do plantão são os mesmos em todas as telas.
"""

from __future__ import annotations

import streamlit as st

from src.core import cadastro
from src.ui import (
    administracao,
    admissao,
    central,
    estilo,
    inicio,
    laboratorio,
    mapa,
    navegacao,
    prontuario,
    tour,
)
from src.ui import plantao_estado as plantao

st.set_page_config(
    page_title="Monitoramento Multimodal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
estilo.aplicar()


def tela_central() -> None:
    central.pagina(plantao.conexao())


paginas = {
    "": [st.Page(inicio.pagina, title="Início", icon="🏠", url_path="inicio", default=True)],
    "Monitoramento": [
        st.Page(mapa.pagina, title="Mapa de Leitos", icon="🛏️", url_path="leitos"),
        st.Page(tela_central, title="Central de Alertas", icon="🚨", url_path="alertas"),
    ],
    "Paciente": [
        st.Page(admissao.pagina, title="Admissão", icon="➕", url_path="admissao"),
        st.Page(prontuario.pagina, title="Prontuário", icon="🩺", url_path="prontuario"),
    ],
    "Análise": [
        st.Page(
            laboratorio.pagina, title="Laboratório de Análise", icon="🔬", url_path="laboratorio"
        ),
    ],
    "Administração": [
        st.Page(administracao.pagina, title="Equipe e Leitos", icon="👥", url_path="equipe"),
    ],
}
navegacao.PAGINAS.update({p.title: p for secao in paginas.values() for p in secao})
navegacao_app = st.navigation(paginas)


def _assumir_plantao() -> None:
    """Plantonista escolhido da equipe cadastrada: é quem reconhece e assina a evolução."""
    equipe = {p.id: p for p in cadastro.equipe(plantao.conexao())}
    escolhido = st.selectbox(
        "👤 Plantonista",
        [None, *equipe],
        format_func=lambda i: "Selecione…" if i is None else equipe[i].rotulo,
        key="plantonista_id",
    )
    st.session_state["responsavel"] = equipe[escolhido].rotulo if escolhido else ""


with st.sidebar:
    _assumir_plantao()
    plantao.controles()
    st.caption(
        "Pacientes compostos de datasets públicos, nomes fictícios (ADR 0001). "
        "Apoio à decisão; não substitui avaliação clínica."
    )


tour.mostrar_se_pedido()
navegacao_app.run()
