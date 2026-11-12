"""Monitoramento Multimodal: app Streamlit (ESTRATEGIA.md, seção 9).

uv run streamlit run app.py

Menu por seções, como num sistema hospitalar: Monitoramento (Mapa de Leitos e Central de
Alertas), Paciente (Prontuário com monitor, tendências, linha do tempo e medicações) e Análise
(Laboratório). A barra superior e o relógio do plantão simulado são os mesmos em todas as telas.
"""

from __future__ import annotations

import streamlit as st

from src.ui import central, estilo, laboratorio, mapa, navegacao, prontuario
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
    "Monitoramento": [
        st.Page(mapa.pagina, title="Mapa de Leitos", icon="🛏️", url_path="leitos", default=True),
        st.Page(tela_central, title="Central de Alertas", icon="🚨", url_path="alertas"),
    ],
    "Paciente": [
        st.Page(prontuario.pagina, title="Prontuário", icon="🩺", url_path="prontuario"),
    ],
    "Análise": [
        st.Page(
            laboratorio.pagina, title="Laboratório de Análise", icon="🔬", url_path="laboratorio"
        ),
    ],
}
navegacao.PAGINAS.update({p.title: p for secao in paginas.values() for p in secao})
navegacao_app = st.navigation(paginas)

with st.sidebar:
    st.text_input("👤 Plantonista", placeholder="ex.: enf. Ana", key="responsavel")
    plantao.controles()
    st.caption(
        "Pacientes de Demonstração compostos de datasets públicos (ADR 0001). "
        "Apoio à decisão; não substitui avaliação clínica."
    )


navegacao_app.run()
