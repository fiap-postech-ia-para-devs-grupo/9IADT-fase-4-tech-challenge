"""Monitoramento Multimodal: app Streamlit com as 3 telas (ESTRATEGIA.md, seção 9).

uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.core import db
from src.ui import central, laboratorio, painel

st.set_page_config(page_title="Monitoramento Multimodal", page_icon="🏥", layout="wide")


@st.cache_resource
def _conexao():
    return db.conectar(check_same_thread=False)


def tela_central() -> None:
    central.pagina(_conexao())


navegacao = st.navigation(
    [
        st.Page(
            tela_central, title="Central de Alertas", icon="🚨", url_path="alertas", default=True
        ),
        st.Page(painel.pagina, title="Painel do Paciente", icon="🩺", url_path="paciente"),
        st.Page(
            laboratorio.pagina, title="Laboratório de Análise", icon="🔬", url_path="laboratorio"
        ),
    ]
)
navegacao.run()
