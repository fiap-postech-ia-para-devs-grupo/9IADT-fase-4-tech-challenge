"""Páginas registradas pelo `app.py`, para uma tela levar a outra (leito → prontuário,
Alerta do paciente → Central). Sem o `app.py` (testes de uma tela isolada), o link some."""

from __future__ import annotations

from typing import Any

import streamlit as st

PAGINAS: dict[str, Any] = {}


def link(nome: str, rotulo: str, container: Any = st, **kw: Any) -> None:
    if nome in PAGINAS:
        container.page_link(PAGINAS[nome], label=rotulo, width="stretch", **kw)
