"""Barra do sistema no topo de cada tela: marca, plantonista, Alertas pendentes e relógio.

Cada tela a chama no início do seu único fragmento: é aí que o relógio do plantão avança e os
Alertas chegam. Um fragmento de barra separado no `app.py`, tocando ao mesmo tempo que o da
tela, quebra a árvore de elementos do Streamlit na troca de página.
"""

from __future__ import annotations

import streamlit as st

from src.core import db
from src.core.alerts import Severidade, Status
from src.ui import estilo
from src.ui import plantao_estado as plantao
from src.ui.estilo import e


def barra(avancar: bool = True) -> None:
    if avancar:
        plantao.avancar()
    agora = plantao.agora()
    internados = set(plantao.internacoes())
    pendentes = [
        a
        for a in db.listar(plantao.conexao(), status=Status.NOVO)
        if a.paciente_id in internados and a.detectado_em <= agora
    ]
    criticos = sum(a.severidade is Severidade.CRITICA for a in pendentes)
    altos = sum(a.severidade is Severidade.ALTA for a in pendentes)
    plantonista = st.session_state.get("responsavel") or "plantonista não identificado"
    estado = "▶ em andamento" if plantao.tocando() else "⏸ pausado"
    chips = ""
    if criticos:
        chips += estilo.chip(f"{criticos} críticos", estilo.COR_SEVERIDADE[Severidade.CRITICA])
    if altos:
        chips += estilo.chip(f"{altos} altos", estilo.COR_SEVERIDADE[Severidade.ALTA])
    st.html(
        f"""
        <div class="barra-sistema">
          <span class="marca">🏥 Monitoramento Multimodal</span>
          <span class="item">Unidade de Internação · {len(internados)} leitos</span>
          <span class="item">👤 {e(plantonista)}</span>
          <span class="direita">
            {chips}
            <span class="item">{estado}</span>
            <span class="relogio">🕒 {agora:%d/%m %H:%M}</span>
          </span>
        </div>
        """
    )
