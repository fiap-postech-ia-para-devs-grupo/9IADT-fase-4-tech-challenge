"""Barra do sistema no topo de cada tela: marca, plantonista, sirene, Alertas e relógio.

Cada tela a chama no início do seu fragmento: é aí que o relógio do plantão avança e os
Alertas chegam. Um fragmento de barra separado no `app.py`, tocando ao mesmo tempo que o da
tela, quebra a árvore de elementos do Streamlit na troca de página.

A sirene fica **dentro** da barra, à direita (não flutua sobre a tela): vermelha piscando com
chamado aberto, âmbar com instabilidade alta sem Reconhecimento, apagada com tudo tratado. Cada
leito tem a sua luz no Mapa de Leitos e no Prontuário.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.core import db
from src.core.alerts import Severidade, Status
from src.core.ficha import ROTULO_SEVERIDADE
from src.ui import estilo
from src.ui import plantao_estado as plantao
from src.ui.estilo import e


def _sirene_html(chamadas: int, altos: int) -> str:
    if chamadas:
        estado, texto = "critico", f"{chamadas} chamado(s)"
    elif altos:
        estado, texto = "alto", f"{altos} alto(s)"
    else:
        estado, texto = "apagada", "sem chamados"
    return (
        f'<span class="sirene-barra {estado}" title="Sirene da unidade">'
        f'<span class="cupula"></span><span class="txt">{e(texto)}</span></span>'
    )


def barra(avancar: bool = True) -> None:
    if avancar:
        plantao.avancar()
    agora = plantao.agora()
    internados = set(plantao.internacoes())
    pendentes = [
        a
        for a in db.listar(plantao.conexao(), status=Status.NOVO, paciente_id=internados)
        if a.detectado_em <= agora
    ]
    chamadas = plantao.chamadas(agora)
    altos = sum(a.severidade is Severidade.ALTA for a in pendentes)
    plantonista = st.session_state.get("responsavel") or "plantonista não identificado"
    estado = "▶ em andamento" if plantao.tocando() else "⏸ pausado"
    fim = plantao.fim_plantao()
    barra_col, tour_col = st.columns([12, 1.3], vertical_alignment="center")
    barra_col.html(
        f"""
        <div class="barra-sistema">
          <span class="marca">🏥 Monitoramento Multimodal</span>
          <span class="item">Unidade de Internação · {len(internados)} leitos</span>
          <span class="item">👤 {e(plantonista)}</span>
          <span class="direita">
            <span class="item">Plantão de hoje ({date.today():%d/%m}) ·
              {plantao.inicio_plantao():%H:%M} → {fim:%H:%M} · {estado}</span>
            <span class="relogio" title="Hora do plantão (replay acelerado)">🕒 {agora:%H:%M}</span>
            {_sirene_html(len(chamadas), altos)}
          </span>
        </div>
        """
    )
    if tour_col.button("🧭 Tour", key="tour_cabecalho", width="stretch", help="Tour do sistema"):
        st.session_state.update(tour_aberto=True, tour_passo=0)
        st.rerun(scope="app")
    _chamados_abertos(chamadas)


def _chamados_abertos(chamadas: list) -> None:
    """Faixa de chamado de emergência, uma linha por leito (o Alerta mais grave de cada um),
    e um aviso por chamado novo."""
    if not chamadas:
        return
    rotulos = plantao.rotulos_pacientes()
    pesos = {s: i for i, s in enumerate(Severidade)}
    por_leito: dict[str, list] = {}
    for alerta in chamadas:
        por_leito.setdefault(alerta.paciente_id, []).append(alerta)
    linhas = []
    for paciente_id, alertas in por_leito.items():
        pior = max(alertas, key=lambda a: (pesos[a.severidade], a.detectado_em))
        extra = f" · +{len(alertas) - 1} alerta(s)" if len(alertas) > 1 else ""
        linhas.append(
            f"<b>{e(rotulos.get(paciente_id, paciente_id))}</b> · "
            f"{e(ROTULO_SEVERIDADE[pior.severidade])} · {e(pior.descricao[:80])} "
            f"({pior.detectado_em:%H:%M}){extra}"
        )
    st.html(
        f"""<div class="luz-chamada"><span class="lampada"></span>
        <div class="texto"><b>🚨 CHAMADA DE EMERGÊNCIA — {len(por_leito)} leito(s)</b><br>
        {"<br>".join(linhas)}<br><small>Equipe acionada no WhatsApp (um disparo por leito) ·
        reconheça na Central de Alertas</small></div></div>"""
    )
    for alerta in chamadas:
        chave = f"toast_{alerta.id}"
        if not st.session_state.get(chave):
            st.session_state[chave] = True
            st.toast(f"🚨 {rotulos.get(alerta.paciente_id, '')}: {alerta.descricao[:80]}")


# `estilo` é usado pelas telas que importam este módulo para montar chips.
__all__ = ["barra", "estilo"]
