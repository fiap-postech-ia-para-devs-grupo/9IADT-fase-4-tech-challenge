"""Mapa de Leitos: visão do plantão (tela inicial).

Um cartão por leito monitorado, no instante do relógio do plantão: sinais atuais no estilo de
monitor, NEWS2 parcial, Alertas pendentes por severidade e o último evento. A cor da borda é a
pior situação do leito (Alerta crítico pendente ou NEWS2 alto). Clicar abre o Prontuário.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.analyzers.vitals.detectors import SERIES
from src.core import db, news2
from src.core.alerts import Alerta, Severidade, Status
from src.core.demo import Internacao
from src.core.ficha import ROTULO_ORIGEM, Ficha
from src.ui import cabecalho, estilo, navegacao
from src.ui import plantao_estado as plantao
from src.ui.estilo import COR_MONITOR, COR_RISCO, COR_SEVERIDADE, e
from src.ui.sinais import NOME_CURTO, UNIDADE, ultimos_sinais

ORDEM_SEVERIDADE = list(Severidade)[::-1]  # crítica primeiro


def alertas_pendentes(paciente_id: str, agora: datetime) -> list[Alerta]:
    return [
        a
        for a in db.listar(plantao.conexao(), paciente_id=paciente_id)
        if a.status is not Status.RESOLVIDO and a.detectado_em <= agora
    ]


def _cor_do_leito(pendentes: list[Alerta], escore: news2.Escore) -> str:
    novos = [a for a in pendentes if a.status is Status.NOVO]
    if any(a.severidade is Severidade.CRITICA for a in novos) or escore.risco == "alto":
        return COR_SEVERIDADE[Severidade.CRITICA]
    if any(a.severidade is Severidade.ALTA for a in novos) or escore.risco == "médio":
        return COR_SEVERIDADE[Severidade.ALTA]
    if novos or escore.risco == "baixo-médio":
        return COR_SEVERIDADE[Severidade.MEDIA]
    return COR_SEVERIDADE[Severidade.BAIXA]


def cartao_leito(ficha: Ficha | None, internacao: Internacao, agora: datetime) -> str:
    paciente = internacao.paciente
    instante = min(max(agora, internacao.inicio), internacao.fim)
    sinais = ultimos_sinais(internacao, instante)
    escore = news2.calcular(sinais)
    pendentes = alertas_pendentes(paciente.id, agora)

    blocos = "".join(
        estilo.bloco_sinal(
            NOME_CURTO[s],
            f"{sinais[s]:.0f}" if s in sinais and sinais[s] == sinais[s] else "—",
            UNIDADE[s],
            COR_MONITOR[escore.pontos.get(s, 0)],
        )
        for s in SERIES
    )
    contagem = "".join(
        estilo.chip(
            f"{sum(a.severidade is sev for a in pendentes)} {sev.value}", COR_SEVERIDADE[sev]
        )
        for sev in ORDEM_SEVERIDADE
        if any(a.severidade is sev for a in pendentes)
    ) or estilo.chip_claro("sem Alertas pendentes")
    ultimo = max(pendentes, key=lambda a: a.detectado_em, default=None)
    ultimo_html = (
        f'<div class="sub">Último: {ultimo.detectado_em:%H:%M} · {e(ROTULO_ORIGEM[ultimo.origem])}'
        f" · {e(ultimo.descricao[:90])}</div>"
        if ultimo
        else '<div class="sub">Nenhum evento até agora.</div>'
    )
    identificacao = ficha.identificacao if ficha else paciente.leito
    titulo = paciente.nome.split(":", 1)[-1].strip()
    return f"""
    <div class="leito" style="--cor:{_cor_do_leito(pendentes, escore)}">
      <div class="topo">
        <span class="nome">🛏️ {e(paciente.leito or paciente.id)}</span>
        {estilo.chip(f"NEWS2 {escore.total} · {escore.risco}", COR_RISCO[escore.risco])}
      </div>
      <div class="sub">{e(identificacao)} · {e(paciente.id)}</div>
      <div class="sub"><b>{e(titulo)}</b></div>
      <div class="sinais">{blocos}</div>
      <div>{contagem}</div>
      {ultimo_html}
    </div>
    """


def _feed(internados: list[str], agora: datetime, fichas: dict[str, Ficha]) -> None:
    eventos = sorted(
        (a for pid in internados for a in alertas_pendentes(pid, agora)),
        key=lambda a: a.detectado_em,
        reverse=True,
    )[:8]
    st.markdown('<div class="secao">Eventos recentes do plantão</div>', unsafe_allow_html=True)
    if not eventos:
        st.caption("Nenhum evento ainda.")
        return
    for alerta in eventos:
        leito = fichas[alerta.paciente_id].leito if alerta.paciente_id in fichas else ""
        st.html(
            f"""<div class="alerta-cartao" style="--cor:{COR_SEVERIDADE[alerta.severidade]}">
              <div class="linha1">{alerta.detectado_em:%H:%M} ·
                {estilo.chip_claro(leito)} {e(ROTULO_ORIGEM[alerta.origem])}</div>
              <div class="desc">{e(alerta.descricao[:120])}</div></div>"""
        )


def _quadro() -> None:
    cabecalho.barra()
    st.title("Mapa de Leitos")
    agora = plantao.agora()
    montadas = plantao.internacoes()
    fichas = plantao.fichas()
    if not montadas:
        st.error(
            "Dados dos pacientes ausentes: `uv run python scripts/download_data.py "
            "mimic3wdb mimic4demo video` e `uv run python scripts/run_batch.py`."
        )
        return
    if plantao.minuto() == 0 and not plantao.tocando():
        st.info("Plantão às 08:00. Clique em **▶ Iniciar** na barra lateral para acompanhar.")

    escores = {
        pid: news2.calcular(ultimos_sinais(i, min(max(agora, i.inicio), i.fim)))
        for pid, i in montadas.items()
    }
    pendentes = {pid: alertas_pendentes(pid, agora) for pid in montadas}
    novos = [a for lista in pendentes.values() for a in lista if a.status is Status.NOVO]
    colunas = st.columns(4)
    colunas[0].metric("Leitos monitorados", len(montadas))
    colunas[1].metric(
        "NEWS2 ≥ 5", sum(s.total >= 5 for s in escores.values()), help="Risco médio ou alto."
    )
    colunas[2].metric(
        "Alertas críticos sem Reconhecimento",
        sum(a.severidade is Severidade.CRITICA for a in novos),
    )
    colunas[3].metric("Alertas pendentes", len(novos))

    leitos, lateral = st.columns([3, 1.2], gap="large")
    with leitos:
        st.markdown('<div class="secao">Leitos</div>', unsafe_allow_html=True)
        grade = st.columns(min(3, len(montadas)))
        for coluna, (pid, internacao) in zip(grade, montadas.items(), strict=False):
            with coluna:
                st.html(cartao_leito(fichas.get(pid), internacao, agora))
                navegacao.link(
                    "Prontuário", "Abrir prontuário", icon="🩺", query_params={"leito": pid}
                )
    with lateral:
        _feed(list(montadas), agora, fichas)


def pagina() -> None:
    st.fragment(_quadro, run_every=plantao.intervalo())()
