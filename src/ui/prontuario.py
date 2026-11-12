"""Prontuário do paciente (Tela 2 da ESTRATEGIA.md, seção 9): o leito no relógio do plantão.

Faixa de identificação (leito, idade, sexo, admissão), monitor de beira de leito com NEWS2
parcial e, abaixo, as seções Tendências, Linha do tempo, Medicações e Alertas do paciente. O
relógio é o mesmo do Mapa de Leitos e da Central; os Alertas surgem conforme ele anda.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.analyzers.vitals.detectors import SERIES
from src.core import db, demo, news2
from src.core.alerts import Origem, Severidade, Status, TransicaoInvalida
from src.core.demo import EventoAlerta, Internacao, PacienteDemo
from src.core.ficha import ROTULO_ORIGEM, ROTULO_SEVERIDADE, ROTULO_STATUS, ROTULO_TIPO, Ficha
from src.ui import cabecalho, estilo, navegacao
from src.ui import plantao_estado as plantao
from src.ui.estilo import COR_MONITOR, COR_MONITOR_RISCO, COR_RISCO, COR_SEVERIDADE, e
from src.ui.graficos import figura_linha_do_tempo, figura_sinais, linhas_de_prescricao
from src.ui.sinais import NOME_SINAL, UNIDADE, ultimos_sinais

MINUTOS_TENDENCIA = 15
SECOES = ("📈 Tendências", "🗓️ Linha do tempo", "💊 Medicações", "🚨 Alertas do paciente")


def faixa_identificacao(paciente: PacienteDemo, ficha: Ficha | None, escore: news2.Escore) -> str:
    dados = []
    if ficha is not None:
        dados = [
            f"{ficha.idade} anos" if ficha.idade else "",
            ficha.sexo or "",
            f"admissão: {ficha.admissao}" if ficha.admissao else "",
        ]
    dados.append(f"internado desde {paciente.internacao_inicio:%d/%m %H:%M}")
    detalhes = " · ".join(e(d) for d in dados if d)
    return f"""
    <div class="leito" style="--cor:{COR_RISCO[escore.risco]}">
      <div class="topo">
        <span class="nome">🛏️ {e(paciente.leito or paciente.id)} — {e(paciente.nome)}</span>
        {estilo.chip(f"NEWS2 {escore.total} · risco {escore.risco}", COR_RISCO[escore.risco])}
      </div>
      <div class="sub">{e(paciente.id)} · {detalhes} · paciente composto (ADR 0001)</div>
    </div>
    """


def monitor(internacao: Internacao, agora: datetime) -> str:
    """Blocos do monitor de beira de leito: valor atual, cor pelo NEWS2 e tendência."""
    atuais = ultimos_sinais(internacao, agora)
    antes = ultimos_sinais(internacao, agora - timedelta(minutes=MINUTOS_TENDENCIA))
    escore = news2.calcular(atuais)
    blocos = []
    for serie in SERIES:
        valor = atuais.get(serie)
        if valor is None or pd.isna(valor):
            blocos.append(estilo.bloco_sinal(NOME_SINAL[serie], "—", UNIDADE[serie], "#90a4ae"))
            continue
        anterior = antes.get(serie)
        tendencia = ""
        if anterior is not None and not pd.isna(anterior):
            delta = round(valor - anterior)
            seta = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            tendencia = f"{seta} {delta:+d} em {MINUTOS_TENDENCIA} min"
        blocos.append(
            estilo.bloco_sinal(
                NOME_SINAL[serie],
                f"{valor:.0f}",
                UNIDADE[serie],
                COR_MONITOR[escore.pontos.get(serie, 0)],
                tendencia,
            )
        )
    blocos.append(
        estilo.bloco_sinal(
            "NEWS2 parcial",
            str(escore.total),
            escore.risco,
            COR_MONITOR_RISCO[escore.risco],
            escore.resposta,
        )
    )
    return f'<div class="monitor">{"".join(blocos)}</div>'


def _avisos(surgidos: list[EventoAlerta]) -> None:
    for evento in surgidos:
        alerta = evento.alerta
        if alerta.origem is Origem.FUSAO:
            aviso = st.error if alerta.severidade is Severidade.CRITICA else st.warning
            aviso(
                f"**Alerta Composto {ROTULO_SEVERIDADE[alerta.severidade].lower()}** "
                f"({evento.aparece_em:%H:%M}): {alerta.descricao}",
                icon="🚨",
            )


def _medicacoes(internacao: Internacao, agora: datetime) -> None:
    prescricoes = internacao.prescricoes
    instante = pd.Timestamp(agora)
    ativas = prescricoes[
        (prescricoes["starttime"] <= instante) & (prescricoes["stoptime"] >= instante)
    ]
    if ativas.empty:
        st.caption("Nenhuma prescrição ativa neste momento.")
        return
    monitoradas = set(linhas_de_prescricao(internacao).index)
    tabela = pd.DataFrame(
        {
            "Vigilância": ["⚠️" if i in monitoradas else "" for i in ativas.index],
            "Medicamento": ativas["drug"].astype(str),
            "Dose": [
                f"{dose:g} {unidade}" if pd.notna(dose) else "—"
                for dose, unidade in zip(
                    ativas["dose"], ativas["dose_unit_rx"].fillna(""), strict=True
                )
            ],
            "Via": ativas["route"].fillna("—"),
            "Início": ativas["starttime"].dt.strftime("%d/%m %H:%M"),
        }
    ).sort_values("Vigilância", ascending=False)
    st.caption(
        f"{len(tabela)} prescrição(ões) ativa(s) às {agora:%H:%M}. "
        "⚠️ alta vigilância ou classe monitorada pelas regras de prescrição."
    )
    st.dataframe(tabela, hide_index=True, width="stretch", height=min(35 * len(tabela) + 40, 420))


def _alertas_do_paciente(paciente_id: str, agora: datetime) -> None:
    conn = plantao.conexao()
    alertas = [a for a in db.listar(conn, paciente_id=paciente_id) if a.detectado_em <= agora]
    if not alertas:
        st.caption("Nenhum Alerta até este momento.")
        return
    responsavel = (st.session_state.get("responsavel") or "").strip()
    for alerta in alertas:
        texto, acao = st.columns([5, 1])
        texto.html(
            f"""<div class="alerta-cartao" style="--cor:{COR_SEVERIDADE[alerta.severidade]}">
              <div class="linha1">{alerta.detectado_em:%H:%M} ·
                {
                estilo.chip(ROTULO_SEVERIDADE[alerta.severidade], COR_SEVERIDADE[alerta.severidade])
            }
                {estilo.chip_claro(ROTULO_ORIGEM[alerta.origem])}
                {estilo.chip_claro(ROTULO_TIPO[alerta.tipo])}
                {estilo.chip_claro(ROTULO_STATUS[alerta.status])}</div>
              <div class="desc">{e(alerta.descricao)}</div></div>"""
        )
        if alerta.status is Status.NOVO and acao.button(
            "✋ Reconhecer", key=f"pront_rec_{alerta.id}", width="stretch"
        ):
            if not responsavel:
                st.error("Informe o plantonista na barra lateral.")
            else:
                try:
                    db.reconhecer(conn, alerta.id, por=responsavel)
                except TransicaoInvalida as erro:
                    st.error(str(erro))
                st.rerun()
        navegacao.link("Central de Alertas", "Evidência", acao, query_params={"alerta": alerta.id})


def _quadro(paciente: PacienteDemo) -> None:
    try:
        internacao = plantao.internacao(paciente.id)
    except demo.DadosAusentes as erro:
        st.error(f"Dados ausentes. {erro}")
        return
    agora = min(max(plantao.agora(), internacao.inicio), internacao.fim)
    eventos = plantao.eventos(paciente.id)
    surgidos = demo.visiveis(eventos, agora)
    escore = news2.calcular(ultimos_sinais(internacao, agora))

    st.html(faixa_identificacao(paciente, plantao.fichas().get(paciente.id), escore))
    st.html(monitor(internacao, agora))
    _avisos(surgidos)

    secao = st.segmented_control(
        "Seção", SECOES, default=SECOES[0], key="prontuario_secao", label_visibility="collapsed"
    )
    if secao == SECOES[1]:
        st.plotly_chart(
            figura_linha_do_tempo(internacao, surgidos, agora), width="stretch", key="linha"
        )
    elif secao == SECOES[2]:
        _medicacoes(internacao, agora)
    elif secao == SECOES[3]:
        _alertas_do_paciente(paciente.id, agora)
    else:
        st.plotly_chart(figura_sinais(internacao, surgidos, agora), width="stretch", key="sinais")


def _tela() -> None:
    cabecalho.barra()
    pacientes = {p.id: p for p in plantao.pacientes()}
    pedido = st.query_params.get("leito")
    if pedido in pacientes:  # veio do Mapa de Leitos
        st.session_state["paciente_aberto"] = pedido
        del st.query_params["leito"]
    st.session_state.setdefault("paciente_aberto", next(iter(pacientes)))
    titulo, seletor = st.columns([2, 3], vertical_alignment="bottom")
    titulo.title("Prontuário")
    escolhido = seletor.segmented_control(
        "Leito",
        list(pacientes),
        format_func=lambda i: f"🛏️ {pacientes[i].leito or i}",
        key="paciente_aberto",
        label_visibility="collapsed",
    )
    _quadro(pacientes[escolhido or next(iter(pacientes))])


def pagina() -> None:
    st.fragment(_tela, run_every=plantao.intervalo())()
