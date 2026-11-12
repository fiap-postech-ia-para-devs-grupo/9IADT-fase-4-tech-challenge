"""Tela 1: Central de Alertas (ESTRATEGIA.md, seção 9).

Única interface pela qual a equipe médica vê e trata Alertas. Abre nos pacientes internados com
Alertas pendentes, mostra os indicadores do plantão, a fila ordenada por severidade e horário e,
ao lado, o Alerta aberto com a Evidência em linguagem clínica e o Reconhecimento/Resolução
gravados no SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analyzers.video.exercise_rules import DESCRICAO_REGRA as REGRAS_VIDEO
from src.core import db, demo, mock
from src.core.alerts import Alerta, Origem, Severidade, Status, TransicaoInvalida
from src.core.ficha import (
    ROTULO_ORIGEM,
    ROTULO_SEVERIDADE,
    ROTULO_STATUS,
    ROTULO_TIPO,
    Ficha,
    carregar_fichas,
    escopo,
    rotulo_paciente,
)

RAIZ = Path(__file__).resolve().parents[2]

ICONE_SEVERIDADE = {
    Severidade.CRITICA: "🔴",
    Severidade.ALTA: "🟠",
    Severidade.MEDIA: "🟡",
    Severidade.BAIXA: "🟢",
}
ESCOPOS = {
    "internados": "Pacientes internados",
    "laboratorio": "Laboratório (sem leito)",
    "avaliacao": "Registros de avaliação (MIMIC-III)",
    "todos": "Todos",
}
SEGUNDOS_ATUALIZACAO = 5

# Rótulos das chaves de Evidência; o que não está aqui aparece com o nome da chave.
_CAMPOS: dict[str, str] = {
    "t": "Instante no vídeo (s)",
    "angulo": "Ângulo medido (°)",
    "limite": "Limite",
    "regra": "Regra",
    "exercicio": "Exercício",
    "repeticao": "Repetição / episódio",
    "lado": "Lado",
    "transicao_s": "Transição em pé → chão (s)",
    "razao_caixa": "Largura/altura do corpo",
    "altura_relativa": "Altura vs. em pé",
    "objetos_proximos": "Objetos no quadro",
    "clipe": "Clipe",
    "serie": "Sinal",
    "tecnica": "Técnica",
    "z_pico": "Pico (σ)",
    "termo": "Termo",
    "categoria": "Categoria",
    "trecho": "Trecho",
    "inicio": "Início",
    "entidade": "Entidade (TA for Health)",
    "probabilidade": "Probabilidade",
    "droga": "Medicamento",
    "classe": "Classe",
    "dose": "Dose",
    "dose_anterior": "Dose anterior",
    "dose_nova": "Dose nova",
    "unidade": "Unidade",
    "via": "Via",
    "janela_h": "Janela da Fusão (h)",
}
_OCULTOS = {"frame", "valores", "janela", "arquivo", "hadm_id", "versao_listas"}
_VALORES = {
    "regras_clinicas": "regra clínica",
    "zscore_movel": "z-score em janela móvel",
    "isolation_forest": "Isolation Forest",
    "insuficiencia_respiratoria": "Sinal Vital respiratório + Termo Crítico respiratório",
    "queda_com_sedativo": "Queda + início de opioide/sedativo",
    "efeito_adverso_cardiovascular": "Sinal Vital hemodinâmico + medicação cardiovascular",
    "agravamento_reabilitacao": "Desvio de Execução recorrente + Termo Crítico de dor",
    "salto_de_dose": "aumento de dose > 50% em < 24 h",
    "alta_vigilancia": "início de droga de alta vigilância",
    "trocas_excessivas": "trocas/suspensões excessivas",
    "duplicidade": "duplicidade terapêutica",
    "respiratorio": "respiratório",
    "dor": "dor",
    "esq": "esquerdo",
    "dir": "direito",
    **{str(regra): texto for regra, texto in REGRAS_VIDEO.items()},
}


@st.cache_data(ttl=300, show_spinner=False)
def _fichas() -> dict[str, Ficha]:
    return carregar_fichas(demo.carregar_pacientes())


def _fmt(chave: str, valor: Any) -> str:
    if chave == "altura_relativa" and isinstance(valor, int | float):
        return f"{valor:.0%}"
    if isinstance(valor, float):
        return f"{valor:.2f}".rstrip("0").rstrip(".")
    if isinstance(valor, list):
        return ", ".join(str(v) for v in valor)
    return _VALORES.get(str(valor), str(valor))


def fatos_da_evidencia(evidencia: dict[str, Any]) -> list[tuple[str, str]]:
    """Pares (rótulo, valor) legíveis; campos técnicos ficam no expander de detalhes."""
    return [
        (_CAMPOS.get(chave, chave.replace("_", " ")), _fmt(chave, valor))
        for chave, valor in evidencia.items()
        if chave not in _OCULTOS and valor not in (None, "")
    ]


def tabela_alertas(alertas: list[Alerta], fichas: dict[str, Ficha] | None = None) -> pd.DataFrame:
    """Uma linha por Alerta, na ordem recebida (já ordenada por `db.listar`)."""
    fichas = fichas or {}
    return pd.DataFrame(
        [
            {
                "Severidade": f"{ICONE_SEVERIDADE[a.severidade]} {ROTULO_SEVERIDADE[a.severidade]}",
                "Horário": a.detectado_em.strftime("%d/%m %H:%M"),
                "Paciente": rotulo_paciente(a.paciente_id, fichas),
                "Origem": ROTULO_ORIGEM[a.origem],
                "Tipo": ROTULO_TIPO[a.tipo],
                "Status": ROTULO_STATUS[a.status],
                "Descrição": a.descricao,
            }
            for a in alertas
        ],
        columns=["Severidade", "Horário", "Paciente", "Origem", "Tipo", "Status", "Descrição"],
    )


def _caminho(relativo: str) -> Path:
    caminho = Path(relativo)
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _grafico_janela(evidencia: dict) -> go.Figure:
    valores = [v[0] if isinstance(v, list) else v for v in evidencia["valores"]]
    t0, t1 = evidencia.get("janela", [0, len(valores) - 1])
    passo = (t1 - t0) / max(len(valores) - 1, 1)
    figura = go.Figure(
        go.Scatter(
            x=[(t0 + i * passo) / 60 for i in range(len(valores))],
            y=valores,
            mode="lines+markers",
            name=evidencia.get("serie", "série"),
        )
    )
    if "limite" in evidencia:
        figura.add_hline(y=float(evidencia["limite"]), line={"dash": "dot", "color": "#c0392b"})
    figura.update_layout(
        height=220,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=f"{evidencia.get('serie', 'Sinal')} na janela do Alerta",
        xaxis_title="minuto do registro",
    )
    return figura


def _mostrar_fatos(evidencia: dict[str, Any]) -> None:
    fatos = fatos_da_evidencia(evidencia)
    if fatos:
        st.markdown("\n".join(f"- **{rotulo}:** {valor}" for rotulo, valor in fatos))


def renderizar_evidencia(alerta: Alerta) -> None:
    """Evidência por origem: frame anotado, trecho da Consulta ou gráfico da janela do sinal."""
    ev = alerta.evidencia
    match alerta.origem:
        case Origem.VIDEO:
            frame = _caminho(ev["frame"]) if "frame" in ev else None
            if frame is not None and frame.is_file():
                st.image(str(frame), caption=f"Frame de Evidência — t = {ev.get('t', '?')} s")
            else:
                st.caption("Frame de Evidência indisponível.")
        case Origem.AUDIO:
            arquivo = _caminho(ev["arquivo"]) if "arquivo" in ev else None
            if arquivo is not None and arquivo.is_file():
                st.audio(str(arquivo), start_time=int(float(ev.get("inicio", 0))))
            if "trecho" in ev:
                st.markdown(f"> {ev['trecho']}")
        case Origem.SINAIS_VITAIS:
            if ev.get("valores") and len(ev["valores"]) > 1:
                st.plotly_chart(_grafico_janela(ev), width="stretch")
    _mostrar_fatos({k: v for k, v in ev.items() if k != "trecho"})
    with st.expander("Detalhes técnicos"):
        st.code(json.dumps(ev, ensure_ascii=False, indent=2), language="json")


def _renderizar_origens(conn: sqlite3.Connection, composto: Alerta) -> None:
    st.markdown("**Alertas que se confirmaram na Fusão**")
    for origem_id in composto.alertas_origem:
        origem = db.obter(conn, origem_id)
        if origem is None:
            st.caption(f"Alerta {origem_id} não encontrado")
            continue
        titulo = (
            f"{ICONE_SEVERIDADE[origem.severidade]} {ROTULO_ORIGEM[origem.origem]} · "
            f"{origem.descricao}"
        )
        with st.expander(titulo):
            renderizar_evidencia(origem)


def _acoes(conn: sqlite3.Connection, alerta: Alerta, responsavel: str) -> None:
    coluna_a, coluna_b = st.columns(2)
    reconhecer = coluna_a.button(
        "✋ Reconhecer",
        disabled=alerta.status is not Status.NOVO,
        type="primary",
        key=f"reconhecer-{alerta.id}",
        width="stretch",
        help="Declara que você viu o Alerta e assumiu o tratamento.",
    )
    resolver = coluna_b.button(
        "✅ Resolver",
        disabled=alerta.status is not Status.RECONHECIDO,
        key=f"resolver-{alerta.id}",
        width="stretch",
    )
    if not (reconhecer or resolver):
        return
    try:
        if reconhecer:
            if not responsavel.strip():
                st.error("Informe o plantonista na barra lateral antes de reconhecer.")
                return
            db.reconhecer(conn, alerta.id, por=responsavel.strip())
        else:
            db.resolver(conn, alerta.id)
    except TransicaoInvalida as erro:
        st.error(str(erro))
        return
    st.rerun()


def _detalhe(
    conn: sqlite3.Connection, alerta: Alerta, responsavel: str, fichas: dict[str, Ficha]
) -> None:
    with st.container(border=True):
        st.markdown(
            f"#### {ICONE_SEVERIDADE[alerta.severidade]} {ROTULO_SEVERIDADE[alerta.severidade]}"
            f" · {ROTULO_TIPO[alerta.tipo]}"
        )
        st.markdown(f"**{alerta.descricao}**")
        ficha = fichas.get(alerta.paciente_id)
        paciente = rotulo_paciente(alerta.paciente_id, fichas)
        if ficha is not None and ficha.identificacao:
            paciente = f"{paciente} ({ficha.identificacao})"
        st.caption(
            f"{paciente} · {ROTULO_ORIGEM[alerta.origem]} · "
            f"{alerta.detectado_em:%d/%m/%Y %H:%M} · **{ROTULO_STATUS[alerta.status]}**"
        )
        if alerta.reconhecido_por and alerta.reconhecido_em:
            st.caption(
                f"Reconhecido por {alerta.reconhecido_por} em {alerta.reconhecido_em:%d/%m %H:%M}"
            )
        _acoes(conn, alerta, responsavel)
    st.markdown("**Evidência**")
    renderizar_evidencia(alerta)
    if alerta.origem is Origem.FUSAO:
        _renderizar_origens(conn, alerta)


def _indicadores(alertas: list[Alerta]) -> None:
    pendentes = [a for a in alertas if a.status is Status.NOVO]
    criticos = sum(a.severidade is Severidade.CRITICA for a in pendentes)
    altos = sum(a.severidade is Severidade.ALTA for a in pendentes)
    em_tratamento = sum(a.status is Status.RECONHECIDO for a in alertas)
    pacientes_graves = {
        a.paciente_id
        for a in alertas
        if a.status is not Status.RESOLVIDO
        and a.severidade in (Severidade.CRITICA, Severidade.ALTA)
    }
    colunas = st.columns(4)
    colunas[0].metric("🔴 Críticos sem reconhecimento", criticos)
    colunas[1].metric("🟠 Altos sem reconhecimento", altos)
    colunas[2].metric("✋ Em tratamento", em_tratamento)
    colunas[3].metric("🛏️ Pacientes com alerta grave", len(pacientes_graves))
    if criticos:
        st.error(f"🚨 {criticos} Alerta(s) crítico(s) aguardando Reconhecimento.")


def _filtros(todos: list[Alerta], fichas: dict[str, Ficha]) -> dict[str, Any]:
    with st.sidebar:
        st.header("Plantão")
        responsavel = st.text_input(
            "Plantonista",
            placeholder="ex.: enf. Ana",
            key="responsavel",
            help="Quem reconhece os Alertas nesta sessão.",
        )
        atualizar = st.toggle(
            f"Atualizar a cada {SEGUNDOS_ATUALIZACAO} s",
            key="central_auto",
            help="Útil com o replay do Painel do Paciente rodando em outra aba.",
        )
        st.header("Filtros")
        recorte = st.radio(
            "Pacientes",
            list(ESCOPOS),
            format_func=ESCOPOS.get,
            key="central_escopo",
        )
        do_recorte = [
            a for a in todos if recorte == "todos" or escopo(a.paciente_id, fichas) == recorte
        ]
        status = st.multiselect(
            "Status",
            list(Status),
            default=[Status.NOVO, Status.RECONHECIDO],
            format_func=ROTULO_STATUS.get,
            key="central_status",
        )
        severidades = st.multiselect(
            "Severidade", list(Severidade), format_func=ROTULO_SEVERIDADE.get, key="central_sev"
        )
        origens = st.multiselect(
            "Origem", list(Origem), format_func=ROTULO_ORIGEM.get, key="central_origem"
        )
        pacientes = st.multiselect(
            "Paciente",
            sorted({a.paciente_id for a in do_recorte}),
            format_func=lambda p: rotulo_paciente(p, fichas),
            key="central_paciente",
        )
    return {
        "responsavel": responsavel,
        "atualizar": atualizar,
        "ids_recorte": {a.paciente_id for a in do_recorte},
        "status": status,
        "severidades": severidades,
        "origens": origens,
        "pacientes": pacientes,
    }


def _fila(conn: sqlite3.Connection, filtros: dict[str, Any], fichas: dict[str, Ficha]) -> None:
    no_recorte = db.listar(conn, paciente_id=filtros["pacientes"] or filtros["ids_recorte"])
    _indicadores(no_recorte)
    alertas = [
        a
        for a in no_recorte
        if (not filtros["status"] or a.status in filtros["status"])
        and (not filtros["severidades"] or a.severidade in filtros["severidades"])
        and (not filtros["origens"] or a.origem in filtros["origens"])
    ]

    lista, detalhe = st.columns([3, 2], gap="large")
    with lista:
        st.caption(f"{len(alertas)} Alerta(s) na fila · marque a caixa à esquerda para abrir")
        if not alertas:
            st.success("Nenhum Alerta pendente para os filtros escolhidos.")
        else:
            selecao = st.dataframe(
                tabela_alertas(alertas, fichas),
                hide_index=True,
                width="stretch",
                height=min(38 * len(alertas) + 40, 560),
                on_select="rerun",
                selection_mode="single-row",
                key="tabela_alertas",
                column_config={"Descrição": st.column_config.TextColumn(width="large")},
            )
            linhas = selecao.selection.rows
            if linhas and linhas[0] < len(alertas):
                st.session_state["alerta_aberto"] = alertas[linhas[0]].id
    with detalhe:
        aberto = st.session_state.get("alerta_aberto")
        # relê do banco para refletir o status mais recente
        escolhido = db.obter(conn, aberto) if aberto else None
        if escolhido is None:
            st.info("Selecione um Alerta na fila para ver a Evidência e agir.")
        else:
            _detalhe(conn, escolhido, filtros["responsavel"], fichas)


def pagina(conn: sqlite3.Connection) -> None:
    st.title("Central de Alertas")
    st.caption(
        "Fila única de Alertas da equipe: Sinais Vitais, prescrições, Consultas, vídeo e "
        "Alertas Compostos da Fusão."
    )

    if not db.listar(conn):
        mock.popular_mock(conn)
        st.info("Banco vazio: Alertas mock carregados (rode `scripts/run_batch.py`).")

    fichas = _fichas()
    filtros = _filtros(db.listar(conn), fichas)
    intervalo = SEGUNDOS_ATUALIZACAO if filtros["atualizar"] else None
    st.fragment(_fila, run_every=intervalo)(conn, filtros, fichas)
