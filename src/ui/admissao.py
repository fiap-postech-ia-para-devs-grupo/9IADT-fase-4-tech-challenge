"""Admissão: dar entrada num paciente, escolher o leito e ligar as fontes de monitoramento.

Cada fonte alimenta um Analisador: o monitor multiparamétrico (Sinais Vitais, MIMIC-III), a
prescrição eletrônica (Mudança Terapêutica Atípica, MIMIC-IV Demo), a Consulta gravada (Termos
Críticos via Azure, PriMock57) e a câmera do leito (Queda e Desvio, YOLOv8). A partir da
admissão, o paciente aparece no Mapa de Leitos e no Prontuário, e os Alertas dele chegam à
Central conforme o relógio do plantão anda.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from src.core import cadastro, fontes
from src.ui import cabecalho, navegacao
from src.ui import plantao_estado as plantao


def pagina() -> None:
    cabecalho.barra(avancar=False)
    st.title("Admissão")
    st.caption(
        "Dê entrada no paciente, escolha o leito e ligue as fontes que vão monitorá-lo. As fontes "
        "são dados públicos de pessoas diferentes: o paciente é composto (ADR 0001)."
    )
    conn = plantao.conexao()
    livres = cadastro.leitos_livres(conn)
    if not livres:
        st.warning("Nenhum leito livre. Cadastre leitos em Administração › Equipe e Leitos.")
        return

    registros = fontes.registros_vitais()
    prescricoes = fontes.prescricoes()
    consultas = fontes.consultas()
    cameras = fontes.cameras()
    if not registros or not prescricoes or not consultas:
        st.error(
            "Fontes ausentes: `uv run python scripts/download_data.py mimic3wdb mimic4demo video`."
        )
        return

    with st.form("admissao", border=True):
        st.markdown('<div class="secao">1 · Paciente</div>', unsafe_allow_html=True)
        nome_col, idade_col, sexo_col = st.columns([3, 1, 1])
        nome = nome_col.text_input("Nome", placeholder="Nome do paciente (fictício)")
        idade = idade_col.number_input("Idade", 0, 120, 60)
        sexo = sexo_col.selectbox("Sexo", ["feminino", "masculino"])
        motivo_col, leito_col = st.columns([3, 2])
        motivo = motivo_col.text_input(
            "Motivo da internação", placeholder="ex.: pneumonia, pós-operatório de quadril"
        )
        leito = leito_col.selectbox(
            "Leito", livres, format_func=lambda leito: f"{leito.id} · {leito.unidade}"
        )

        st.markdown('<div class="secao">2 · Fontes de monitoramento</div>', unsafe_allow_html=True)
        vitais_col, prescricao_col = st.columns(2)
        registro = vitais_col.selectbox(
            "🫀 Monitor multiparamétrico (Sinais Vitais)",
            registros,
            format_func=lambda r: f"registro {r.rsplit('/', 1)[-1]} · MIMIC-III",
            help="FC, FR, SpO2 e PA minuto a minuto: regras clínicas, z-score e Isolation Forest.",
        )
        prescricao = prescricao_col.selectbox(
            "💊 Prescrição eletrônica",
            prescricoes,
            format_func=lambda o: o.rotulo,
            help="Salto de dose, alta vigilância, trocas e duplicidade terapêutica (MIMIC-IV).",
        )
        consulta_col, camera_col = st.columns(2)
        consulta = consulta_col.selectbox(
            "🎙️ Consulta gravada (áudio)",
            list(consultas),
            format_func=lambda c: f"{c} · {consultas[c]} Termo(s) Crítico(s)",
            help="Transcrição Azure + Termos Críticos (TA for Health) já processados (PriMock57).",
        )
        camera = camera_col.selectbox(
            "📷 Câmera do leito",
            [None, *cameras],
            format_func=lambda c: "Sem câmera" if c is None else cameras[c],
            help="Vigilância contínua de Queda com YOLOv8-pose; ligue no Prontuário.",
        )
        st.markdown('<div class="secao">3 · Linha do tempo</div>', unsafe_allow_html=True)
        prescr_min, consulta_min, camera_min = st.columns(3)
        minuto_prescricao = prescr_min.number_input(
            "Prescrições começam (min após a admissão)", 0, 600, 30, step=10
        )
        minuto_consulta = consulta_min.number_input(
            "Consulta (min após a admissão)", 0, 600, 60, step=10
        )
        minuto_camera = camera_min.number_input(
            "Sessão/câmera (min após a admissão)", 0, 600, 120, step=10
        )
        admitir = st.form_submit_button("Admitir paciente", type="primary")

    if not admitir:
        return
    if not nome.strip():
        st.error("Informe o nome do paciente.")
        return
    responsavel = st.session_state.get("responsavel") or "plantonista não identificado"
    admitido_em = plantao.agora()
    internacao = cadastro.Internacao(
        id=cadastro.novo_id(),
        nome=nome.strip(),
        idade=int(idade),
        sexo=sexo,
        leito_id=leito.id,
        motivo=motivo.strip(),
        admitido_em=admitido_em,
        admitido_por=responsavel,
        status="internado",
        fontes=cadastro.Fontes(
            registro_vitais=registro,
            subject_id=prescricao.subject_id,
            hadm_id=prescricao.hadm_id,
            ancora_prescricao=prescricao.inicio.to_pydatetime().isoformat(),
            ancora_min=float(minuto_prescricao),
            consulta_id=consulta,
            consulta_inicio_min=float(minuto_consulta),
            camera=camera,
            camera_inicio_min=float(minuto_camera),
        ),
    )
    try:
        cadastro.admitir(conn, internacao, responsavel)
    except cadastro.LeitoOcupado as erro:
        st.error(str(erro))
        return
    st.success(
        f"{internacao.nome} admitido(a) no leito {leito.id} às {admitido_em:%H:%M} por "
        f"{responsavel}. O monitoramento começa agora; previsão de prescrições às "
        f"{admitido_em + timedelta(minutes=minuto_prescricao):%H:%M}."
    )
    navegacao.link(
        "Prontuário", "Abrir prontuário", icon="🩺", query_params={"leito": internacao.id}
    )
