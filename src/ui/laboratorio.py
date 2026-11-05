"""Tela 3: Laboratório de Análise (ESTRATEGIA.md, seção 9) — upload de áudio.

Upload de vídeo aguarda o Analisador de Vídeo (#13); áudio usa `analyze_audio`
(`src/analyzers/audio/lab.py`, #14): transcrição do Azure, Termos Críticos, Sentimento,
métricas acústicas e probabilidade de Alteração Vocal, com os Alertas prontos para gravar.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzers.audio.azure import ConfigAzureIncompleta, carregar_config
from src.analyzers.audio.lab import ResultadoLaboratorio, analyze_audio
from src.core import db
from src.ui.central import ICONE_SEVERIDADE, renderizar_evidencia


@st.cache_resource
def _conexao() -> sqlite3.Connection:
    return db.conectar(check_same_thread=False)


def _formatar_sentimento(sentimento: str) -> str:
    icones = {"positive": "🙂", "neutral": "😐", "negative": "🙁", "mixed": "😕"}
    return f"{icones.get(sentimento, '')} {sentimento}"


def _mostrar_transcricao(resultado: ResultadoLaboratorio) -> None:
    st.markdown("**Transcrição (Azure Speech to Text)**")
    if not resultado.transcricao:
        st.caption("Nenhuma fala reconhecida.")
        return
    for trecho, sent in zip(resultado.transcricao, resultado.sentimentos, strict=True):
        sentimento = _formatar_sentimento(sent.sentimento)
        st.markdown(f"`{trecho.inicio:6.1f}s` {trecho.texto} — {sentimento}")


def _mostrar_features_acusticas(resultado: ResultadoLaboratorio) -> None:
    st.markdown("**Métricas acústicas**")
    f = resultado.features_acusticas
    colunas = st.columns(4)
    pares = [
        ("F0 médio", f"{f.get('f0_media', 0):.0f} Hz"),
        ("Jitter", f"{f.get('jitter_local', 0):.4f}"),
        ("Shimmer", f"{f.get('shimmer_local', 0):.4f}"),
        ("HNR", f"{f.get('hnr_media', 0):.1f} dB"),
        ("Proporção de pausas", f"{f.get('proporcao_pausas', 0):.0%}"),
        ("Taxa de fala (prox.)", f"{f.get('taxa_fala', 0):.2f} /s"),
    ]
    for i, (rotulo, valor) in enumerate(pares):
        colunas[i % 4].metric(rotulo, valor)

    if resultado.probabilidade_alteracao_vocal is None:
        st.caption(
            "Classificador acústico ainda não treinado "
            "(`uv run python -m scripts.treinar_acustico`): sem probabilidade de Alteração Vocal."
        )
    else:
        probabilidade = resultado.probabilidade_alteracao_vocal
        st.metric("Probabilidade de Alteração Vocal", f"{probabilidade:.0%}")


def _mostrar_alertas(conn: sqlite3.Connection, resultado: ResultadoLaboratorio) -> None:
    st.markdown("**Alertas gerados**")
    if not resultado.alertas:
        st.caption("Nenhum Alerta disparado nesta análise.")
        return
    for alerta in resultado.alertas:
        with st.expander(f"{ICONE_SEVERIDADE[alerta.severidade]} {alerta.descricao}"):
            renderizar_evidencia(alerta)
    if st.button("Gravar Alertas no Banco", type="primary", key="gravar_alertas_audio"):
        # Só os novos: regravar sobrescreveria o status de um Alerta já tratado pela equipe.
        novos = [a for a in resultado.alertas if db.obter(conn, a.id) is None]
        for alerta in novos:
            db.salvar(conn, alerta)
        st.success(f"{len(novos)} Alerta(s) novo(s) gravado(s) na Central de Alertas.")


def _analisar_upload(arquivo) -> None:
    load_dotenv()
    try:
        config = carregar_config(os.environ)
    except ConfigAzureIncompleta as erro:
        st.error(str(erro))
        return

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / arquivo.name
        caminho.write_bytes(arquivo.getvalue())
        with st.spinner("Transcrevendo e analisando (Azure Speech + Language)..."):
            try:
                resultado = analyze_audio(caminho, config)
            except Exception as erro:  # Azure/ffmpeg podem falhar por motivos variados
                st.error(f"Falha ao analisar o áudio: {erro}")
                return

    st.session_state["resultado_audio"] = resultado


def pagina() -> None:
    st.title("Laboratório de Análise")
    st.info("Upload de vídeo em construção (#13): o esqueleto YOLOv8-pose chega com ele.")

    arquivo = st.file_uploader(
        "Áudio da Consulta (WAV, MP3, M4A...)", type=None, key="upload_audio"
    )
    if arquivo is not None and st.button("Analisar", key="analisar_audio"):
        _analisar_upload(arquivo)

    resultado = st.session_state.get("resultado_audio")
    if resultado is None:
        return

    st.divider()
    st.caption(f"Resultado de: {resultado.arquivo}")
    _mostrar_transcricao(resultado)
    _mostrar_features_acusticas(resultado)
    _mostrar_alertas(_conexao(), resultado)
