"""Tela 3: Laboratório de Análise (ESTRATEGIA.md, seção 9) — upload de vídeo e de áudio.

Vídeo usa `analyze_video` (`src/analyzers/video/lab.py`, #13): esqueleto YOLOv8-pose sobre os
frames de Evidência, ângulos, Desvios de Execução, Quedas e o Relatório de Sessão. Áudio usa
`analyze_audio` (`src/analyzers/audio/lab.py`, #14): transcrição do Azure, Termos Críticos,
Sentimento, métricas acústicas e probabilidade de Alteração Vocal. Os dois devolvem os Alertas
prontos para gravar.
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
from src.analyzers.video import pose
from src.analyzers.video.exercise_rules import DESCRICAO_REGRA, EXERCICIOS, Vista
from src.analyzers.video.lab import SAIDA_PADRAO, ResultadoVideo, analyze_video
from src.core import db
from src.core.alerts import Alerta
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


def _mostrar_alertas(conn: sqlite3.Connection, alertas: list[Alerta], chave: str) -> None:
    st.markdown("**Alertas gerados**")
    if not alertas:
        st.caption("Nenhum Alerta disparado nesta análise.")
        return
    for alerta in alertas:
        with st.expander(f"{ICONE_SEVERIDADE[alerta.severidade]} {alerta.descricao}"):
            renderizar_evidencia(alerta)
    if st.button("Gravar Alertas no Banco", type="primary", key=f"gravar_alertas_{chave}"):
        # Só os novos: regravar sobrescreveria o status de um Alerta já tratado pela equipe.
        novos = [a for a in alertas if db.obter(conn, a.id) is None]
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


CENARIOS: dict[str, str | None] = {
    **{f"Sessão de Fisioterapia: {ex.rotulo}": nome for nome, ex in EXERCICIOS.items()},
    "Quarto: monitoramento de Queda": None,
}
PASTA_LABORATORIO = SAIDA_PADRAO / "laboratorio"


@st.cache_resource
def _modelo_pose():
    return pose.carregar_modelo()


def _analisar_video(arquivo, exercicio: str | None, vista: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / arquivo.name
        caminho.write_bytes(arquivo.getvalue())
        with st.spinner("Estimando a pose quadro a quadro (YOLOv8n-pose, CPU)..."):
            try:
                resultado = analyze_video(
                    caminho,
                    exercicio=exercicio,
                    vista=vista,
                    saida=PASTA_LABORATORIO,
                    modelo=_modelo_pose(),
                )
            except Exception as erro:  # vídeo corrompido, codec sem suporte no OpenCV...
                st.error(f"Falha ao analisar o vídeo: {erro}")
                return
    st.session_state["resultado_video"] = resultado


def _mostrar_video(conn: sqlite3.Connection, resultado: ResultadoVideo) -> None:
    st.caption(f"Resultado de: {resultado.arquivo}")
    colunas = st.columns(4)
    colunas[0].metric("Repetições", len(resultado.repeticoes))
    colunas[1].metric("Desvios de Execução", len(resultado.desvios))
    colunas[2].metric("Quedas", len(resultado.quedas))
    colunas[3].metric(
        "CPU / s de vídeo",
        f"{resultado.fator_tempo_real:.2f}",
        help=f"{resultado.sequencia.segundos_processamento:.1f} s para "
        f"{resultado.sequencia.duracao:.1f} s de vídeo",
    )

    if resultado.evidencias:
        st.markdown("**Evidências (esqueleto YOLOv8-pose)**")
        colunas = st.columns(3)
        for i, evidencia in enumerate(resultado.evidencias):
            colunas[i % 3].image(
                evidencia.caminho, caption=f"{evidencia.titulo} — {evidencia.t:.1f} s"
            )
    else:
        st.caption("Nenhum Desvio nem Queda: sem frames de Evidência.")

    if resultado.desvios:
        st.dataframe(
            [
                {
                    "desvio": DESCRICAO_REGRA[d.regra],
                    "rep./episódio": d.repeticao,
                    "instante (s)": round(d.t, 1),
                    "medido (°)": round(d.valor),
                    "limite (°)": round(d.limite),
                }
                for d in resultado.desvios
            ],
            hide_index=True,
        )

    grafico = resultado.relatorio.parent / "angulos.png"
    if grafico.is_file():
        st.image(str(grafico), caption="Ângulos ao longo do vídeo")
    st.download_button(
        "Baixar Relatório de Sessão (Markdown)",
        resultado.relatorio.read_text(encoding="utf-8"),
        file_name=f"relatorio_{resultado.clipe_id}.md",
        mime="text/markdown",
        key="baixar_relatorio_video",
    )
    _mostrar_alertas(conn, resultado.alertas, "video")


def _secao_video() -> None:
    arquivo = st.file_uploader(
        "Vídeo da Sessão de Fisioterapia ou do quarto (MP4, WebM, AVI, MOV, OGV...)",
        type=["mp4", "webm", "avi", "mov", "ogv", "mkv"],
        key="upload_video",
    )
    cenario, vista = st.columns(2)
    rotulo = cenario.selectbox("Cenário", list(CENARIOS), key="cenario_video")
    vista_escolhida = vista.radio(
        "Vista da câmera",
        [str(v) for v in Vista],
        horizontal=True,
        key="vista_video",
        help="Valgo e assimetria só são avaliados com a câmera de frente (ou de costas).",
    )
    if arquivo is not None and st.button("Analisar vídeo", key="analisar_video"):
        _analisar_video(arquivo, CENARIOS[rotulo], vista_escolhida)

    resultado = st.session_state.get("resultado_video")
    if resultado is not None:
        st.divider()
        _mostrar_video(_conexao(), resultado)


def _secao_audio() -> None:
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
    _mostrar_alertas(_conexao(), resultado.alertas, "audio")


def pagina() -> None:
    st.title("Laboratório de Análise")
    aba_video, aba_audio = st.tabs(["🎥 Vídeo", "🎙️ Áudio"])
    with aba_video:
        _secao_video()
    with aba_audio:
        _secao_audio()
