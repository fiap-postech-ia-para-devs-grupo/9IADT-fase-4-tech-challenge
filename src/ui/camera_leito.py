"""Monitoramento do leito no Prontuário: integração (simulada) com a câmera e o microfone.

Cada leito tem dois dispositivos: a **câmera** (YOLOv8-pose vigiando Queda) e o **microfone**
(AST/AudioSet + Whisper vigiando alarme, grito, gemido e chamado da equipe). A fonte pode ser o
dispositivo real deste computador (webcam / microfone) ou uma gravação tocada em tempo real,
como se fosse o quarto. Os Alertas saem para o paciente daquele leito.

As vigilâncias ficam num registro do processo (uma câmera e um microfone por leito): continuam
rodando quando a pessoa troca de tela.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from src.analyzers.audio.vigilancia_audio import MICROFONE, VigilanciaAudio
from src.analyzers.video import fall, objetos, pose
from src.analyzers.video.vigilancia import Vigilancia
from src.core import fontes, notificacao
from src.core.demo import PacienteDemo
from src.ui import plantao_estado as plantao

SEGUNDOS_TELA = 0.25
LARGURA_IMAGEM = 460  # px: quadro menor, sobra espaço para o restante do prontuário
_SONS = Path(fontes._DADOS) / "audio" / "sons"
SONS_EXEMPLO = {
    "alarme_monitor.mp3": "Gravação: alarme do monitor",
    "grito.ogg": "Gravação: grito",
    "gemido_choro.ogg": "Gravação: gemido / choro",
    "chamado_enfermeira.wav": "Gravação: “Enfermeira! Socorro!”",
}


@st.cache_resource
def _registro() -> dict[str, dict]:
    return {}


@st.cache_resource(show_spinner="Carregando o YOLOv8 (pose e objetos)...")
def _modelos_video():
    return pose.carregar_modelo(), objetos.carregar_modelo()


def _do_leito(paciente_id: str) -> dict:
    return _registro().setdefault(paciente_id, {"video": None, "audio": None})


def ativo(paciente_id: str, tipo: str) -> bool:
    vigia = _do_leito(paciente_id)[tipo]
    return vigia is not None and vigia.estado.ativo


def algum_ativo() -> bool:
    return any(
        v is not None and v.estado.ativo for leito in _registro().values() for v in leito.values()
    )


def parar_tudo(paciente_id: str) -> None:
    for vigia in _do_leito(paciente_id).values():
        if vigia is not None:
            vigia.parar()


def _fontes_audio() -> dict[str, str]:
    opcoes = {MICROFONE: "Microfone deste computador (ao vivo)"}
    opcoes |= {
        str(_SONS / arquivo): rotulo
        for arquivo, rotulo in SONS_EXEMPLO.items()
        if (_SONS / arquivo).exists()
    }
    return opcoes


def _ligar_video(paciente: PacienteDemo, escolha: str) -> None:
    modelo_pose, modelo_objetos = _modelos_video()
    if escolha == fontes.WEBCAM:
        fonte, recorte = 0, None
    else:
        clipe = fontes.clipe(escolha) or {}
        fonte = str(fontes._DADOS / "video" / clipe.get("arquivo", ""))
        recorte = tuple(clipe["recorte"]) if clipe.get("recorte") else None
    nova = Vigilancia(
        paciente.id,
        fonte,
        inicio_plantao=plantao.agora(),
        recorte=recorte,
        modelo_pose=modelo_pose,
        modelo_objetos=modelo_objetos,
    )
    _do_leito(paciente.id)["video"] = nova
    nova.iniciar()


def _ligar_audio(paciente: PacienteDemo, escolha: str) -> None:
    nova = VigilanciaAudio(
        paciente.id, escolha, relogio=plantao.agora()
    )  # sem classificador: a thread usa o compartilhado, carregado em segundo plano
    _do_leito(paciente.id)["audio"] = nova
    nova.iniciar()


def controles(paciente: PacienteDemo, camera_admissao: str | None) -> None:
    """Dispositivos do leito: marque vídeo e/ou áudio, escolha as fontes e inicie."""
    leito = _do_leito(paciente.id)
    rodando = ativo(paciente.id, "video") or ativo(paciente.id, "audio")
    st.markdown(
        '<div class="secao">Dispositivos do leito (integração simulada)</div>',
        unsafe_allow_html=True,
    )
    usar_video = st.checkbox(
        "📷 Vídeo — câmera do leito (Queda, YOLOv8-pose)",
        value=True,
        key=f"usar_video_{paciente.id}",
        disabled=rodando,
    )
    cameras = fontes.cameras()
    padrao = camera_admissao if camera_admissao in cameras else fontes.WEBCAM
    camera = st.selectbox(
        "Fonte da câmera",
        list(cameras),
        index=list(cameras).index(padrao),
        format_func=lambda c: cameras[c],
        key=f"camera_fonte_{paciente.id}",
        disabled=rodando or not usar_video,
    )
    usar_audio = st.checkbox(
        "🎙️ Áudio — microfone do leito (alarme, grito, gemido, chamado)",
        value=True,
        key=f"usar_audio_{paciente.id}",
        disabled=rodando,
    )
    sons = _fontes_audio()
    som = st.selectbox(
        "Fonte do microfone",
        list(sons),
        format_func=lambda s: sons[s],
        key=f"audio_fonte_{paciente.id}",
        disabled=rodando or not usar_audio,
    )
    if rodando:
        if st.button("⏹ Parar monitoramento", key=f"parar_{paciente.id}", width="stretch"):
            parar_tudo(paciente.id)
            st.rerun()
        return
    if st.button(
        "▶ Iniciar monitoramento do leito",
        type="primary",
        key=f"iniciar_{paciente.id}",
        width="stretch",
        disabled=not (usar_video or usar_audio),
    ):
        if usar_video:
            _ligar_video(paciente, camera)
        if usar_audio:
            _ligar_audio(paciente, som)
        st.rerun()
    if leito["video"] is None and leito["audio"] is None:
        st.caption(
            "A webcam e o microfone abrem no computador onde o app roda. As gravações tocam em "
            "tempo real como se fossem o quarto."
        )


def _alarme(paciente_id: str, video, audio) -> None:
    """Alarme visual imediato quando a câmera ou o microfone do leito detectam algo."""
    eventos = []
    if video is not None and video.estado.quedas:
        eventos.append("QUEDA DETECTADA")
    if audio is not None:
        criticos = [e for _, e, _ in audio.estado.eventos if e in {"grito", "chamado"}]
        if criticos:
            eventos.append(
                "PACIENTE CHAMANDO A EQUIPE" if "chamado" in criticos else "GRITO DETECTADO"
            )
    if not eventos:
        return
    st.html(
        f"""<div class="luz-chamada"><span class="lampada"></span>
        <div class="texto"><b>🚨 {" · ".join(eventos)}</b><br>
        <small>Alerta enviado à Central e luz do leito acesa ·
        {html.escape(notificacao.situacao_acionamento(plantao.conexao()))}.
        </small></div></div>"""
    )
    chave = f"alarme_{paciente_id}_{'_'.join(eventos)}"
    if not st.session_state.get(chave):
        st.session_state[chave] = True
        st.toast(f"🚨 {' · '.join(eventos)}", icon="🚨")


def ao_vivo(paciente_id: str) -> None:
    """Imagem da câmera e medidor do microfone (dentro de um fragmento que se atualiza)."""
    leito = _do_leito(paciente_id)
    video, audio = leito["video"], leito["audio"]
    _alarme(paciente_id, video, audio)
    if video is None and audio is None:
        st.info(
            "Monitoramento do leito desligado. Marque vídeo e/ou áudio e clique em ▶ Iniciar: "
            "uma Queda, um grito ou um chamado vira Alerta deste paciente, acende a luz do leito "
            "e aciona a equipe de plantão."
        )
        return
    agora = plantao.agora()
    if video is not None:
        video.relogio = agora  # os Alertas saem no horário do plantão
        estado = video.estado
        if estado.erro:
            st.error(f"Câmera: {estado.erro}")
        if estado.ultimo_quadro is not None:
            st.image(estado.ultimo_quadro, channels="BGR", width=LARGURA_IMAGEM)
        elif estado.ativo:
            st.caption("Abrindo a câmera e carregando o modelo...")
        situacao = (
            "🚨 Queda detectada · câmera parada no quadro da Queda (▶ Iniciar volta a vigiar)"
            if getattr(estado, "parada_por_queda", False)
            else "🟢 vigiando"
            if estado.ativo
            else "🎬 gravação terminou"
            if getattr(estado, "fim_da_gravacao", False)
            else "⚪ parada"
        )
        st.caption(
            f"📷 {situacao} · {estado.fps:.1f} quadros/s · Quedas detectadas: {len(estado.quedas)}"
        )
        if getattr(estado, "leitura", ""):
            st.caption(f"Leitura do detector: {estado.leitura}")
        ponto = getattr(estado, "score", None)
        if ponto is not None:
            st.progress(
                ponto.score / 100,
                text=f"Score de Queda {ponto.score}/100 · alerta em ≥ {fall.SCORE_QUEDA} com "
                f"ângulo ≥ {fall.ANGULO_QUEDA:.0f}°",
            )
            a, v, d = ponto.componentes
            st.caption(
                f"Ângulo {ponto.angulo:.0f}° ({a:.0%} × {fall.PESO_INCLINACAO:.0%}) · "
                f"variação {ponto.variacao:+.0f}° em {fall.JANELA_BRUSCA:.0f} s "
                f"({v:.0%} × {fall.PESO_VARIACAO:.0%}) · ombros desceram "
                f"{ponto.descida:.1f} largura ({d:.0%} × {fall.PESO_DESCIDA:.0%})"
            )
    if audio is not None:
        audio.relogio = agora
        estado = audio.estado
        if estado.erro:
            st.error(f"Microfone: {estado.erro}")
        nivel = min(estado.nivel * 8, 1.0)
        situacao = (
            "⏳ carregando os modelos de som (AST/AudioSet e Whisper)…"
            if getattr(estado, "carregando", False)
            else "🟢 ouvindo"
            if estado.ativo
            else "⚪ parado"
        )
        st.progress(nivel, text=f"🎙️ {situacao} · nível do som")
        if estado.ultimas_classes:
            st.caption(
                "AudioSet: " + ", ".join(f"{nome} {p:.0%}" for nome, p in estado.ultimas_classes)
            )
        if getattr(estado, "transcricao", ""):
            st.caption(f'Whisper ouviu: "{estado.transcricao}"')
        for quando, evento, _ in reversed(estado.eventos[-4:]):
            st.warning(f"🔊 {quando:%H:%M} · {evento} detectado → Alerta enviado à Central")
