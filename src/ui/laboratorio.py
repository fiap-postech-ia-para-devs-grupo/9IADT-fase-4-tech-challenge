"""Tela 3: Laboratório de Análise (ESTRATEGIA.md, seção 9) — vídeo e áudio sob demanda.

Vídeo usa `analyze_video` (`src/analyzers/video/lab.py`, #13): esqueleto YOLOv8-pose, ângulos,
Desvios de Execução, Quedas (com as áreas críticas do quarto pelo YOLOv8 de objetos) e o
Relatório de Sessão. Entra por clipe de exemplo, upload ou webcam ao vivo, com a faixa esperada
prescrita pelo fisioterapeuta. Áudio usa `analyze_audio` (`src/analyzers/audio/lab.py`, #14):
transcrição do Azure, Termos Críticos, Sentimento, métricas acústicas e probabilidade de
Alteração Vocal, por upload, microfone ou áudio de exemplo.

Os Alertas podem ir para um Paciente de Demonstração, no horário escolhido da internação, e aí
entram na Fusão do Painel do Paciente; sem paciente, ficam como `avulso`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzers.audio.azure import ConfigAzureIncompleta, carregar_config
from src.analyzers.audio.lab import ResultadoLaboratorio, analyze_audio
from src.analyzers.video import camera, objetos, pose
from src.analyzers.video.exercise_rules import DESCRICAO_REGRA, EXERCICIOS, Exercicio, Vista
from src.analyzers.video.lab import SAIDA_PADRAO, ResultadoVideo, analyze_video
from src.core import db, demo
from src.core.alerts import Alerta
from src.core.ficha import AVULSO
from src.ui import cabecalho
from src.ui import plantao_estado as plantao
from src.ui.central import ICONE_SEVERIDADE, renderizar_evidencia

RAIZ = Path(__file__).resolve().parents[2]
PASTA_VIDEO = RAIZ / "data" / "video"
AUDIO_EXEMPLO = RAIZ / "data" / "audio" / "teste_upload" / "teste_falta_de_ar.wav"
PASTA_LABORATORIO = SAIDA_PADRAO / "laboratorio"

CENARIOS: dict[str, str | None] = {
    **{f"Sessão de Fisioterapia: {ex.rotulo}": nome for nome, ex in EXERCICIOS.items()},
    "Quarto: monitoramento de Queda": None,
}
ROTULO_VISTA = {Vista.LATERAL: "de lado", Vista.FRONTAL: "de frente ou de costas"}
# Faixa esperada que o fisioterapeuta pode ajustar: (rótulo, mínimo, máximo, ajuda).
PARAMETROS = {
    "joelho_alvo": (
        "Joelho no fundo da repetição até (°)",
        60,
        150,
        "Menor é mais profundo: 90° é coxa na horizontal.",
    ),
    "tronco_max": ("Inclinação máxima do tronco (°)", 20, 80, "Em relação à vertical."),
    "assimetria_max": ("Diferença máxima entre joelhos (°)", 5, 45, "Só com câmera frontal."),
    "valgo_max": (
        "Valgo máximo do joelho de apoio (°)",
        5,
        25,
        "Ângulo de projeção no plano frontal (FPPA); 10° é o corte usual.",
    ),
}
FONTES_VIDEO = ("Clipe de exemplo", "Enviar arquivo", "Câmera ao vivo")
FONTES_AUDIO = ("Enviar arquivo", "Gravar pelo microfone", "Áudio de exemplo")


def _conexao() -> sqlite3.Connection:
    return plantao.conexao()


@st.cache_resource(show_spinner="Carregando o YOLOv8n-pose...")
def _modelo_pose():
    return pose.carregar_modelo()


@st.cache_resource(show_spinner="Carregando o YOLOv8n de objetos...")
def _modelo_objetos():
    return objetos.carregar_modelo()


@st.cache_data(show_spinner=False)
def _clipes() -> list[dict]:
    manifesto = PASTA_VIDEO / "clipes.json"
    if not manifesto.exists():
        return []
    clipes = json.loads(manifesto.read_text(encoding="utf-8"))["clipes"]
    return [c for c in clipes if (PASTA_VIDEO / c["arquivo"]).exists()]


# --- paciente e horário ---------------------------------------------------------------------


def _seletor_paciente(chave: str, minuto_padrao: str) -> tuple[str, datetime | None]:
    """Paciente de destino dos Alertas e o instante da gravação na internação simulada."""
    pacientes = {p.id: p for p in demo.carregar_pacientes()}
    coluna_paciente, coluna_minuto = st.columns([3, 2])
    escolhido = coluna_paciente.selectbox(
        "Paciente",
        [AVULSO, *pacientes],
        format_func=lambda i: (
            "Sem leito (avulso)" if i == AVULSO else f"{pacientes[i].leito} · {pacientes[i].nome}"
        ),
        key=f"paciente_{chave}",
        help="Num Paciente de Demonstração, os Alertas entram na internação e na Fusão.",
    )
    if escolhido == AVULSO:
        coluna_minuto.caption("Alertas com o horário de agora.")
        return AVULSO, None
    paciente = pacientes[escolhido]
    padrao = getattr(paciente, minuto_padrao).inicio_min
    minuto = coluna_minuto.number_input(
        "Minuto da internação",
        min_value=0,
        max_value=600,
        value=int(padrao),
        step=5,
        key=f"minuto_{chave}_{escolhido}",
        help=f"Internação começa em {paciente.internacao_inicio:%H:%M}.",
    )
    inicio = paciente.em(minuto)
    coluna_minuto.caption(f"Gravação às {inicio:%H:%M} da internação.")
    return escolhido, inicio


def _mostrar_alertas(conn: sqlite3.Connection, alertas: list[Alerta], chave: str) -> None:
    st.markdown("**Alertas gerados**")
    if not alertas:
        st.caption("Nenhum Alerta disparado nesta análise.")
        return
    for alerta in alertas:
        with st.expander(f"{ICONE_SEVERIDADE[alerta.severidade]} {alerta.descricao}"):
            renderizar_evidencia(alerta)
    if st.button("Enviar Alertas à Central", type="primary", key=f"gravar_alertas_{chave}"):
        # Só os novos: regravar sobrescreveria o status de um Alerta já tratado pela equipe.
        novos = [a for a in alertas if db.obter(conn, a.id) is None]
        for alerta in novos:
            db.salvar(conn, alerta)
        st.success(f"{len(novos)} Alerta(s) novo(s) na Central de Alertas.")


# --- vídeo ----------------------------------------------------------------------------------


def _parametros(exercicio: Exercicio, chave: str) -> dict[str, float]:
    """Sliders da faixa esperada, só para os limiares que o exercício usa."""
    usados = [
        f.name for f in fields(exercicio) if f.name in PARAMETROS and getattr(exercicio, f.name)
    ]
    escolhidos: dict[str, float] = {}
    with st.expander("Faixa esperada (prescrição do fisioterapeuta)"):
        st.caption(f"Padrão do exercício {exercicio.rotulo.lower()}; ajuste para o paciente.")
        for nome in usados:
            rotulo, minimo, maximo, ajuda = PARAMETROS[nome]
            escolhidos[nome] = st.slider(
                rotulo,
                minimo,
                maximo,
                int(getattr(exercicio, nome)),
                key=f"{chave}_{exercicio.nome}_{nome}",
                help=ajuda,
            )
    return escolhidos


def _executar_analise(caminho: Path, opcoes: dict, **extras) -> None:
    with st.spinner("Estimando a pose quadro a quadro (YOLOv8n-pose, CPU)..."):
        try:
            resultado = analyze_video(
                caminho,
                exercicio=opcoes["exercicio"],
                vista=opcoes["vista"],
                paciente_id=opcoes["paciente_id"],
                inicio=opcoes["inicio"],
                parametros=opcoes["parametros"] or None,
                saida=PASTA_LABORATORIO,
                modelo=_modelo_pose(),
                modelo_objetos=_modelo_objetos() if opcoes["objetos"] else None,
                detectar_objetos=opcoes["objetos"],
                video_anotado=opcoes["video"],
                **extras,
            )
        except Exception as erro:  # vídeo corrompido, codec sem suporte no OpenCV...
            st.error(f"Falha ao analisar o vídeo: {erro}")
            return
    st.session_state["resultado_video"] = resultado


def _opcoes_video(cenario_padrao: str | None, vista_padrao: str, chave: str) -> dict:
    nomes = list(CENARIOS)
    indice = list(CENARIOS.values()).index(cenario_padrao)
    coluna_cenario, coluna_vista = st.columns([3, 2])
    rotulo = coluna_cenario.selectbox("Cenário", nomes, index=indice, key=f"cenario_{chave}")
    vista = coluna_vista.radio(
        "Câmera",
        list(Vista),
        index=list(Vista).index(Vista(vista_padrao)),
        format_func=ROTULO_VISTA.get,
        horizontal=True,
        key=f"vista_{chave}",
        help="Valgo e assimetria só são avaliados com a câmera de frente (ou de costas).",
    )
    exercicio = CENARIOS[rotulo]
    parametros = _parametros(EXERCICIOS[exercicio], chave) if exercicio else {}
    paciente_id, inicio = _seletor_paciente(f"video_{chave}", "video")
    marca_video, marca_objetos = st.columns(2)
    return {
        "exercicio": exercicio,
        "vista": str(vista),
        "parametros": parametros,
        "paciente_id": paciente_id,
        "inicio": inicio,
        "video": marca_video.checkbox(
            "Gerar vídeo com o esqueleto", value=True, key=f"anotado_{chave}"
        ),
        "objetos": marca_objetos.checkbox(
            "Detectar áreas críticas (YOLOv8)",
            value=exercicio is None,
            key=f"objetos_{chave}",
            help="Cama, cadeira, sofá e vaso sanitário perto da Queda.",
        ),
    }


def _video_de_exemplo() -> None:
    clipes = _clipes()
    if not clipes:
        st.info("Nenhum clipe baixado: `uv run python scripts/download_data.py video`.")
        return
    clipe = st.selectbox(
        "Clipe",
        clipes,
        format_func=lambda c: f"{c['id']} — {c['gabarito']}",
        key="clipe_exemplo",
    )
    st.caption(f"{clipe['autor']} · {clipe['licenca']} · [fonte]({clipe['pagina']})")
    opcoes = _opcoes_video(clipe["exercicio"], clipe["vista"], f"exemplo_{clipe['id']}")
    if st.button("Analisar vídeo", type="primary", key="analisar_exemplo"):
        _executar_analise(
            PASTA_VIDEO / clipe["arquivo"],
            opcoes,
            recorte=tuple(clipe["recorte"]) if clipe["recorte"] else None,
            clipe_id=f"exemplo-{clipe['id']}",
        )


def _video_enviado() -> None:
    arquivo = st.file_uploader(
        "Vídeo da Sessão de Fisioterapia ou do quarto (MP4, WebM, AVI, MOV, OGV...)",
        type=["mp4", "webm", "avi", "mov", "ogv", "mkv"],
        key="upload_video",
    )
    opcoes = _opcoes_video("agachamento", Vista.LATERAL, "upload")
    if arquivo is not None and st.button("Analisar vídeo", type="primary", key="analisar_video"):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / arquivo.name
            caminho.write_bytes(arquivo.getvalue())
            _executar_analise(caminho, opcoes)


def _video_da_camera() -> None:
    st.caption(
        "Abre a webcam do computador onde o app está rodando (demonstração local) e mostra o "
        "esqueleto ao vivo; ao fim, a gravação é analisada como uma Sessão de Fisioterapia."
    )
    controles, imagem = st.columns([2, 3], gap="large")
    with controles:
        coluna_camera, coluna_tempo = st.columns(2)
        indice = coluna_camera.number_input("Câmera", 0, 5, 0, key="camera_indice")
        segundos = coluna_tempo.slider("Duração (s)", 5, 30, 10, key="camera_segundos")
        opcoes = _opcoes_video("agachamento", Vista.FRONTAL, "camera")
        iniciar = st.button("📷 Iniciar captura", type="primary", key="camera_iniciar")
    with imagem:
        tela = st.empty()
        if not iniciar:
            tela.info("A imagem da câmera aparece aqui, com o esqueleto do YOLOv8-pose.")
            return
    destino = PASTA_LABORATORIO / f"camera-{datetime.now():%Y%m%d-%H%M%S}.avi"

    def _mostrar_quadro(quadro, _segundo: float) -> None:
        tela.image(quadro, channels="BGR", width="stretch")

    try:
        camera.capturar(
            destino,
            segundos=segundos,
            indice=int(indice),
            modelo=_modelo_pose(),
            ao_vivo=_mostrar_quadro,
        )
    except camera.CameraIndisponivel as erro:
        st.error(str(erro))
        return
    tela.empty()
    # A webcam rende ~8-15 fps em CPU: analisa todos os quadros gravados.
    _executar_analise(destino, opcoes, passo=1, clipe_id=destino.stem)


def _mostrar_video(conn: sqlite3.Connection, resultado: ResultadoVideo) -> None:
    exercicio = resultado.exercicio.rotulo if resultado.exercicio else "Quarto"
    st.caption(
        f"Resultado de: {resultado.arquivo} · {exercicio} · paciente {resultado.paciente_id}"
    )
    colunas = st.columns(4)
    colunas[0].metric("Repetições", len(resultado.repeticoes))
    colunas[1].metric("Desvios de Execução", len(resultado.desvios))
    colunas[2].metric("Quedas", len(resultado.quedas))
    colunas[3].metric(
        "CPU / s de vídeo",
        f"{resultado.fator_tempo_real:.2f}",
        help=f"{resultado.sequencia.segundos_processamento:.1f} s para "
        f"{resultado.sequencia.duracao:.1f} s de vídeo; abaixo de 1, dá para analisar ao vivo.",
    )

    if resultado.video is not None and resultado.video.is_file():
        st.markdown("**Vídeo com o esqueleto (YOLOv8-pose)**")
        st.video(str(resultado.video))

    if resultado.evidencias:
        st.markdown("**Evidências**")
        colunas = st.columns(3)
        for i, evidencia in enumerate(resultado.evidencias):
            colunas[i % 3].image(
                evidencia.caminho, caption=f"{evidencia.titulo} — {evidencia.t:.1f} s"
            )
    else:
        st.success("Nenhum Desvio nem Queda: execução dentro da faixa esperada.")

    if resultado.desvios:
        st.dataframe(
            [
                {
                    "Desvio": DESCRICAO_REGRA[d.regra],
                    "Repetição": d.repeticao,
                    "Instante (s)": round(d.t, 1),
                    "Medido (°)": round(d.valor),
                    "Limite (°)": round(d.limite),
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
    fonte = st.radio("Fonte", FONTES_VIDEO, horizontal=True, key="fonte_video")
    if fonte == "Clipe de exemplo":
        _video_de_exemplo()
    elif fonte == "Enviar arquivo":
        _video_enviado()
    else:
        _video_da_camera()

    resultado = st.session_state.get("resultado_video")
    if resultado is not None:
        st.divider()
        _mostrar_video(_conexao(), resultado)


# --- áudio ----------------------------------------------------------------------------------


def _formatar_sentimento(sentimento: str) -> str:
    icones = {"positive": "🙂", "neutral": "😐", "negative": "🙁", "mixed": "😕"}
    nomes = {"positive": "positivo", "neutral": "neutro", "negative": "negativo", "mixed": "misto"}
    return f"{icones.get(sentimento, '')} {nomes.get(sentimento, sentimento)}"


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


def _analisar_audio(nome: str, conteudo: bytes, paciente_id: str, inicio) -> None:
    load_dotenv()
    try:
        config = carregar_config(os.environ)
    except ConfigAzureIncompleta as erro:
        st.error(str(erro))
        return

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / nome
        caminho.write_bytes(conteudo)
        with st.spinner("Transcrevendo e analisando (Azure Speech + Language)..."):
            try:
                resultado = analyze_audio(caminho, config, paciente_id=paciente_id, inicio=inicio)
            except Exception as erro:  # Azure/ffmpeg podem falhar por motivos variados
                st.error(f"Falha ao analisar o áudio: {erro}")
                return

    st.session_state["resultado_audio"] = resultado


def _azure_configurado() -> bool:
    load_dotenv()
    try:
        carregar_config(os.environ)
    except ConfigAzureIncompleta:
        return False
    return True


def _secao_audio() -> None:
    if not _azure_configurado():
        st.warning(
            "Sem chaves do Azure no `.env`: a transcrição (Speech to Text) e os Termos Críticos "
            "(Text Analytics for Health) não rodam. Veja `docs/azure-setup.md`. As Consultas do "
            "PriMock57 já processadas aparecem na Central e no Painel pelo cache."
        )
    fonte = st.radio("Fonte", FONTES_AUDIO, horizontal=True, key="fonte_audio")
    nome, conteudo = None, None
    if fonte == "Enviar arquivo":
        arquivo = st.file_uploader(
            "Áudio da Consulta (WAV, MP3, M4A...)", type=None, key="upload_audio"
        )
        if arquivo is not None:
            nome, conteudo = arquivo.name, arquivo.getvalue()
    elif fonte == "Gravar pelo microfone":
        gravacao = st.audio_input("Fale como o paciente na Consulta", key="microfone_audio")
        if gravacao is not None:
            nome, conteudo = "microfone.wav", gravacao.getvalue()
    elif AUDIO_EXEMPLO.exists():
        st.audio(str(AUDIO_EXEMPLO))
        st.caption("Paciente relata falta de ar (inglês, como o PriMock57).")
        nome, conteudo = AUDIO_EXEMPLO.name, AUDIO_EXEMPLO.read_bytes()
    paciente_id, inicio = _seletor_paciente("audio", "consulta")
    if conteudo is not None and st.button("Analisar", type="primary", key="analisar_audio"):
        _analisar_audio(nome or "audio.wav", conteudo, paciente_id, inicio)

    resultado = st.session_state.get("resultado_audio")
    if resultado is None:
        return

    st.divider()
    st.caption(f"Resultado de: {resultado.arquivo}")
    _mostrar_transcricao(resultado)
    _mostrar_features_acusticas(resultado)
    _mostrar_alertas(_conexao(), resultado.alertas, "audio")


def pagina() -> None:
    cabecalho.barra(avancar=False)  # o relógio não anda durante uma análise sob demanda
    st.title("Laboratório de Análise")
    st.caption(
        "Análise sob demanda de vídeo (YOLOv8-pose) e áudio (Azure). Os Alertas vão para a "
        "Central e, num Paciente de Demonstração, para a Fusão."
    )
    modalidade = st.segmented_control(
        "Modalidade",
        ["🎥 Vídeo", "🎙️ Áudio"],
        default="🎥 Vídeo",
        key="modalidade_lab",
        label_visibility="collapsed",
    )
    if modalidade == "🎙️ Áudio":
        _secao_audio()
    else:
        _secao_video()
