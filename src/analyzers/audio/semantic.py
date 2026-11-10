"""Camada semântica do Analisador de Áudio (ESTRATEGIA.md, seção 5): Azure Speech to Text,
WER contra a transcrição manual do PriMock57, Text Analytics for Health → Termos Críticos e
Sentiment por fala do paciente.

Pipeline de uma Consulta (`processar_consulta`):

    1. `converter_wav`      garante WAV 16 kHz mono (pydub/ffmpeg)
    2. `transcrever`        Azure STT en-US, cacheado em JSON (ESTRATEGIA.md, seção 15)
    3. `calcular_wer`       contra `transcripts/<consulta>_patient.TextGrid` (jiwer)
    4. `extrair_entidades_saude`  TA for Health, cacheado
    5. `termos_criticos`    entidades × lista curada (`termos.py`) → Alerta
    6. `analisar_sentimento`  Sentiment por trecho do paciente, cacheado

A transcrição usada nos passos 4–6 é a do Azure (passo 2), não a manual: na operação real
(upload no Laboratório) não existe transcrição manual, só a do Azure.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import jiwer

from src.analyzers.audio import termos
from src.analyzers.audio.azure import AzureConfig
from src.analyzers.audio.cache import obter_ou_computar
from src.core.alerts import Alerta, Origem, Severidade, Tipo

IDIOMA = "en-US"
# Documentos por chamada (HTTP 400 acima disso): limites distintos por recurso do Language.
TAMANHO_LOTE_HEALTH = 25
TAMANHO_LOTE_SENTIMENT = 10

_NAMESPACE = uuid.UUID("9d6e8c2a-9b44-4a4d-9a7b-5a7b7f8e0a92")


@dataclass(frozen=True)
class Trecho:
    """Um trecho de fala, com os limites em segundos desde o início da Consulta."""

    inicio: float
    fim: float
    texto: str


@dataclass(frozen=True)
class EntidadeClinica:
    texto: str
    categoria: str
    confianca: float
    certeza: str | None = None  # "positive", "negative", "negativePossible"...
    associacao: str | None = None  # "subject" (o paciente) ou "other" (ex.: um familiar)

    @property
    def afirmada_sobre_o_paciente(self) -> bool:
        negada = self.certeza in ("negative", "negativePossible")
        return not negada and self.associacao != "other"


@dataclass(frozen=True)
class SentimentoTrecho:
    trecho: Trecho
    sentimento: str  # "positive" | "neutral" | "negative" | "mixed"
    positivo: float
    neutro: float
    negativo: float


@dataclass(frozen=True)
class ResultadoSemantico:
    consulta_id: str
    transcricao: list[Trecho]
    wer: float | None  # None quando não há transcrição manual para comparar
    wer_sem_hesitacoes: float | None
    alertas_termo_critico: list[Alerta]
    sentimentos: list[SentimentoTrecho]


# --- 1. conversão de áudio ---------------------------------------------------------------


def converter_wav(origem: Path, destino: Path) -> Path:
    """WAV 16 kHz mono; pula a conversão se `destino` já existe (idempotente)."""
    if destino.exists():
        return destino
    from pydub import AudioSegment

    audio = AudioSegment.from_file(origem)
    audio = audio.set_frame_rate(16_000).set_channels(1)
    destino.parent.mkdir(parents=True, exist_ok=True)
    audio.export(destino, format="wav")
    return destino


# --- 2. Azure Speech to Text --------------------------------------------------------------


def _reconhecer_continuo(
    caminho_wav: Path, config: AzureConfig, idioma: str = IDIOMA
) -> list[Trecho]:
    """Transcreve o arquivo inteiro (não só a primeira frase) via reconhecimento contínuo."""
    import azure.cognitiveservices.speech as speechsdk

    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key, region=config.speech_region
    )
    speech_config.speech_recognition_language = idioma
    audio_config = speechsdk.audio.AudioConfig(filename=str(caminho_wav))
    reconhecedor = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    trechos: list[Trecho] = []
    erros: list[str] = []
    concluido = threading.Event()

    def ao_reconhecer(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        resultado = evt.result
        if resultado.reason == speechsdk.ResultReason.RecognizedSpeech and resultado.text:
            inicio = resultado.offset / 10_000_000  # ticks de 100 ns -> segundos
            duracao = resultado.duration / 10_000_000
            trechos.append(Trecho(inicio=inicio, fim=inicio + duracao, texto=resultado.text))

    def ao_cancelar(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        detalhes = evt.result.cancellation_details
        if detalhes is not None and detalhes.reason == speechsdk.CancellationReason.Error:
            erros.append(detalhes.error_details)
        concluido.set()

    def ao_parar(evt: object) -> None:
        concluido.set()

    reconhecedor.recognized.connect(ao_reconhecer)
    reconhecedor.canceled.connect(ao_cancelar)
    reconhecedor.session_stopped.connect(ao_parar)

    reconhecedor.start_continuous_recognition_async().get()
    concluido.wait()
    reconhecedor.stop_continuous_recognition_async().get()

    if erros:
        raise RuntimeError(f"Azure STT cancelado: {erros[0]}")
    return sorted(trechos, key=lambda t: t.inicio)


def transcrever(
    caminho_wav: Path, config: AzureConfig, cache_dir: Path, idioma: str = IDIOMA
) -> list[Trecho]:
    """Transcreve com cache: reprocessar o mesmo WAV lê o JSON em vez de chamar o Azure."""
    caminho_cache = cache_dir / f"{caminho_wav.stem}.stt.json"
    return obter_ou_computar(
        caminho_cache,
        lambda: _reconhecer_continuo(caminho_wav, config, idioma),
        para_json=lambda trechos: [asdict(t) for t in trechos],
        de_json=lambda dados: [Trecho(**d) for d in dados],
    )


# --- 3. WER contra a transcrição manual (PriMock57) ---------------------------------------

_TAG_TEXTGRID = re.compile(r"<[A-Z/]+>")
_PONTUACAO = re.compile(r"[^\w\s']")
# A transcrição manual registra hesitações que o Azure quase sempre omite; elas pesam no WER
# sem afetar nenhum Termo Crítico.
_HESITACOES = re.compile(r"\b(?:um+|uh+|er+m*|mm+|hm+)\b")


def _normalizar(texto: str) -> str:
    sem_tags = _TAG_TEXTGRID.sub(" ", texto)
    sem_pontuacao = _PONTUACAO.sub(" ", sem_tags.lower())
    return " ".join(sem_pontuacao.split())


def ler_textgrid(caminho: Path) -> list[Trecho]:
    """Parser do formato "long text" do Praat TextGrid (ver transcripts/README.md do
    PriMock57): intervalos não vazios de uma única IntervalTier, em ordem."""
    conteudo = caminho.read_text(encoding="utf-8")
    padrao = re.compile(
        r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"', re.MULTILINE
    )
    trechos = [
        Trecho(inicio=float(inicio), fim=float(fim), texto=texto.strip())
        for inicio, fim, texto in padrao.findall(conteudo)
        if texto.strip()
    ]
    return trechos


def calcular_wer(
    hipotese: list[Trecho], referencia: list[Trecho], *, sem_hesitacoes: bool = False
) -> float:
    """WER (jiwer) entre a transcrição do Azure e a manual, concatenadas em ordem e
    normalizadas (minúsculas, sem pontuação, sem as tags `<UNSURE>`/`<UNIN/>` do PriMock57).

    `sem_hesitacoes` também remove "um", "uh" e afins das duas transcrições.
    """
    texto_hipotese = _normalizar(" ".join(t.texto for t in hipotese))
    texto_referencia = _normalizar(" ".join(t.texto for t in referencia))
    if sem_hesitacoes:
        texto_hipotese = " ".join(_HESITACOES.sub(" ", texto_hipotese).split())
        texto_referencia = " ".join(_HESITACOES.sub(" ", texto_referencia).split())
    if not texto_referencia:
        return float("nan")
    return float(jiwer.wer(texto_referencia, texto_hipotese))


# --- 4. Text Analytics for Health → Termos Críticos ----------------------------------------


def _cliente_language(config: AzureConfig):
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    return TextAnalyticsClient(
        endpoint=config.language_endpoint, credential=AzureKeyCredential(config.language_key)
    )


def _em_lotes(itens: list[Trecho], tamanho: int) -> list[list[Trecho]]:
    return [itens[i : i + tamanho] for i in range(0, len(itens), tamanho)]


def _exigir_sucesso(documento: object, tipo_erro: type) -> None:
    """Falha em vez de devolver um resultado vazio: o que sai daqui vai para o cache, e uma
    falha cacheada como "sem Termos Críticos" ou "neutro" nunca mais seria reprocessada."""
    if isinstance(documento, tipo_erro):
        raise RuntimeError(f"Azure Language recusou um documento: {documento.error}")  # type: ignore[attr-defined]


def _extrair_entidades_azure(
    trechos: list[Trecho], config: AzureConfig
) -> list[list[EntidadeClinica]]:
    from azure.ai.textanalytics import DocumentError

    resultado: list[list[EntidadeClinica]] = []
    with _cliente_language(config) as cliente:
        for lote in _em_lotes(trechos, TAMANHO_LOTE_HEALTH):
            poller = cliente.begin_analyze_healthcare_entities(
                [t.texto for t in lote], language="en"
            )
            for documento in poller.result():
                _exigir_sucesso(documento, DocumentError)
                resultado.append(
                    [
                        EntidadeClinica(
                            texto=e.text,
                            categoria=e.category,
                            confianca=e.confidence_score,
                            certeza=e.assertion.certainty if e.assertion else None,
                            associacao=e.assertion.association if e.assertion else None,
                        )
                        for e in documento.entities
                    ]
                )
    return resultado


def extrair_entidades_saude(
    trechos: list[Trecho], config: AzureConfig, cache_dir: Path, consulta_id: str
) -> list[list[EntidadeClinica]]:
    """Entidades clínicas por trecho, na mesma ordem de `trechos`; cacheado em JSON."""
    caminho_cache = cache_dir / f"{consulta_id}.ta_health_v2.json"  # v2: com certeza e associação
    return obter_ou_computar(
        caminho_cache,
        lambda: _extrair_entidades_azure(trechos, config),
        para_json=lambda listas: [[asdict(e) for e in lista] for lista in listas],
        de_json=lambda dados: [[EntidadeClinica(**e) for e in lista] for lista in dados],
    )


def _alerta_termo_critico(
    paciente_id: str,
    consulta_id: str,
    trecho: Trecho,
    termo: termos.TermoCritico,
    inicio: datetime,
) -> Alerta:
    evidencia: dict[str, object] = {
        "trecho": trecho.texto,
        "inicio": trecho.inicio,
        "entidade": termo.entidade_categoria,
        "termo": termo.termo,
        "arquivo": consulta_id,
    }
    if termo.categoria is not None:
        evidencia["categoria"] = termo.categoria
    severidade = Severidade.ALTA if termo.categoria is not None else Severidade.MEDIA
    chave = f"{paciente_id}|{consulta_id}|{trecho.inicio}|{termo.termo}"
    return Alerta(
        id=str(uuid.uuid5(_NAMESPACE, chave)),
        paciente_id=paciente_id,
        origem=Origem.AUDIO,
        tipo=Tipo.TERMO_CRITICO,
        severidade=severidade,
        descricao=f'Paciente menciona termo crítico ({termo.termo}): "{trecho.texto}"',
        evidencia=evidencia,
        detectado_em=inicio + timedelta(seconds=trecho.inicio),
    )


def termos_criticos(
    trechos: list[Trecho],
    entidades_por_trecho: list[list[EntidadeClinica]],
    paciente_id: str,
    consulta_id: str,
    inicio: datetime,
) -> list[Alerta]:
    """Um Alerta por (trecho, termo curado encontrado), sem repetir o par dentro do trecho; um
    trecho pode gerar Alertas de termos diferentes.

    `inicio` é o instante (simulado ou real) do começo da Consulta; cada Alerta soma o
    deslocamento do trecho dentro dela.
    """
    alertas: list[Alerta] = []
    for trecho, entidades in zip(trechos, entidades_por_trecho, strict=True):
        termos_do_trecho: set[str] = set()
        for entidade in entidades:
            if entidade.categoria not in termos.CATEGORIAS_AZURE_RELEVANTES:
                continue
            if not entidade.afirmada_sobre_o_paciente:
                continue  # "no fever at all", "my child was vomiting"
            termo = termos.localizar(entidade.texto, entidade.categoria)
            if termo is None or termo.termo in termos_do_trecho:
                continue
            if not termos.excluido_pelo_contexto(termo.termo, trecho.texto):
                termos_do_trecho.add(termo.termo)
                alertas.append(
                    _alerta_termo_critico(paciente_id, consulta_id, trecho, termo, inicio)
                )
    return alertas


# --- 5. Sentiment por fala do paciente -----------------------------------------------------


def _analisar_sentimento_azure(trechos: list[Trecho], config: AzureConfig) -> list[dict]:
    from azure.ai.textanalytics import DocumentError

    resultado: list[dict] = []
    with _cliente_language(config) as cliente:
        for lote in _em_lotes(trechos, TAMANHO_LOTE_SENTIMENT):
            documentos = cliente.analyze_sentiment([t.texto for t in lote], language="en")
            for documento in documentos:
                _exigir_sucesso(documento, DocumentError)
                scores = documento.confidence_scores
                resultado.append(
                    {
                        "sentimento": documento.sentiment,
                        "positivo": scores.positive,
                        "neutro": scores.neutral,
                        "negativo": scores.negative,
                    }
                )
    return resultado


def analisar_sentimento(
    trechos: list[Trecho], config: AzureConfig, cache_dir: Path, consulta_id: str
) -> list[SentimentoTrecho]:
    caminho_cache = cache_dir / f"{consulta_id}.sentiment.json"
    brutos = obter_ou_computar(caminho_cache, lambda: _analisar_sentimento_azure(trechos, config))
    return [
        SentimentoTrecho(
            trecho=trecho,
            sentimento=bruto["sentimento"],
            positivo=bruto["positivo"],
            neutro=bruto["neutro"],
            negativo=bruto["negativo"],
        )
        for trecho, bruto in zip(trechos, brutos, strict=True)
    ]


# --- orquestração ---------------------------------------------------------------------------


def processar_consulta(
    consulta_id: str,
    pasta_primock57: Path,
    config: AzureConfig,
    cache_dir: Path,
    *,
    paciente_id: str = "avulso",
    inicio: datetime | None = None,
) -> ResultadoSemantico:
    """Pipeline completo de uma Consulta do PriMock57 (canal do paciente).

    `pasta_primock57` é `data/audio/primock57` (saída de `scripts/download_data.py primock57`):
    espera `audio/<consulta_id>_patient.wav` e, se existir,
    `transcripts/<consulta_id>_patient.TextGrid`.
    `inicio` é o instante do começo da Consulta (padrão: agora, como no upload do Laboratório);
    um Paciente de Demonstração passa o horário simulado da sua internação.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_original = pasta_primock57 / "audio" / f"{consulta_id}_patient.wav"
    audio_16k = cache_dir / f"{consulta_id}_patient_16k.wav"
    audio_pronto = converter_wav(audio_original, audio_16k)

    transcricao = transcrever(audio_pronto, config, cache_dir)

    wer: float | None = None
    wer_sem_hesitacoes: float | None = None
    textgrid = pasta_primock57 / "transcripts" / f"{consulta_id}_patient.TextGrid"
    if textgrid.exists():
        referencia = ler_textgrid(textgrid)
        wer = calcular_wer(transcricao, referencia)
        wer_sem_hesitacoes = calcular_wer(transcricao, referencia, sem_hesitacoes=True)

    entidades = extrair_entidades_saude(transcricao, config, cache_dir, consulta_id)
    alertas = termos_criticos(
        transcricao, entidades, paciente_id, consulta_id, inicio or datetime.now()
    )
    sentimentos = analisar_sentimento(transcricao, config, cache_dir, consulta_id)

    return ResultadoSemantico(
        consulta_id=consulta_id,
        transcricao=transcricao,
        wer=wer,
        wer_sem_hesitacoes=wer_sem_hesitacoes,
        alertas_termo_critico=alertas,
        sentimentos=sentimentos,
    )
