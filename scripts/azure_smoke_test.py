"""Smoke test dos recursos Azure F0: Speech to Text, Text Analytics for Health e Sentiment.

Lê as chaves do `.env` (ver `.env.example`) e faz uma chamada mínima a cada serviço. Sem
`--audio`, sintetiza uma frase curta com o próprio Speech (TTS) e a transcreve de volta, para
não depender de dataset baixado. Custo de cota: ~10 s de STT e 2 registros de Language.

    uv run python -m scripts.azure_smoke_test
    uv run python -m scripts.azure_smoke_test --audio data/audio/primock57/<arquivo>.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import DocumentError, TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from src.analyzers.audio.azure import AzureConfig, ConfigAzureIncompleta, carregar_config

FRASE = "I have been feeling short of breath and I have chest pain since yesterday."
TEXTO_CLINICO = "Patient reports dyspnea and chest pain. Started on 81 mg aspirin daily."
TEXTO_SENTIMENTO = "I am really worried, I can barely breathe at night."


def _detalhe_falha(
    resultado: speechsdk.SpeechSynthesisResult | speechsdk.SpeechRecognitionResult | None,
) -> str:
    if resultado is None:
        return "sem resposta"
    if resultado.reason == speechsdk.ResultReason.Canceled:
        return resultado.cancellation_details.error_details
    return str(resultado.reason)


def testar_stt(config: AzureConfig, audio: Path | None) -> str:
    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key, region=config.speech_region
    )
    speech_config.speech_recognition_language = "en-US"
    with tempfile.TemporaryDirectory() as tmp:
        if audio is None:
            audio = Path(tmp) / "frase.wav"
            sintetizador = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=speechsdk.audio.AudioOutputConfig(filename=str(audio)),
            )
            sintese = sintetizador.speak_text_async(FRASE).get()
            concluida = speechsdk.ResultReason.SynthesizingAudioCompleted
            if sintese is None or sintese.reason != concluida:
                raise RuntimeError(f"TTS falhou: {_detalhe_falha(sintese)}")
            del sintetizador  # libera o arquivo WAV antes de reconhecer

        reconhecedor = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=speechsdk.audio.AudioConfig(filename=str(audio)),
        )
        resultado = reconhecedor.recognize_once_async().get()
        if resultado is None or resultado.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise RuntimeError(f"STT falhou: {_detalhe_falha(resultado)}")
        return f"transcrição: {resultado.text!r}"


def cliente_language(config: AzureConfig) -> TextAnalyticsClient:
    return TextAnalyticsClient(
        endpoint=config.language_endpoint, credential=AzureKeyCredential(config.language_key)
    )


def testar_health(config: AzureConfig) -> str:
    with cliente_language(config) as cliente:
        poller = cliente.begin_analyze_healthcare_entities([TEXTO_CLINICO], language="en")
        documento = next(iter(poller.result()))
        if isinstance(documento, DocumentError):
            raise RuntimeError(f"TA for Health falhou: {documento.error}")
        entidades = ", ".join(f"{e.text} ({e.category})" for e in documento.entities)
        return f"entidades: {entidades}"


def testar_sentimento(config: AzureConfig) -> str:
    with cliente_language(config) as cliente:
        documento = cliente.analyze_sentiment([TEXTO_SENTIMENTO], language="en")[0]
        if isinstance(documento, DocumentError):
            raise RuntimeError(f"Sentiment falhou: {documento.error}")
        scores = documento.confidence_scores
        return (
            f"{documento.sentiment} (pos={scores.positive:.2f} "
            f"neu={scores.neutral:.2f} neg={scores.negative:.2f})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test dos recursos Azure F0")
    parser.add_argument("--audio", type=Path, help="WAV 16 kHz mono para o STT (opcional)")
    args = parser.parse_args()

    load_dotenv()
    try:
        config = carregar_config(os.environ)
    except ConfigAzureIncompleta as erro:
        print(erro, file=sys.stderr)
        return 2

    print(f"Speech: região {config.speech_region} · Language: {config.language_endpoint}")
    testes: list[tuple[str, Callable[[], str]]] = [
        ("Speech to Text", lambda: testar_stt(config, args.audio)),
        ("TA for Health", lambda: testar_health(config)),
        ("Sentiment", lambda: testar_sentimento(config)),
    ]
    falhas = 0
    for nome, testar in testes:
        try:
            print(f"[OK]    {nome}: {testar()}")
        except Exception as erro:  # o smoke test quer ver todas as falhas, não só a primeira
            falhas += 1
            print(f"[FALHA] {nome}: {erro}", file=sys.stderr)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
