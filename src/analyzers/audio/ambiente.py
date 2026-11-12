"""Sons do leito: alarme de equipamento, grito, gemido de dor e o paciente chamando a equipe.

Dois modelos locais, em CPU:

- **AST** (Audio Spectrogram Transformer) treinado no **AudioSet** — o dataset de áudio que o
  PDF sugere — classifica a janela de som em 527 classes. As classes de interesse são agrupadas
  em eventos clínicos (`EVENTOS`).
- **Whisper tiny** transcreve, em português, só as janelas que o AST diz serem fala ou grito,
  e palavras de chamado ("enfermeira", "socorro", "doutor"...) viram o evento `chamado`.

A Alteração Vocal (fadiga/dispneia na voz da Consulta) continua em `acoustic.py`; aqui é o
ambiente do quarto, monitorado continuamente (ADR 0003).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from src.core.alerts import Alerta, Origem, Severidade, Tipo

TAXA = 16_000
MODELO_EVENTOS = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODELO_FALA = "openai/whisper-tiny"

# evento → (classes do AudioSet, probabilidade mínima, severidade, descrição)
EVENTOS: dict[str, tuple[frozenset[str], float, Severidade, str]] = {
    "alarme": (
        frozenset(
            {
                "Beep, bleep",
                "Alarm",
                "Alarm clock",
                "Buzzer",
                "Smoke detector, smoke alarm",
                "Fire alarm",
                "Telephone bell ringing",
                "Busy signal",
                "Dial tone",
            }
        ),
        0.20,
        Severidade.MEDIA,
        "Alarme de equipamento soando no leito",
    ),
    "grito": (
        frozenset({"Screaming", "Shout", "Yell", "Bellow", "Battle cry"}),
        0.20,
        Severidade.ALTA,
        "Paciente gritando",
    ),
    "gemido": (
        frozenset({"Groan", "Whimper", "Wail, moan", "Crying, sobbing", "Grunt", "Sigh"}),
        0.15,
        Severidade.MEDIA,
        "Possível dor: gemido ou choro do paciente",
    ),
}
CLASSES_FALA = frozenset(
    {
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Shout",
        "Yell",
        "Screaming",
        "Conversation",
    }
)
LIMIAR_FALA = 0.15
RMS_SILENCIO = 0.0008  # abaixo disso a janela nem é processada (detecção de atividade)
RMS_ALVO = 0.05  # volume normalizado antes do modelo: microfone baixo não vira 'Door'
# Radicais, não palavras inteiras: o Whisper tiny erra a grafia de fala gritada ou cortada
# ("infirmeira", "socorr..."), e o que importa é o pedido de ajuda.
PALAVRAS_CHAMADO = re.compile(
    r"(enferm|infirm|socorr|ajud|doutor|medic|chama a|alguem|help|nurse|doctor)"
)


@dataclass(frozen=True)
class EventoSonoro:
    evento: str  # "alarme", "grito", "gemido" ou "chamado"
    confianca: float
    classes: tuple[str, ...]  # top classes do AudioSet na janela
    transcricao: str = ""


_COMPARTILHADO: dict[str, Any] = {}
_TRAVA = __import__("threading").Lock()
_PRONTO = __import__("threading").Event()


def pre_carregar() -> None:
    """Carrega os modelos de som em segundo plano (uma vez por processo), para que ligar o
    microfone do leito não trave a tela esperando ~2 min de carregamento."""
    import threading

    def carregar() -> None:
        with _TRAVA:
            if "classificador" not in _COMPARTILHADO:
                try:
                    _COMPARTILHADO["classificador"] = Classificador()
                except Exception as erro:  # sem rede/modelo: o microfone mostra o erro
                    _COMPARTILHADO["erro"] = str(erro)
        _PRONTO.set()

    if not _PRONTO.is_set() and not _TRAVA.locked():
        threading.Thread(target=carregar, daemon=True, name="modelos-de-som").start()


def classificador_pronto() -> bool:
    return _PRONTO.is_set() and "classificador" in _COMPARTILHADO


def classificador(timeout: float | None = None) -> Classificador:
    """O classificador compartilhado; espera o carregamento em segundo plano terminar."""
    pre_carregar()
    _PRONTO.wait(timeout)
    if "classificador" not in _COMPARTILHADO:
        raise RuntimeError(_COMPARTILHADO.get("erro", "modelos de som ainda carregando"))
    return _COMPARTILHADO["classificador"]


class Classificador:
    """Carrega os dois modelos uma vez e classifica janelas de áudio mono a 16 kHz."""

    def __init__(self, com_fala: bool = True) -> None:
        from transformers import pipeline  # import tardio: carregar o transformers é lento

        self._eventos = pipeline("audio-classification", model=MODELO_EVENTOS, device="cpu")
        self._fala: Any = None
        if com_fala:
            self._fala = pipeline("automatic-speech-recognition", model=MODELO_FALA, device="cpu")

    def classes(self, audio: np.ndarray, top: int = 10) -> list[tuple[str, float]]:
        saida = self._eventos({"raw": audio.astype(np.float32), "sampling_rate": TAXA}, top_k=top)
        return [(item["label"], float(item["score"])) for item in saida]

    def transcrever(self, audio: np.ndarray) -> str:
        if self._fala is None:
            return ""
        saida = self._fala(
            {"raw": audio.astype(np.float32), "sampling_rate": TAXA},
            generate_kwargs={"language": "portuguese", "task": "transcribe", "max_new_tokens": 32},
        )
        return str(saida.get("text", "")).strip()

    def classificar(self, audio: np.ndarray) -> list[EventoSonoro]:
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < RMS_SILENCIO:
            return []  # processamento parcial: janela sem som não passa pelos modelos
        audio = np.clip(audio * (RMS_ALVO / rms), -1.0, 1.0)  # ganho automático
        top = self.classes(audio)
        return eventos_de(top, self.transcrever if self._precisa_fala(top) else None, audio)

    @staticmethod
    def _precisa_fala(top: list[tuple[str, float]]) -> bool:
        return sum(p for nome, p in top if nome in CLASSES_FALA) >= LIMIAR_FALA


def _sem_acento(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def eventos_de(
    top: list[tuple[str, float]], transcrever: Any = None, audio: np.ndarray | None = None
) -> list[EventoSonoro]:
    """Eventos clínicos a partir das classes do AudioSet (e da transcrição, se houver fala)."""
    nomes = tuple(nome for nome, _ in top[:5])
    eventos = []
    for evento, (classes, limiar, _, _) in EVENTOS.items():
        confianca = sum(p for nome, p in top if nome in classes)
        if confianca >= limiar:
            eventos.append(EventoSonoro(evento, round(confianca, 3), nomes))
    if transcrever is not None and audio is not None:
        texto = transcrever(audio)
        if PALAVRAS_CHAMADO.search(_sem_acento(texto)):
            confianca = sum(p for nome, p in top if nome in CLASSES_FALA)
            eventos.append(EventoSonoro("chamado", round(confianca, 3), nomes, texto))
    return eventos


SEVERIDADE_CHAMADO = Severidade.ALTA
DESCRICAO_CHAMADO = "Paciente chamando a equipe"


def alerta(
    evento: EventoSonoro,
    paciente_id: str,
    detectado_em: datetime,
    arquivo: str | None = None,
    identificador: str = "",
) -> Alerta:
    if evento.evento == "chamado":
        severidade, descricao = SEVERIDADE_CHAMADO, DESCRICAO_CHAMADO
    else:
        _, _, severidade, descricao = EVENTOS[evento.evento]
    if evento.transcricao:
        descricao += f': "{evento.transcricao[:80]}"'
    evidencia: dict[str, Any] = {
        "evento": evento.evento,
        "confianca": evento.confianca,
        "classes_audioset": list(evento.classes),
        "modelo": MODELO_EVENTOS,
    }
    if evento.transcricao:
        evidencia["trecho"] = evento.transcricao
    if arquivo:
        evidencia["arquivo"] = arquivo
    import uuid

    return Alerta(
        id=str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"som|{paciente_id}|{evento.evento}|{identificador}")
        ),
        paciente_id=paciente_id,
        origem=Origem.AUDIO,
        tipo=Tipo.EVENTO_SONORO,
        severidade=severidade,
        descricao=descricao,
        evidencia=evidencia,
        detectado_em=detectado_em,
    )
