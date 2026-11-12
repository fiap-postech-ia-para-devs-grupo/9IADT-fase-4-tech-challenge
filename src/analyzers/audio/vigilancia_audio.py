"""Microfone do leito: vigilância contínua de sons (alarme, grito, gemido, chamado da equipe).

Uma `VigilanciaAudio` por leito roda numa thread: lê o microfone do computador (ou um áudio
gravado, tocado em tempo real como se fosse o quarto), a cada `PASSO_S` segundos classifica os
últimos `JANELA_S` segundos com o AST/AudioSet (+ Whisper para o chamado) e grava um Alerta
`evento_sonoro` **para o paciente do leito**, com o trecho em WAV como Evidência. O mesmo evento
não se repete antes de `INTERVALO_REPETICAO_S`.
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.analyzers.audio import ambiente
from src.core import db

_RAIZ = Path(__file__).resolve().parents[3]
PASTA_EVIDENCIA = _RAIZ / "results" / "audio" / "leito"
JANELA_S = 3.0
PASSO_S = 2.0
INTERVALO_REPETICAO_S = 30.0
MICROFONE = "microfone"


@dataclass
class EstadoAudio:
    ativo: bool = False
    erro: str | None = None
    nivel: float = 0.0  # RMS da última janela, para o medidor da tela
    ultimas_classes: list[tuple[str, float]] = field(default_factory=list)
    eventos: list[tuple[datetime, str, str]] = field(default_factory=list)  # (quando, evento, id)
    janelas: int = 0


class VigilanciaAudio:
    def __init__(
        self,
        paciente_id: str,
        fonte: str,  # MICROFONE ou caminho de arquivo de áudio
        *,
        relogio: datetime,
        classificador: ambiente.Classificador,
        caminho_db: Path = db.CAMINHO_PADRAO,
    ) -> None:
        self.paciente_id = paciente_id
        self.fonte = fonte
        self.relogio = relogio  # horário do plantão; a tela o atualiza
        self.classificador = classificador
        self.caminho_db = caminho_db
        self.estado = EstadoAudio()
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None

    def iniciar(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._parar.clear()
        self.estado = EstadoAudio(ativo=True)
        self._thread = threading.Thread(
            target=self._rodar, daemon=True, name=f"audio-{self.paciente_id}"
        )
        self._thread.start()

    def parar(self) -> None:
        self._parar.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.estado.ativo = False

    def _rodar(self) -> None:
        try:
            if self.fonte == MICROFONE:
                self._laco(self._blocos_microfone())
            else:
                self._laco(self._blocos_arquivo())
        except Exception as erro:  # dispositivo ou modelo: a thread não pode morrer calada
            self.estado.erro = str(erro)
        finally:
            self.estado.ativo = False

    def _blocos_microfone(self):
        import sounddevice as sd

        fila: queue.Queue[np.ndarray] = queue.Queue()
        with sd.InputStream(
            samplerate=ambiente.TAXA,
            channels=1,
            dtype="float32",
            blocksize=int(ambiente.TAXA * 0.25),
            callback=lambda dados, *_: fila.put(dados[:, 0].copy()),
        ):
            while not self._parar.is_set():
                try:
                    yield fila.get(timeout=1)
                except queue.Empty:
                    continue

    def _blocos_arquivo(self):
        import librosa

        sinal, _ = librosa.load(self.fonte, sr=ambiente.TAXA, mono=True)
        bloco = int(ambiente.TAXA * 0.25)
        while not self._parar.is_set():  # em loop, como o som contínuo do quarto
            for inicio in range(0, len(sinal), bloco):
                if self._parar.is_set():
                    return
                time.sleep(0.25)  # tempo real
                yield sinal[inicio : inicio + bloco]
            yield np.zeros(ambiente.TAXA, dtype=np.float32)  # pausa entre repetições

    def _laco(self, blocos) -> None:
        janela: deque[np.ndarray] = deque()
        amostras = 0
        ultimo_passo = time.perf_counter()
        ultimos: dict[str, float] = {}
        for bloco in blocos:
            janela.append(bloco)
            amostras += len(bloco)
            while amostras - len(janela[0]) >= JANELA_S * ambiente.TAXA:
                amostras -= len(janela.popleft())
            if time.perf_counter() - ultimo_passo < PASSO_S:
                continue
            ultimo_passo = time.perf_counter()
            audio = np.concatenate(list(janela))
            self.estado.nivel = float(np.sqrt(np.mean(audio**2)))
            self.estado.janelas += 1
            eventos = self.classificador.classificar(audio)
            if self.estado.nivel >= 0.003:
                self.estado.ultimas_classes = self.classificador.classes(audio, top=3)
            for evento in eventos:
                agora = time.perf_counter()
                if agora - ultimos.get(evento.evento, -1e9) < INTERVALO_REPETICAO_S:
                    continue
                ultimos[evento.evento] = agora
                self._emitir(evento, audio)

    def _emitir(self, evento: ambiente.EventoSonoro, audio: np.ndarray) -> None:
        detectado = self.relogio
        marca = f"{datetime.now():%Y%m%d_%H%M%S}"
        destino = PASTA_EVIDENCIA / self.paciente_id / f"{evento.evento}_{marca}.wav"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destino), "wb") as saida:
            saida.setnchannels(1)
            saida.setsampwidth(2)
            saida.setframerate(ambiente.TAXA)
            saida.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        alerta = ambiente.alerta(
            evento,
            self.paciente_id,
            detectado,
            destino.relative_to(_RAIZ).as_posix(),
            identificador=marca,
        )
        conn = db.conectar(self.caminho_db)  # conexão própria: esta é outra thread
        try:
            db.salvar(conn, alerta)
        finally:
            conn.close()
        self.estado.eventos.append((detectado, evento.evento, alerta.id))
