"""Câmera do leito: vigilância contínua de Queda com YOLOv8-pose (Monitoramento Contínuo).

Uma `Vigilancia` por leito roda numa thread própria: lê a câmera (webcam ou um clipe gravado,
tocado em tempo real como se fosse a câmera), estima a pose em CPU, guarda os últimos segundos
e roda o detector de Queda (`fall.py`) sobre essa janela. Cada Queda nova vira Alerta no
SQLite para o paciente do leito, com frame de Evidência e as áreas críticas próximas (YOLOv8 de
objetos), e daí segue para a Central e para a Fusão como qualquer outro Alerta.

O relógio do Alerta é o do plantão: a tela atualiza `relogio` a cada quadro mostrado, e o
Alerta sai com esse horário (o relógio do plantão pode estar pausado enquanto a câmera roda).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.analyzers.video import fall, objetos
from src.analyzers.video import pose as p
from src.analyzers.video.lab import alerta_queda
from src.analyzers.video.pose import Pose, Recorte, SequenciaPose
from src.core import db
from src.core.alerts import Tipo

_RAIZ = Path(__file__).resolve().parents[3]
PASTA_EVIDENCIA = _RAIZ / "results" / "video" / "camera"
JANELA_S = 8.0  # segundos de pose guardados para o detector de Queda
PASSO = 2  # processamento parcial: pose a cada N quadros; todos os quadros vão para a tela
TAMANHO_POSE = 480  # resolução da inferência ao vivo: corpo inteiro longe da câmera ainda aparece
LARGURA_WEBCAM, ALTURA_WEBCAM = 1280, 720  # campo de visão maior que o 640x480 padrão


@dataclass
class Estado:
    ativo: bool = False
    erro: str | None = None
    quadros: int = 0
    fps: float = 0.0
    ultimo_quadro: np.ndarray | None = None
    quedas: list[str] = field(default_factory=list)  # ids dos Alertas gerados
    fim_da_gravacao: bool = False
    leitura: str = ""  # o que o detector vê agora: postura, eixo da cabeça, caixa


def _leitura(pose: Pose) -> str:
    """Resumo legível do que o detector de Queda está vendo neste quadro."""
    if pose.caixa is None:
        return "ninguém no quadro"
    x0, y0, x1, y1 = pose.caixa
    razao = (x1 - x0) / max(y1 - y0, 1e-6)
    eixo = fall.eixo_cabeca(pose)
    if eixo is not None and fall.EIXO_DEITADO <= eixo <= fall.EIXO_CURVADO:
        postura = "deitado de lado"
    elif razao >= fall.RAZAO_NO_CHAO:
        postura = "corpo na horizontal"
    elif razao < fall.RAZAO_EM_PE:
        postura = "em pé (corpo inteiro)"
    else:
        postura = "sentado / parcial"
    eixo_txt = f"{eixo:.0f}°" if eixo is not None else "—"
    return f"{postura} · eixo da cabeça {eixo_txt} · caixa largura/altura {razao:.2f}"


class Vigilancia:
    def __init__(
        self,
        paciente_id: str,
        fonte: str | int,
        *,
        inicio_plantao: datetime,
        caminho_db: Path = db.CAMINHO_PADRAO,
        recorte: Recorte | None = None,
        modelo_pose: Any = None,
        modelo_objetos: Any = None,
    ) -> None:
        self.paciente_id = paciente_id
        self.fonte = fonte
        self.relogio = inicio_plantao  # horário do plantão; a tela o atualiza
        self.caminho_db = caminho_db
        self.recorte = recorte
        self.modelo_pose = modelo_pose
        self.modelo_objetos = modelo_objetos
        self.estado = Estado()
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ao_vivo(self) -> bool:
        return isinstance(self.fonte, int)

    def iniciar(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._parar.clear()
        self.estado = Estado(ativo=True)
        self._thread = threading.Thread(target=self._rodar, daemon=True, name=self.paciente_id)
        self._thread.start()

    def parar(self) -> None:
        self._parar.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.estado.ativo = False

    def _abrir(self) -> cv2.VideoCapture:
        if self.ao_vivo:
            import sys

            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            captura = cv2.VideoCapture(int(self.fonte), backend)
            captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_WEBCAM)
            captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_WEBCAM)
            return captura
        return cv2.VideoCapture(str(self.fonte))

    def _rodar(self) -> None:
        try:
            self._laco()
        except Exception as erro:  # câmera ou modelo: a thread não pode morrer calada
            self.estado.erro = str(erro)
        finally:
            self.estado.ativo = False

    def _laco(self) -> None:
        modelo = self.modelo_pose or p.carregar_modelo()
        captura = self._abrir()
        if not captura.isOpened():
            raise RuntimeError(
                "Não foi possível abrir a câmera: confira se ela existe, se outro programa a "
                "usa e se o app roda no mesmo computador da câmera."
            )
        fps_fonte = captura.get(cv2.CAP_PROP_FPS) or 15.0
        janela: deque[tuple[Pose, np.ndarray]] = deque()
        vistas: set[float] = set()
        inicio = time.perf_counter()
        indice = 0
        try:
            while not self._parar.is_set():
                ok, quadro = captura.read()
                if not ok:
                    if self.ao_vivo:
                        raise RuntimeError("A câmera parou de enviar imagem.")
                    self.estado.fim_da_gravacao = True  # clipe: toca uma vez, como o evento real
                    return
                if self.recorte is not None:
                    x0, y0, x1, y1 = self.recorte
                    quadro = quadro[y0:y1, x0:x1]
                decorrido = time.perf_counter() - inicio
                if not self.ao_vivo:  # clipe: toca no ritmo real
                    atraso = indice / fps_fonte - decorrido
                    if atraso > 0:
                        time.sleep(atraso)
                    decorrido = indice / fps_fonte
                if indice % PASSO == 0:
                    resultado = modelo(quadro, verbose=False, device="cpu", imgsz=TAMANHO_POSE)[0]
                    keypoints, caixa = p.pessoa_principal(resultado)
                    ultima = Pose(indice, decorrido, keypoints, caixa)
                    self.estado.leitura = _leitura(ultima)
                    if self.ao_vivo:
                        self._registrar(ultima, quadro.shape[0])
                    janela.append((ultima, quadro))
                    while janela and decorrido - janela[0][0].t > JANELA_S:
                        janela.popleft()
                    self._procurar_queda(janela, vistas)
                # Quadro intermediário: mostra a imagem nova com o último esqueleto (fluidez).
                if janela:
                    self.estado.ultimo_quadro = p.anotar(quadro, janela[-1][0])
                else:
                    self.estado.ultimo_quadro = quadro
                indice += 1
                self.estado.quadros = indice
                self.estado.fps = indice / max(time.perf_counter() - inicio, 1e-6)
        finally:
            captura.release()

    def _registrar(self, pose: Pose, altura_quadro: int) -> None:
        """Trilha quadro a quadro da webcam (para calibrar a Queda em câmera próxima)."""
        destino = PASTA_EVIDENCIA / self.paciente_id / "trilha.csv"
        destino.parent.mkdir(parents=True, exist_ok=True)
        ombros = fall._ombros(pose)
        eixo = fall.eixo_cabeca(pose)
        razao = None
        if pose.caixa is not None:
            x0, y0, x1, y1 = pose.caixa
            razao = (x1 - x0) / max(y1 - y0, 1e-6)
        nariz = None
        if pose.keypoints is not None and pose.keypoints[p.NARIZ, 2] >= p.CONFIANCA_MINIMA:
            nariz = float(pose.keypoints[p.NARIZ, 1]) / altura_quadro
        linha = [
            f"{pose.t:.2f}",
            "" if eixo is None else f"{eixo:.0f}",
            "" if ombros is None else f"{ombros[0] / altura_quadro:.3f}",
            "" if ombros is None else f"{ombros[1] / altura_quadro:.3f}",
            "" if nariz is None else f"{nariz:.3f}",
            "" if razao is None else f"{razao:.2f}",
        ]
        novo = not destino.exists()
        with open(destino, "a", encoding="utf-8") as arquivo:
            if novo:
                arquivo.write("t,eixo,ombro_y,ombro_largura,nariz_y,razao_caixa\n")
            arquivo.write(",".join(linha) + "\n")

    def _procurar_queda(self, janela: deque, vistas: set[float]) -> None:
        sequencia = SequenciaPose(
            arquivo=str(self.fonte),
            fps=self.estado.fps or 15.0,
            largura=0,
            altura=0,
            total_frames=len(janela),
            passo=PASSO,
            recorte=None,
            segundos_processamento=0.0,
            poses=[pose for pose, _ in janela],
        )
        # Webcam próxima não enquadra o corpo inteiro: soma o critério do eixo da cabeça.
        detector = fall.detectar_todas if self.ao_vivo else fall.detectar
        for queda in detector(sequencia):
            # A mesma Queda reaparece enquanto está na janela: conta uma vez.
            if any(abs(queda.t - t) < JANELA_S for t in vistas):
                continue
            vistas.add(queda.t)
            self._emitir(queda, janela)

    def _emitir(self, queda: fall.Queda, janela: deque) -> None:
        pose, quadro = next((pq for pq in janela if pq[0].frame == queda.frame), janela[-1])
        no_quadro = objetos.detectar(quadro, self.modelo_objetos) if pose.caixa else []
        perto = objetos.proximos(no_quadro, pose.caixa) if pose.caixa else []
        anotado = p.anotar(objetos.anotar(quadro, no_quadro), pose)
        detectado = self.relogio
        destino = PASTA_EVIDENCIA / self.paciente_id / f"queda_{detectado:%Y%m%d_%H%M%S}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino), anotado)
        alerta = alerta_queda(
            queda,
            self.paciente_id,
            f"camera-{self.paciente_id}-{datetime.now():%H%M%S}",
            detectado - timedelta(seconds=queda.t),
            destino.relative_to(_RAIZ).as_posix(),
            perto,
        )
        conn = db.conectar(self.caminho_db)  # conexão própria: esta é outra thread
        try:
            # Uma Queda aberta por leito: enquanto não for reconhecida, não abre outra.
            abertas = db.listar(conn, paciente_id=self.paciente_id, status="novo", origem="video")
            if not any(a.tipo is Tipo.QUEDA for a in abertas):
                db.salvar(conn, alerta)
                self.estado.quedas.append(alerta.id)
        finally:
            conn.close()
