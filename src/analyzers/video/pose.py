"""Estimativa de pose com YOLOv8n-pose em CPU (ESTRATEGIA.md, seção 4).

`estimar` percorre o clipe e devolve, para cada frame processado, os 17 keypoints COCO da pessoa
mais confiável. Em CPU, processar 1 a cada `passo` frames (padrão 3, ~10 por segundo num clipe de
30 fps) basta para ângulos e quedas e mantém o tempo de processamento perto do tempo do clipe.

    sequencia = pose.estimar(Path("agachamento.webm"))
    sequencia.poses[0].keypoints  # (17, 3): x, y, confiança
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_RAIZ = Path(__file__).resolve().parents[3]
# Os pesos (~6 MB) são baixados pelo Ultralytics na primeira execução; `*.pt` está no .gitignore.
MODELO_PADRAO = _RAIZ / "data" / "video" / "yolov8n-pose.pt"
PASSO_PADRAO = 3

# Índices COCO-17.
NARIZ = 0
OMBRO_ESQ, OMBRO_DIR = 5, 6
COTOVELO_ESQ, COTOVELO_DIR = 7, 8
PUNHO_ESQ, PUNHO_DIR = 9, 10
QUADRIL_ESQ, QUADRIL_DIR = 11, 12
JOELHO_ESQ, JOELHO_DIR = 13, 14
TORNOZELO_ESQ, TORNOZELO_DIR = 15, 16

ESQUELETO = (
    (OMBRO_ESQ, OMBRO_DIR),
    (OMBRO_ESQ, COTOVELO_ESQ),
    (COTOVELO_ESQ, PUNHO_ESQ),
    (OMBRO_DIR, COTOVELO_DIR),
    (COTOVELO_DIR, PUNHO_DIR),
    (OMBRO_ESQ, QUADRIL_ESQ),
    (OMBRO_DIR, QUADRIL_DIR),
    (QUADRIL_ESQ, QUADRIL_DIR),
    (QUADRIL_ESQ, JOELHO_ESQ),
    (JOELHO_ESQ, TORNOZELO_ESQ),
    (QUADRIL_DIR, JOELHO_DIR),
    (JOELHO_DIR, TORNOZELO_DIR),
)

# Abaixo disso o keypoint é tratado como ausente (oculto ou fora do quadro).
CONFIANCA_MINIMA = 0.5

Recorte = tuple[int, int, int, int]  # x0, y0, x1, y1 em pixels do frame original


@dataclass(frozen=True)
class Pose:
    frame: int
    t: float  # segundos desde o início do clipe
    keypoints: np.ndarray | None  # (17, 3): x, y, confiança; None se ninguém foi detectado
    caixa: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1


@dataclass(frozen=True)
class SequenciaPose:
    arquivo: str
    fps: float
    largura: int
    altura: int
    total_frames: int
    passo: int
    recorte: Recorte | None
    segundos_processamento: float
    poses: list[Pose] = field(default_factory=list)

    @property
    def duracao(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def carregar_modelo(caminho: Path = MODELO_PADRAO) -> Any:
    from ultralytics import YOLO  # import tardio: o Ultralytics demora a carregar

    caminho.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(caminho))


def _recortar(frame: np.ndarray, recorte: Recorte | None) -> np.ndarray:
    if recorte is None:
        return frame
    x0, y0, x1, y1 = recorte
    return frame[y0:y1, x0:x1]


def frames(caminho: Path, recorte: Recorte | None = None) -> Iterator[tuple[int, np.ndarray]]:
    captura = cv2.VideoCapture(str(caminho))
    if not captura.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo {caminho}")
    try:
        indice = 0
        while True:
            ok, frame = captura.read()
            if not ok:
                return
            yield indice, _recortar(frame, recorte)
            indice += 1
    finally:
        captura.release()


def ler_frame(caminho: Path, indice: int, recorte: Recorte | None = None) -> np.ndarray:
    """Um frame específico, para anotar a Evidência depois da análise."""
    captura = cv2.VideoCapture(str(caminho))
    try:
        captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = captura.read()
    finally:
        captura.release()
    if not ok:
        raise ValueError(f"frame {indice} não existe em {caminho}")
    return _recortar(frame, recorte)


def _pessoa_principal(resultado: Any) -> tuple[np.ndarray | None, tuple | None]:
    """A pessoa de maior confiança; num quarto ou sessão há um paciente por clipe."""
    if resultado.keypoints is None or len(resultado.boxes) == 0:
        return None, None
    melhor = int(np.argmax(resultado.boxes.conf.cpu().numpy()))
    keypoints = resultado.keypoints.data[melhor].cpu().numpy().astype(float)
    caixa = tuple(float(v) for v in resultado.boxes.xyxy[melhor].cpu().numpy())
    return keypoints, caixa


def estimar(
    caminho: Path,
    *,
    passo: int = PASSO_PADRAO,
    recorte: Recorte | None = None,
    modelo: Any = None,
) -> SequenciaPose:
    """Pose frame a frame (1 a cada `passo`), com o tempo de processamento medido em CPU."""
    captura = cv2.VideoCapture(str(caminho))
    if not captura.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo {caminho}")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    captura.release()
    if recorte is not None:
        largura, altura = recorte[2] - recorte[0], recorte[3] - recorte[1]

    modelo = modelo or carregar_modelo()
    inicio = time.perf_counter()
    poses: list[Pose] = []
    lidos = 0
    for indice, frame in frames(caminho, recorte):
        lidos = indice + 1
        if indice % passo:
            continue
        resultado = modelo(frame, verbose=False, device="cpu")[0]
        keypoints, caixa = _pessoa_principal(resultado)
        poses.append(Pose(indice, indice / fps, keypoints, caixa))

    return SequenciaPose(
        arquivo=caminho.name,
        fps=fps,
        largura=largura,
        altura=altura,
        # O CAP_PROP_FRAME_COUNT de WebM costuma vir errado; vale o que foi lido.
        total_frames=lidos or total,
        passo=passo,
        recorte=recorte,
        segundos_processamento=time.perf_counter() - inicio,
        poses=poses,
    )


def anotar(frame: np.ndarray, pose: Pose, *, destaque: tuple[int, ...] = ()) -> np.ndarray:
    """Desenha o esqueleto (e a caixa) sobre uma cópia do frame; `destaque` pinta articulações."""
    saida = frame.copy()
    if pose.keypoints is None:
        return saida
    espessura = max(2, round(max(saida.shape[:2]) / 300))
    kp = pose.keypoints
    visivel = kp[:, 2] >= CONFIANCA_MINIMA
    for a, b in ESQUELETO:
        if visivel[a] and visivel[b]:
            pa = (round(kp[a, 0]), round(kp[a, 1]))
            pb = (round(kp[b, 0]), round(kp[b, 1]))
            cv2.line(saida, pa, pb, (255, 200, 0), espessura, cv2.LINE_AA)
    for i, (x, y, conf) in enumerate(kp):
        if conf >= CONFIANCA_MINIMA:
            cor = (0, 0, 255) if i in destaque else (0, 220, 0)
            raio = espessura * (3 if i in destaque else 2)
            cv2.circle(saida, (round(x), round(y)), raio, cor, -1, cv2.LINE_AA)
    if pose.caixa is not None:
        x0, y0, x1, y1 = (round(v) for v in pose.caixa)
        cv2.rectangle(saida, (x0, y0), (x1, y1), (200, 120, 0), max(1, espessura // 2))
    return saida
