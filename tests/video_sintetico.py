"""Poses sintéticas para os testes do Analisador de Vídeo: sem YOLO, sem clipe."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.analyzers.video import pose as p
from src.analyzers.video.pose import Pose, SequenciaPose

FPS = 10.0


def keypoints(pontos: dict[int, tuple[float, float]]) -> np.ndarray:
    kp = np.zeros((17, 3))
    for indice, (x, y) in pontos.items():
        kp[indice] = (x, y, 0.9)
    return kp


def corpo(joelho: float, tronco: float = 10.0) -> np.ndarray:
    """De perfil: as duas pernas com o joelho em `joelho` graus e o tronco inclinado `tronco`."""
    quadril = np.array([100.0, 100.0])
    joelho_xy = quadril + (0, 50)
    desvio = math.radians(180 - joelho)
    tornozelo = joelho_xy + 50 * np.array([math.sin(desvio), math.cos(desvio)])
    inclinacao = math.radians(tronco)
    ombro = quadril + 60 * np.array([math.sin(inclinacao), -math.cos(inclinacao)])
    pontos: dict[int, tuple[float, float]] = {}
    for q, j, t, o in (
        (p.QUADRIL_ESQ, p.JOELHO_ESQ, p.TORNOZELO_ESQ, p.OMBRO_ESQ),
        (p.QUADRIL_DIR, p.JOELHO_DIR, p.TORNOZELO_DIR, p.OMBRO_DIR),
    ):
        pontos |= {q: tuple(quadril), j: tuple(joelho_xy), t: tuple(tornozelo), o: tuple(ombro)}
    return keypoints(pontos)


def frontal_unipodal(valgo: float) -> np.ndarray:
    """De frente, apoio na perna direita (tornozelo mais baixo), joelho `valgo` graus para dentro.

    Na imagem, o quadril direito do paciente fica à esquerda (x menor): o lado medial da perna
    direita é +x, onde está o quadril esquerdo.
    """
    quadril_dir, quadril_esq = np.array([80.0, 100.0]), np.array([120.0, 100.0])
    angulo = math.radians(valgo / 2)
    joelho_dir = quadril_dir + 50 * np.array([math.sin(angulo), math.cos(angulo)])
    tornozelo_dir = joelho_dir + 50 * np.array([-math.sin(angulo), math.cos(angulo)])
    return keypoints(
        {
            p.QUADRIL_DIR: tuple(quadril_dir),
            p.JOELHO_DIR: tuple(joelho_dir),
            p.TORNOZELO_DIR: tuple(tornozelo_dir),
            p.QUADRIL_ESQ: tuple(quadril_esq),
            p.JOELHO_ESQ: (120.0, 130.0),
            p.TORNOZELO_ESQ: (125.0, 150.0),  # perna esquerda erguida
        }
    )


def sequencia(poses: list[Pose], largura: int = 640, altura: int = 480) -> SequenciaPose:
    total = poses[-1].frame + 1 if poses else 0
    return SequenciaPose(
        arquivo="sintetico.mp4",
        fps=FPS,
        largura=largura,
        altura=altura,
        total_frames=total,
        passo=1,
        recorte=None,
        segundos_processamento=0.5,
        poses=poses,
    )


def de_keypoints(lista: list[np.ndarray]) -> SequenciaPose:
    return sequencia(
        [Pose(i, i / FPS, kp, (50.0, 30.0, 150.0, 210.0)) for i, kp in enumerate(lista)]
    )


def de_caixas(caixas: Sequence[tuple[float, float] | None]) -> SequenciaPose:
    """Uma pose por frame a partir de (largura, altura) da caixa; None é frame sem pessoa."""
    poses = []
    for i, caixa in enumerate(caixas):
        if caixa is None:
            poses.append(Pose(i, i / FPS, None, None))
            continue
        largura, altura = caixa
        poses.append(Pose(i, i / FPS, None, (100.0, 300.0 - altura, 100.0 + largura, 300.0)))
    return sequencia(poses)
