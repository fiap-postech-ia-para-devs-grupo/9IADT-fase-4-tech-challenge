"""Ângulos articulares a partir dos keypoints (ESTRATEGIA.md, seção 4).

Ângulos em graus. 180° é o segmento estendido (joelho reto); quanto menor, mais flexionado.
`tronco` é a inclinação do tronco (meio dos ombros → meio dos quadris) em relação à vertical:
0° é em pé, 90° é deitado. Keypoint com confiança abaixo de `pose.CONFIANCA_MINIMA` vira NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analyzers.video import pose as p
from src.analyzers.video.pose import SequenciaPose

ARTICULACOES: dict[str, tuple[int, int, int]] = {
    "joelho_esq": (p.QUADRIL_ESQ, p.JOELHO_ESQ, p.TORNOZELO_ESQ),
    "joelho_dir": (p.QUADRIL_DIR, p.JOELHO_DIR, p.TORNOZELO_DIR),
    "quadril_esq": (p.OMBRO_ESQ, p.QUADRIL_ESQ, p.JOELHO_ESQ),
    "quadril_dir": (p.OMBRO_DIR, p.QUADRIL_DIR, p.JOELHO_DIR),
    "ombro_esq": (p.QUADRIL_ESQ, p.OMBRO_ESQ, p.COTOVELO_ESQ),
    "ombro_dir": (p.QUADRIL_DIR, p.OMBRO_DIR, p.COTOVELO_DIR),
    "cotovelo_esq": (p.OMBRO_ESQ, p.COTOVELO_ESQ, p.PUNHO_ESQ),
    "cotovelo_dir": (p.OMBRO_DIR, p.COTOVELO_DIR, p.PUNHO_DIR),
}
COLUNAS = (*ARTICULACOES, "tronco")


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo ABC em graus, com vértice em B."""
    v1, v2 = a[:2] - b[:2], c[:2] - b[:2]
    normas = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if normas == 0:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / normas, -1.0, 1.0))))


def _visivel(kp: np.ndarray, *indices: int) -> bool:
    return all(kp[i, 2] >= p.CONFIANCA_MINIMA for i in indices)


def inclinacao_tronco(kp: np.ndarray) -> float:
    """Graus entre o tronco e a vertical da imagem; usa o lado visível se só um aparecer."""
    pares = [
        (o, q)
        for o, q in ((p.OMBRO_ESQ, p.QUADRIL_ESQ), (p.OMBRO_DIR, p.QUADRIL_DIR))
        if _visivel(kp, o, q)
    ]
    if not pares:
        return math.nan
    ombro = np.mean([kp[o, :2] for o, _ in pares], axis=0)
    quadril = np.mean([kp[q, :2] for _, q in pares], axis=0)
    dx, dy = ombro - quadril
    if dx == 0 and dy == 0:
        return math.nan
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def angulos(kp: np.ndarray | None) -> dict[str, float]:
    if kp is None:
        return dict.fromkeys(COLUNAS, math.nan)
    medidas = {
        nome: angulo(kp[a], kp[b], kp[c]) if _visivel(kp, a, b, c) else math.nan
        for nome, (a, b, c) in ARTICULACOES.items()
    }
    medidas["tronco"] = inclinacao_tronco(kp)
    return medidas


def serie(sequencia: SequenciaPose) -> pd.DataFrame:
    """Uma linha por frame processado, indexada pelo tempo (s); coluna `frame` para a Evidência."""
    linhas = [
        {"t": pose.t, "frame": pose.frame, **angulos(pose.keypoints)} for pose in sequencia.poses
    ]
    if not linhas:
        return pd.DataFrame(columns=["frame", *COLUNAS], index=pd.Index([], name="t"))
    return pd.DataFrame(linhas).set_index("t")
