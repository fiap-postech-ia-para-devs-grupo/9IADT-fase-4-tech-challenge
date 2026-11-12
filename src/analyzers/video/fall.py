"""Queda: o paciente passa de em pé para o chão (ESTRATEGIA.md, seção 4).

Critério, sobre a caixa da pessoa detectada pelo YOLOv8-pose:

1. **em pé**: caixa vertical, largura/altura < `RAZAO_EM_PE` (0,6);
2. **no chão**: caixa deitada, largura/altura >= `RAZAO_NO_CHAO` (1,0), e altura da caixa
   <= `ALTURA_NO_CHAO` (50%) da altura em pé;
3. **queda**: o estado "no chão" começa até `TRANSICAO_MAXIMA` (2 s) depois do último frame em
   pé e se mantém (mediana da janela) por `PERMANENCIA` (1 s).

A altura relativa separa a queda de um exercício com a perna estendida (caixa larga, mas a
pessoa continua alta). A transição curta separa a queda de deitar-se devagar na cama. A
permanência descarta um frame isolado com a caixa errada. O ponto fraco é a câmera de frente
para a pessoa que cai na direção dela: a caixa encurta menos (ver relatório, seção 3.1).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.analyzers.video.pose import Pose, SequenciaPose

RAZAO_EM_PE = 0.6
RAZAO_NO_CHAO = 1.0
ALTURA_NO_CHAO = 0.5
TRANSICAO_MAXIMA = 2.0  # s
PERMANENCIA = 1.0  # s
MIN_FRAMES_PERMANENCIA = 3


@dataclass(frozen=True)
class Queda:
    t: float  # primeiro frame no chão
    frame: int
    t_em_pe: float  # último frame em pé antes da queda
    razao_caixa: float  # largura/altura no primeiro frame no chão
    altura_relativa: float  # altura da caixa no chão / altura em pé

    @property
    def duracao_transicao(self) -> float:
        return self.t - self.t_em_pe

    @property
    def descricao(self) -> str:
        return (
            f"Queda: de em pé para o chão em {self.duracao_transicao:.1f} s "
            f"(t = {self.t:.1f} s), e o paciente permaneceu no chão"
        )


def _razao(pose: Pose) -> float | None:
    if pose.caixa is None:
        return None
    x0, y0, x1, y1 = pose.caixa
    return (x1 - x0) / (y1 - y0) if y1 > y0 else None


def _altura(pose: Pose) -> float:
    assert pose.caixa is not None
    return pose.caixa[3] - pose.caixa[1]


def _no_chao(pose: Pose, altura_em_pe: float) -> bool:
    razao = _razao(pose)
    return (
        razao is not None
        and razao >= RAZAO_NO_CHAO
        and _altura(pose) <= ALTURA_NO_CHAO * altura_em_pe
    )


def _permanece(poses: list[Pose], inicio: int, altura_em_pe: float) -> bool:
    t0 = poses[inicio].t
    janela = [q for q in poses[inicio:] if q.t - t0 <= PERMANENCIA and q.caixa is not None]
    if len(janela) < MIN_FRAMES_PERMANENCIA:
        return False
    razoes = [_razao(q) or 0.0 for q in janela]
    alturas = [_altura(q) for q in janela]
    return (
        statistics.median(razoes) >= RAZAO_NO_CHAO
        and statistics.median(alturas) <= ALTURA_NO_CHAO * altura_em_pe
    )


def detectar(sequencia: SequenciaPose) -> list[Queda]:
    """Quedas do clipe; depois de uma queda, só conta outra se o paciente ficar em pé de novo."""
    poses = sequencia.poses
    quedas: list[Queda] = []
    em_pe: list[Pose] = []
    altura_em_pe = 0.0
    caido = False

    for i, pose in enumerate(poses):
        razao = _razao(pose)
        if razao is None:
            continue
        if razao < RAZAO_EM_PE:
            # Referência: a maior altura em pé nos últimos segundos. A pessoa pode estar andando
            # em direção à câmera, e o último frame em pé pode já ser o início da queda.
            em_pe = [q for q in em_pe if pose.t - q.t <= TRANSICAO_MAXIMA] + [pose]
            altura_em_pe = max(_altura(q) for q in em_pe)
            caido = False
            continue
        ultimo_em_pe = em_pe[-1] if em_pe else None
        if caido or ultimo_em_pe is None or pose.t - ultimo_em_pe.t > TRANSICAO_MAXIMA:
            continue
        if _no_chao(pose, altura_em_pe) and _permanece(poses, i, altura_em_pe):
            quedas.append(
                Queda(
                    t=pose.t,
                    frame=pose.frame,
                    t_em_pe=ultimo_em_pe.t,
                    razao_caixa=razao,
                    altura_relativa=_altura(pose) / altura_em_pe,
                )
            )
            caido = True
    return quedas
