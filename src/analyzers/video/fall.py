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
    score: ScoreQueda | None = None  # critério da câmera próxima (ao vivo)

    @property
    def duracao_transicao(self) -> float:
        return self.t - self.t_em_pe

    @property
    def descricao(self) -> str:
        if self.score is not None and self.score.saiu_por_baixo:
            return (
                "Queda pela câmera do leito: o paciente desceu e saiu da imagem por baixo, "
                f"sem voltar em {AUSENCIA_SAIDA:.0f} s (Score de Queda {self.score.score}/100)"
            )
        if self.score is not None:
            return (
                f"Queda pela câmera do leito: corpo inclinado a {self.score.angulo:.0f}° da "
                f"vertical (Score de Queda {self.score.score}/100) e o paciente permaneceu deitado"
            )
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


# --- Câmera próxima (webcam): sem o corpo inteiro no quadro -----------------------------------
#
# Com a câmera perto do paciente, a caixa nunca é "em pé" (só aparece o tronco). O sinal
# usado é o ângulo de inclinação, o eixo ombros → nariz: vertical com a pessoa sentada ou em
# pé, horizontal com ela deitada. Ele vira um Score de Queda de 0 a 100 (abaixo), mostrado ao
# vivo na tela, com os limiares explícitos.

EIXO_EM_PE = 30.0
EIXO_DEITADO = 60.0
# Acima disso o nariz está abaixo dos ombros: é a pessoa se curvando para a frente (pegar algo,
# sentar), não deitada de lado; não conta como queda.
EIXO_CURVADO = 140.0


def eixo_cabeca(pose: Pose) -> float | None:
    """Graus entre o vetor meio-dos-ombros → nariz e a vertical para cima (0 = em pé)."""
    import math

    from src.analyzers.video import pose as p

    kp = pose.keypoints
    if kp is None or min(kp[[p.NARIZ, p.OMBRO_ESQ, p.OMBRO_DIR], 2]) < p.CONFIANCA_MINIMA:
        return None
    ombros = (kp[p.OMBRO_ESQ, :2] + kp[p.OMBRO_DIR, :2]) / 2
    dx, dy = kp[p.NARIZ, :2] - ombros
    if dx == 0 and dy == 0:
        return None
    # dy < 0: nariz acima dos ombros (y da imagem cresce para baixo).
    return math.degrees(math.atan2(abs(dx), -dy))


def _ombros(pose: Pose) -> tuple[float, float] | None:
    """(y médio dos ombros, largura dos ombros) em pixels."""
    from src.analyzers.video import pose as p

    kp = pose.keypoints
    if kp is None or min(kp[[p.OMBRO_ESQ, p.OMBRO_DIR], 2]) < p.CONFIANCA_MINIMA:
        return None
    largura = float(abs(kp[p.OMBRO_ESQ, 0] - kp[p.OMBRO_DIR, 0]))
    return float((kp[p.OMBRO_ESQ, 1] + kp[p.OMBRO_DIR, 1]) / 2), max(largura, 1.0)


# Score de Queda (câmera próxima, ao vivo). Ângulo da inclinação = eixo ombros → nariz em
# relação à vertical: 0° ereto, 90° deitado de lado. Componentes, cada um de 0 a 1:
ANGULO_INCLINADO = EIXO_EM_PE  # abaixo disto, postura ereta (componente de inclinação = 0)
ANGULO_QUEDA = EIXO_DEITADO  # a partir disto, corpo deitado: o ângulo que caracteriza a Queda
VARIACAO_BRUSCA = 45.0  # graus ganhos em até JANELA_BRUSCA segundos = mudança brusca total
JANELA_BRUSCA = 3.0  # s
PESO_INCLINACAO, PESO_VARIACAO, PESO_DESCIDA = 0.4, 0.4, 0.2
SCORE_QUEDA = 60  # score que dispara o Alerta na hora, com o ângulo em ANGULO_QUEDA..EIXO_CURVADO


@dataclass(frozen=True)
class ScoreQueda:
    t: float
    angulo: float  # graus da vertical
    variacao: float  # graus ganhos nos últimos JANELA_BRUSCA s
    descida: float  # quanto os ombros desceram, em tamanhos de tronco, nos últimos JANELA_BRUSCA s
    score: int  # 0 a 100
    saiu_por_baixo: bool = False  # caiu para fora da imagem (câmera não vê o chão)

    @property
    def componentes(self) -> tuple[float, float, float]:
        curvado = self.angulo > EIXO_CURVADO  # nariz abaixo dos ombros: se curvou, não deitou
        return (
            0.0 if curvado else _faixa(self.angulo, ANGULO_INCLINADO, ANGULO_QUEDA),
            _faixa(self.variacao, 0.0, VARIACAO_BRUSCA),
            _faixa(self.descida, 0.0, 1.0),
        )

    @property
    def explicacao(self) -> str:
        if self.saiu_por_baixo:
            return (
                f"score {self.score}/100 · cabeça e ombros desceram e saíram da imagem por baixo "
                f"(ombros desceram {self.descida:.1f} largura(s)) e não voltaram em "
                f"{AUSENCIA_SAIDA:.0f} s"
            )
        return (
            f"score {self.score}/100 · ângulo {self.angulo:.0f}° (queda ≥ {ANGULO_QUEDA:.0f}°) · "
            f"variação {self.variacao:+.0f}° em {JANELA_BRUSCA:.0f} s · "
            f"ombros desceram {self.descida:.1f} largura(s)"
        )


def _faixa(valor: float, zero: float, um: float) -> float:
    return min(max((valor - zero) / (um - zero), 0.0), 1.0)


def _escala(pose: Pose) -> float:
    """Tamanho do tronco em pixels: a largura dos ombros, ou a distância ombros → nariz quando a
    pessoa está de lado (deitada de lado os ombros se sobrepõem e a largura vai a zero)."""
    from src.analyzers.video import pose as p

    ombros = _ombros(pose)
    assert ombros is not None and pose.keypoints is not None
    meio = (pose.keypoints[p.OMBRO_ESQ, :2] + pose.keypoints[p.OMBRO_DIR, :2]) / 2
    pescoco = float(((pose.keypoints[p.NARIZ, :2] - meio) ** 2).sum() ** 0.5)
    return max(ombros[1], pescoco, 1.0)


def scores(sequencia: SequenciaPose) -> list[ScoreQueda]:
    """Score de Queda de cada frame com ombros e nariz visíveis."""
    validas = [q for q in sequencia.poses if eixo_cabeca(q) is not None and _ombros(q)]
    saida: list[ScoreQueda] = []
    for i, pose in enumerate(validas):
        angulo = eixo_cabeca(pose) or 0.0
        anteriores = [q for q in validas[: i + 1] if pose.t - q.t <= JANELA_BRUSCA]
        variacao = angulo - min(eixo_cabeca(q) or 0.0 for q in anteriores)
        ombros = _ombros(pose)
        assert ombros is not None
        # Ombros, não o nariz: girar a cabeça também abaixa o nariz; cair abaixa os ombros.
        alturas = [o[0] for q in anteriores if (o := _ombros(q)) is not None]
        descida = (ombros[0] - min(alturas)) / _escala(pose)
        parcial = ScoreQueda(pose.t, angulo, variacao, max(descida, 0.0), 0)
        a, v, d = parcial.componentes
        nota = round(100 * (PESO_INCLINACAO * a + PESO_VARIACAO * v + PESO_DESCIDA * d))
        saida.append(ScoreQueda(pose.t, angulo, variacao, max(descida, 0.0), nota))
    return saida


def detectar_proxima(sequencia: SequenciaPose) -> list[Queda]:
    """Quedas pelo Score de Queda: vale no primeiro quadro com score >= SCORE_QUEDA e o corpo
    deitado (ângulo >= ANGULO_QUEDA), sem esperar confirmação: uma Queda detectada não é
    anulada depois. Uma por vez, até a pessoa voltar a ficar ereta."""
    pontos = scores(sequencia)
    por_t = {q.t: q for q in sequencia.poses}
    quedas: list[Queda] = []
    caido = False
    for i, ponto in enumerate(pontos):
        if ponto.angulo < ANGULO_INCLINADO:
            caido = False
        deitado = ANGULO_QUEDA <= ponto.angulo <= EIXO_CURVADO
        if caido or ponto.score < SCORE_QUEDA or not deitado:
            continue
        pose = por_t[ponto.t]
        inicio = min(
            (q for q in pontos[:i] if ponto.t - q.t <= JANELA_BRUSCA),
            key=lambda q: q.angulo,
            default=ponto,
        )
        quedas.append(
            Queda(
                t=ponto.t,
                frame=pose.frame,
                t_em_pe=inicio.t,
                razao_caixa=_razao(pose) or 0.0,
                altura_relativa=1.0,
                score=ponto,
            )
        )
        caido = True
    return quedas


# Webcam na altura da mesa não enxerga o chão: quem cai some pela parte de baixo da imagem.
SAIDA_FAIXA_INFERIOR = 0.7  # ombros na faixa de baixo da imagem (fração da altura)...
SAIDA_DESCIDA = 0.05  # ...depois de descer pelo menos isto da altura da imagem em 2 s...
AUSENCIA_SAIDA = 3.0  # ...e a cabeça não reaparece por este tempo (s)
SCORE_SAIDA = 80


def detectar_saida_por_baixo(sequencia: SequenciaPose) -> list[Queda]:
    """Quedas para fora da imagem: ombros descendo na faixa de baixo e, em seguida, nem cabeça
    nem ombros por AUSENCIA_SAIDA segundos. Precisa da altura do quadro na sequência."""
    if sequencia.altura <= 0:
        return []
    poses = sequencia.poses
    visivel = [eixo_cabeca(q) is not None and _ombros(q) is not None for q in poses]
    quedas: list[Queda] = []
    for i, pose in enumerate(poses):
        if not visivel[i] or (i + 1 < len(poses) and visivel[i + 1]):
            continue  # só o último quadro visível antes de sumir
        depois = poses[i + 1 :]
        if not depois or depois[-1].t - pose.t < AUSENCIA_SAIDA:
            continue
        ombros = _ombros(pose)
        assert ombros is not None
        antes = [o[0] for q in poses[: i + 1] if pose.t - q.t <= 2.0 and (o := _ombros(q))]
        descida = (ombros[0] - min(antes)) / sequencia.altura
        if ombros[0] < SAIDA_FAIXA_INFERIOR * sequencia.altura or descida < SAIDA_DESCIDA:
            continue
        ponto = ScoreQueda(
            pose.t,
            eixo_cabeca(pose) or 0.0,
            0.0,
            (ombros[0] - min(antes)) / _escala(pose),
            SCORE_SAIDA,
            saiu_por_baixo=True,
        )
        quedas.append(Queda(pose.t, pose.frame, pose.t, _razao(pose) or 0.0, 1.0, ponto))
    return quedas


def detectar_todas(sequencia: SequenciaPose) -> list[Queda]:
    """Critério do corpo inteiro e, sem ele, os da câmera próxima (sem contar a mesma 2 vezes)."""
    quedas = detectar(sequencia)
    for queda in detectar_proxima(sequencia) + detectar_saida_por_baixo(sequencia):
        if all(abs(queda.t - q.t) > TRANSICAO_MAXIMA + PERMANENCIA for q in quedas):
            quedas.append(queda)
    return sorted(quedas, key=lambda q: q.t)
