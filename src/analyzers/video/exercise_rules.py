"""Desvio de Execução por exercício (ESTRATEGIA.md, seção 4).

Cada `Exercicio` declara a faixa esperada. Os limiares abaixo são os documentados no relatório
(seção 3.1). São referências de literatura de reabilitação, não calibradas por paciente:

- **amplitude insuficiente**: no fundo da repetição, o joelho mais flexionado não chega a
  `joelho_alvo` graus. O agachamento terapêutico pede coxa perto da horizontal, ~90-100°.
- **compensação de tronco**: durante a repetição, o tronco inclina mais que `tronco_max` graus
  em relação à vertical. Inclinação excessiva desloca a carga do joelho para a lombar.
- **valgo dinâmico**: na perna de apoio, o joelho desvia para dentro (medial) da linha
  quadril-tornozelo. O ângulo de projeção no plano frontal (FPPA) acima de ~10° é o corte usual
  para valgo dinâmico no agachamento unipodal. Só faz sentido com câmera frontal.
- **assimetria**: no fundo da repetição, os dois joelhos diferem mais que `assimetria_max`
  graus. Também exige câmera frontal ou de costas: de lado, uma perna esconde a outra.

Repetição é um trecho em que o joelho mais flexionado fica abaixo de `limiar_repeticao`, por
pelo menos `DURACAO_MINIMA_REPETICAO` segundos. Valgo é contado por episódio (trecho contínuo
acima do limiar), porque no agachamento unipodal visto de frente a flexão do joelho quase não
aparece na imagem e não serve para separar repetições.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.analyzers.video import pose as p
from src.analyzers.video.pose import SequenciaPose

DURACAO_MINIMA_REPETICAO = 0.4  # s
DURACAO_MINIMA_EPISODIO = 0.3  # s
# Frames sem medida dentro de um trecho (oclusão momentânea) não quebram a repetição.
LACUNA_TOLERADA = 0.35  # s


class Vista(StrEnum):
    LATERAL = "lateral"
    FRONTAL = "frontal"  # de frente ou de costas


class Regra(StrEnum):
    AMPLITUDE_INSUFICIENTE = "amplitude_insuficiente"
    COMPENSACAO_TRONCO = "compensacao_tronco"
    VALGO_DINAMICO = "valgo_dinamico"
    ASSIMETRIA = "assimetria"


DESCRICAO_REGRA = {
    Regra.AMPLITUDE_INSUFICIENTE: "amplitude insuficiente",
    Regra.COMPENSACAO_TRONCO: "compensação de tronco",
    Regra.VALGO_DINAMICO: "valgo dinâmico do joelho",
    Regra.ASSIMETRIA: "assimetria entre os joelhos",
}


@dataclass(frozen=True)
class Exercicio:
    nome: str
    rotulo: str
    limiar_repeticao: float | None  # joelho abaixo disso = em repetição; None: sem repetições
    joelho_alvo: float | None = None
    tronco_max: float | None = None
    valgo_max: float | None = None
    assimetria_max: float | None = None


EXERCICIOS: dict[str, Exercicio] = {
    "agachamento": Exercicio(
        "agachamento",
        "Agachamento",
        limiar_repeticao=140,
        joelho_alvo=100,
        tronco_max=50,
        assimetria_max=20,
    ),
    "agachamento_unipodal": Exercicio(
        "agachamento_unipodal",
        "Agachamento unipodal (mini agachamento em uma perna)",
        limiar_repeticao=None,
        valgo_max=10,
    ),
}


@dataclass(frozen=True)
class Desvio:
    regra: Regra
    t: float  # instante mais representativo (fundo da repetição ou pico do episódio)
    frame: int
    valor: float  # graus medidos
    limite: float  # graus do limiar violado
    repeticao: int | None  # 1, 2, ...; None quando a regra é por episódio
    lado: str | None = None  # "esq"/"dir" para valgo

    @property
    def descricao(self) -> str:
        onde = f"repetição {self.repeticao}" if self.repeticao else f"t = {self.t:.1f} s"
        return (
            f"{DESCRICAO_REGRA[self.regra]} ({onde}): {self.valor:.0f}° (limite {self.limite:.0f}°)"
        )


@dataclass(frozen=True)
class Repeticao:
    numero: int
    inicio: float
    fim: float
    t_fundo: float
    frame_fundo: int
    joelho_fundo: float


def _trechos(mascara: pd.Series, duracao_minima: float) -> list[tuple[float, float]]:
    """Trechos [t0, t1] em que `mascara` é verdadeira, juntando lacunas curtas de NaN/False."""
    trechos: list[list[float]] = []
    for t, ativo in zip(mascara.index.astype(float), mascara, strict=True):
        if not ativo:
            continue
        if trechos and t - trechos[-1][1] <= LACUNA_TOLERADA:
            trechos[-1][1] = t
        else:
            trechos.append([t, t])
    return [(a, b) for a, b in trechos if b - a >= duracao_minima]


def joelho_mais_flexionado(serie: pd.DataFrame) -> pd.Series:
    return serie[["joelho_esq", "joelho_dir"]].min(axis=1, skipna=True)


def repeticoes(serie: pd.DataFrame, exercicio: Exercicio) -> list[Repeticao]:
    if exercicio.limiar_repeticao is None or serie.empty:
        return []
    joelho = joelho_mais_flexionado(serie)
    resultado = []
    for numero, (t0, t1) in enumerate(
        _trechos(joelho < exercicio.limiar_repeticao, DURACAO_MINIMA_REPETICAO), start=1
    ):
        trecho = joelho.loc[t0:t1]
        t_fundo = float(trecho.idxmin())
        resultado.append(
            Repeticao(
                numero,
                t0,
                t1,
                t_fundo,
                int(serie.loc[t_fundo, "frame"]),
                float(trecho.min()),
            )
        )
    return resultado


def _desvios_por_repeticao(
    serie: pd.DataFrame, exercicio: Exercicio, reps: list[Repeticao]
) -> list[Desvio]:
    desvios: list[Desvio] = []
    for rep in reps:
        if exercicio.joelho_alvo is not None and rep.joelho_fundo > exercicio.joelho_alvo:
            desvios.append(
                Desvio(
                    Regra.AMPLITUDE_INSUFICIENTE,
                    rep.t_fundo,
                    rep.frame_fundo,
                    rep.joelho_fundo,
                    exercicio.joelho_alvo,
                    rep.numero,
                )
            )
        trecho = serie.loc[rep.inicio : rep.fim]
        tronco = trecho["tronco"].dropna()
        if exercicio.tronco_max is not None and not tronco.empty:
            if tronco.max() > exercicio.tronco_max:
                t = float(tronco.idxmax())
                desvios.append(
                    Desvio(
                        Regra.COMPENSACAO_TRONCO,
                        t,
                        int(serie.loc[t, "frame"]),
                        float(tronco.max()),
                        exercicio.tronco_max,
                        rep.numero,
                    )
                )
        if exercicio.assimetria_max is not None:
            esq = serie.loc[rep.t_fundo, "joelho_esq"]
            dir_ = serie.loc[rep.t_fundo, "joelho_dir"]
            diferenca = abs(float(esq) - float(dir_))
            if not math.isnan(diferenca) and diferenca > exercicio.assimetria_max:
                desvios.append(
                    Desvio(
                        Regra.ASSIMETRIA,
                        rep.t_fundo,
                        rep.frame_fundo,
                        diferenca,
                        exercicio.assimetria_max,
                        rep.numero,
                    )
                )
    return desvios


def fppa_medial(kp: np.ndarray, lado: str) -> float:
    """Ângulo de projeção no plano frontal do joelho, positivo quando o joelho vai para dentro.

    0° é quadril, joelho e tornozelo alinhados. O lado medial é o do outro quadril.
    """
    quadril, joelho, tornozelo, outro_quadril = (
        (p.QUADRIL_ESQ, p.JOELHO_ESQ, p.TORNOZELO_ESQ, p.QUADRIL_DIR)
        if lado == "esq"
        else (p.QUADRIL_DIR, p.JOELHO_DIR, p.TORNOZELO_DIR, p.QUADRIL_ESQ)
    )
    if min(kp[[quadril, joelho, tornozelo, outro_quadril], 2]) < p.CONFIANCA_MINIMA:
        return math.nan
    h, k, a = kp[quadril, :2], kp[joelho, :2], kp[tornozelo, :2]
    coxa, perna = k - h, a - k
    normas = float(np.linalg.norm(coxa) * np.linalg.norm(perna))
    if normas == 0:
        return math.nan
    desvio = math.degrees(math.acos(float(np.clip(np.dot(coxa, perna) / normas, -1.0, 1.0))))
    # De que lado da reta quadril→tornozelo o joelho está, comparado ao lado do outro quadril.
    eixo = a - h
    lado_joelho = np.sign(eixo[0] * (k - h)[1] - eixo[1] * (k - h)[0])
    lado_medial = np.sign(
        eixo[0] * (kp[outro_quadril, 1] - h[1]) - eixo[1] * (kp[outro_quadril, 0] - h[0])
    )
    return desvio if lado_joelho == lado_medial else -desvio


def perna_de_apoio(sequencia: SequenciaPose) -> str:
    """A perna cujo tornozelo fica mais baixo na imagem (maior y) na maior parte do clipe."""
    votos = [
        pose.keypoints[p.TORNOZELO_DIR, 1] > pose.keypoints[p.TORNOZELO_ESQ, 1]
        for pose in sequencia.poses
        if pose.keypoints is not None
    ]
    return "dir" if votos and sum(votos) > len(votos) / 2 else "esq"


def serie_valgo(sequencia: SequenciaPose, lado: str | None = None) -> pd.Series:
    """FPPA medial da perna de apoio por frame processado, indexado pelo tempo (s)."""
    lado = lado or perna_de_apoio(sequencia)
    return pd.Series(
        {
            pose.t: fppa_medial(pose.keypoints, lado) if pose.keypoints is not None else math.nan
            for pose in sequencia.poses
        },
        dtype=float,
        name="valgo",
    )


def _desvios_de_valgo(sequencia: SequenciaPose, exercicio: Exercicio) -> list[Desvio]:
    if exercicio.valgo_max is None:
        return []
    lado = perna_de_apoio(sequencia)
    fppa = serie_valgo(sequencia, lado)
    frames = {pose.t: pose.frame for pose in sequencia.poses}
    desvios = []
    for numero, (t0, t1) in enumerate(
        _trechos(fppa > exercicio.valgo_max, DURACAO_MINIMA_EPISODIO), start=1
    ):
        trecho = fppa.loc[t0:t1]
        t_pico = float(trecho.idxmax())
        desvios.append(
            Desvio(
                Regra.VALGO_DINAMICO,
                t_pico,
                frames[t_pico],
                float(trecho.max()),
                exercicio.valgo_max,
                numero,
                lado,
            )
        )
    return desvios


def avaliar(
    sequencia: SequenciaPose,
    serie: pd.DataFrame,
    exercicio: Exercicio,
    vista: Vista = Vista.LATERAL,
) -> tuple[list[Repeticao], list[Desvio]]:
    """Repetições e Desvios de Execução do clipe, em ordem de tempo.

    Valgo e assimetria só são avaliados com `vista` frontal: de lado, o plano frontal não aparece.
    """
    if vista is Vista.LATERAL:
        exercicio = Exercicio(
            exercicio.nome,
            exercicio.rotulo,
            exercicio.limiar_repeticao,
            joelho_alvo=exercicio.joelho_alvo,
            tronco_max=exercicio.tronco_max,
        )
    reps = repeticoes(serie, exercicio)
    desvios = [
        *_desvios_por_repeticao(serie, exercicio, reps),
        *_desvios_de_valgo(sequencia, exercicio),
    ]
    return reps, sorted(desvios, key=lambda d: d.t)
