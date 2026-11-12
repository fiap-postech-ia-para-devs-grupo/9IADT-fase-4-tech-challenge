import math

import numpy as np
import pytest

from src.analyzers.video import angles
from src.analyzers.video import pose as p
from tests import video_sintetico as vs


def test_angulo_reto_e_dobrado():
    a, b = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    assert angles.angulo(a, b, np.array([0.0, 2.0])) == pytest.approx(180)
    assert angles.angulo(a, b, np.array([1.0, 1.0])) == pytest.approx(90)


def test_angulo_com_pontos_coincidentes_e_nan():
    ponto = np.array([1.0, 1.0])
    assert math.isnan(angles.angulo(ponto, ponto, np.array([2.0, 2.0])))


@pytest.mark.parametrize("joelho", [170, 120, 90, 60])
def test_angulos_do_corpo_sintetico(joelho):
    medidas = angles.angulos(vs.corpo(joelho, tronco=30))

    assert medidas["joelho_esq"] == pytest.approx(joelho, abs=0.5)
    assert medidas["joelho_dir"] == pytest.approx(joelho, abs=0.5)
    assert medidas["tronco"] == pytest.approx(30, abs=0.5)


def test_keypoint_com_baixa_confianca_vira_nan():
    kp = vs.corpo(90)
    kp[p.TORNOZELO_ESQ, 2] = 0.1

    medidas = angles.angulos(kp)

    assert math.isnan(medidas["joelho_esq"])
    assert medidas["joelho_dir"] == pytest.approx(90, abs=0.5)


def test_sem_pessoa_todos_os_angulos_nan():
    assert all(math.isnan(v) for v in angles.angulos(None).values())


def test_serie_indexada_pelo_tempo_com_frame():
    seq = vs.de_keypoints([vs.corpo(170), vs.corpo(90)])

    serie = angles.serie(seq)

    assert list(serie.index) == [0.0, 0.1]
    assert list(serie["frame"]) == [0, 1]
    assert serie.loc[0.1, "joelho_esq"] == pytest.approx(90, abs=0.5)


def test_serie_vazia_tem_as_colunas():
    serie = angles.serie(vs.sequencia([]))

    assert serie.empty
    assert {"frame", "joelho_esq", "tronco"} <= set(serie.columns)
