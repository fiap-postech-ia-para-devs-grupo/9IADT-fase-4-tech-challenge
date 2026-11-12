import numpy as np
import pytest

from src.analyzers.video import angles
from src.analyzers.video import exercise_rules as er
from src.analyzers.video.exercise_rules import EXERCICIOS, Regra, Vista
from tests import video_sintetico as vs

AGACHAMENTO = EXERCICIOS["agachamento"]
UNIPODAL = EXERCICIOS["agachamento_unipodal"]


def _agachamentos(fundos: list[float], tronco_no_fundo: float = 15.0) -> list[np.ndarray]:
    """Em pé (170°) → desce até cada fundo → sobe; 2 s por repetição a 10 fps."""
    frames = [vs.corpo(170)] * 5
    for fundo in fundos:
        descida = np.linspace(170, fundo, 10)
        for joelho in [*descida, *descida[::-1]]:
            tronco = 5 + (tronco_no_fundo - 5) * (170 - joelho) / (170 - fundo)
            frames.append(vs.corpo(joelho, tronco))
        frames += [vs.corpo(170)] * 5
    return frames


def _avaliar(frames, exercicio=AGACHAMENTO, vista=Vista.LATERAL):
    seq = vs.de_keypoints(frames)
    return er.avaliar(seq, angles.serie(seq), exercicio, vista)


def test_agachamentos_profundos_sem_desvio():
    reps, desvios = _avaliar(_agachamentos([60, 70, 80]))

    assert [r.numero for r in reps] == [1, 2, 3]
    assert [round(r.joelho_fundo) for r in reps] == [60, 70, 80]
    assert desvios == []


def test_repeticao_rasa_e_amplitude_insuficiente():
    _, desvios = _avaliar(_agachamentos([60, 125, 70]))

    assert [(d.regra, d.repeticao) for d in desvios] == [(Regra.AMPLITUDE_INSUFICIENTE, 2)]
    assert desvios[0].valor == pytest.approx(125, abs=0.5)
    assert desvios[0].limite == AGACHAMENTO.joelho_alvo


def test_tronco_inclinado_e_compensacao():
    _, desvios = _avaliar(_agachamentos([70], tronco_no_fundo=65))

    assert [d.regra for d in desvios] == [Regra.COMPENSACAO_TRONCO]
    assert desvios[0].valor == pytest.approx(65, abs=1)


def test_flexao_curta_demais_nao_e_repeticao():
    frames = [vs.corpo(170)] * 5 + [vs.corpo(90)] * 2 + [vs.corpo(170)] * 5

    reps, _ = _avaliar(frames)

    assert reps == []


def test_oclusao_momentanea_nao_quebra_a_repeticao():
    frames = _agachamentos([70])
    frames[9] = vs.corpo(70) * 0  # pessoa não detectada por um frame, no fundo

    reps, _ = _avaliar(frames)

    assert len(reps) == 1


def test_fppa_medial_positivo_para_dentro():
    assert er.fppa_medial(vs.frontal_unipodal(20), "dir") == pytest.approx(20, abs=0.5)
    assert er.fppa_medial(vs.frontal_unipodal(-15), "dir") == pytest.approx(-15, abs=0.5)
    assert er.fppa_medial(vs.frontal_unipodal(0), "dir") == pytest.approx(0, abs=0.5)


def test_perna_de_apoio_e_a_do_tornozelo_mais_baixo():
    assert er.perna_de_apoio(vs.de_keypoints([vs.frontal_unipodal(0)] * 3)) == "dir"


def _unipodal(picos: list[float]) -> list[np.ndarray]:
    frames = [vs.frontal_unipodal(3)] * 5
    for pico in picos:
        frames += [vs.frontal_unipodal(v) for v in (6, pico, pico, pico, pico, pico, 6)]
        frames += [vs.frontal_unipodal(3)] * 5
    return frames


def test_valgo_por_episodio_na_vista_frontal():
    reps, desvios = _avaliar(_unipodal([18, 5, 15]), UNIPODAL, Vista.FRONTAL)

    assert reps == []
    assert [d.regra for d in desvios] == [Regra.VALGO_DINAMICO] * 2
    assert [round(d.valor) for d in desvios] == [18, 15]
    assert {d.lado for d in desvios} == {"dir"}


def test_vista_lateral_nao_avalia_valgo_nem_assimetria():
    _, desvios = _avaliar(_unipodal([18]), UNIPODAL, Vista.LATERAL)

    assert desvios == []


def test_assimetria_so_na_vista_frontal():
    frames = _agachamentos([70])
    for i, kp in enumerate(frames):
        torto = kp.copy()
        torto[13:16:2] = vs.corpo(170)[13:16:2]  # joelho e tornozelo esquerdos sempre retos
        frames[i] = torto

    _, lateral = _avaliar(frames)
    _, frontal = _avaliar(frames, vista=Vista.FRONTAL)

    assert Regra.ASSIMETRIA not in {d.regra for d in lateral}
    assert Regra.ASSIMETRIA in {d.regra for d in frontal}


def test_descricao_do_desvio():
    desvio = er.Desvio(Regra.AMPLITUDE_INSUFICIENTE, 2.0, 20, 125, 100, 2)

    assert desvio.descricao == "amplitude insuficiente (repetição 2): 125° (limite 100°)"
