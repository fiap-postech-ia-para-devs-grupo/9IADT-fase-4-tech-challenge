import pytest

from src.analyzers.video import fall
from tests import video_sintetico as vs

EM_PE = (60.0, 170.0)  # razão 0,35
NO_CHAO = (160.0, 70.0)  # razão 2,3, 41% da altura em pé
CAINDO = (100.0, 120.0)


def test_queda_rapida_que_permanece_no_chao():
    seq = vs.de_caixas([EM_PE] * 20 + [CAINDO] * 3 + [NO_CHAO] * 15)

    quedas = fall.detectar(seq)

    assert len(quedas) == 1
    queda = quedas[0]
    assert queda.t == pytest.approx(2.3)
    assert queda.duracao_transicao == pytest.approx(0.4)
    assert queda.altura_relativa == pytest.approx(70 / 170)


def test_deitar_devagar_nao_e_queda():
    transicao = [CAINDO] * 25  # 2,5 s entre em pé e deitado
    seq = vs.de_caixas([EM_PE] * 10 + transicao + [NO_CHAO] * 15)

    assert fall.detectar(seq) == []


def test_caixa_larga_com_a_pessoa_alta_nao_e_queda():
    perna_estendida = (150.0, 110.0)  # razão 1,36, mas 65% da altura em pé
    seq = vs.de_caixas([EM_PE] * 10 + [perna_estendida] * 15)

    assert fall.detectar(seq) == []


def test_frame_isolado_no_chao_nao_e_queda():
    seq = vs.de_caixas([EM_PE] * 10 + [NO_CHAO] + [EM_PE] * 10)

    assert fall.detectar(seq) == []


def test_frames_sem_pessoa_no_chao_sao_tolerados():
    seq = vs.de_caixas([EM_PE] * 10 + [NO_CHAO, None, NO_CHAO, None, NO_CHAO, NO_CHAO])

    assert len(fall.detectar(seq)) == 1


def test_levanta_e_cai_de_novo_conta_duas_quedas():
    uma_queda = [EM_PE] * 10 + [NO_CHAO] * 12
    seq = vs.de_caixas(uma_queda + uma_queda)

    assert len(fall.detectar(seq)) == 2


def test_sem_pessoa_nenhuma_queda():
    assert fall.detectar(vs.de_caixas([None] * 10)) == []


def _cabeca(angulo: float, i: int, descida: float = 0.0):
    """Pose com o nariz a `angulo` graus da vertical em relação ao meio dos ombros."""
    import math

    import numpy as np

    from src.analyzers.video import pose as p
    from src.analyzers.video.pose import Pose

    kp = np.zeros((17, 3))
    y = 200 + descida
    kp[p.OMBRO_ESQ] = (110, y, 0.9)
    kp[p.OMBRO_DIR] = (90, y, 0.9)
    rad = math.radians(angulo)
    kp[p.NARIZ] = (100 + 40 * math.sin(rad), y - 40 * math.cos(rad), 0.9)
    return Pose(i, i / vs.FPS, kp, (50.0, 100.0, 150.0, 300.0))


def test_score_de_queda_explica_angulo_variacao_e_descida():
    angulos = [5] * 10 + [85] * 5
    seq = vs.sequencia([_cabeca(a, i, descida=0 if i < 10 else 40) for i, a in enumerate(angulos)])

    ereto, deitado = fall.scores(seq)[0], fall.scores(seq)[-1]

    assert ereto.score == 0
    assert deitado.angulo == pytest.approx(85)
    assert deitado.variacao == pytest.approx(80)
    assert deitado.componentes[:2] == (1.0, 1.0)
    assert deitado.score >= fall.SCORE_QUEDA


def test_camera_proxima_queda_de_lado_pelo_eixo_da_cabeca():
    angulos = [5] * 10 + [40, 75] + [85] * 15  # sentado → deitado de lado em 0,2 s
    seq = vs.sequencia([_cabeca(a, i, descida=0 if i < 10 else 80) for i, a in enumerate(angulos)])

    assert fall.detectar(seq) == []  # a caixa não mostra o corpo inteiro
    assert len(fall.detectar_proxima(seq)) == 1
    assert len(fall.detectar_todas(seq)) == 1


def test_camera_proxima_curvar_para_frente_nao_e_queda():
    angulos = [5] * 10 + [60, 120] + [160] * 15  # nariz abaixo dos ombros: se curvou
    seq = vs.sequencia([_cabeca(a, i) for i, a in enumerate(angulos)])

    assert fall.detectar_proxima(seq) == []


def test_camera_proxima_inclinar_e_voltar_logo_nao_e_queda():
    angulos = [5] * 10 + [70] * 5 + [10] * 15  # inclinou meio segundo e voltou
    seq = vs.sequencia([_cabeca(a, i) for i, a in enumerate(angulos)])

    assert fall.detectar_proxima(seq) == []


def test_camera_proxima_deitar_devagar_nao_e_queda():
    passos = int(8 * vs.FPS)  # 8 s de sentado a deitado, como quem se deita na cama
    angulos = [5] * 10 + [5 + 87 * k / passos for k in range(passos)] + [92] * 20
    seq = vs.sequencia([_cabeca(a, i) for i, a in enumerate(angulos)])

    assert fall.detectar_proxima(seq) == []


def test_camera_proxima_cair_para_fora_da_imagem():
    from dataclasses import replace

    descidas = [0] * 10 + [60, 120, 160]  # ombros descendo até a faixa de baixo do quadro
    poses = [_cabeca(5, i, descida=d) for i, d in enumerate(descidas)]
    sumiu = [vs.Pose(13 + k, (13 + k) / vs.FPS, None, None) for k in range(40)]  # 4 s fora
    seq = replace(vs.sequencia(poses + sumiu), altura=480)

    quedas = fall.detectar_saida_por_baixo(seq)

    assert len(quedas) == 1
    assert quedas[0].score is not None and quedas[0].score.saiu_por_baixo


def test_camera_proxima_sair_andando_nao_e_queda():
    from dataclasses import replace

    poses = [_cabeca(5, i) for i in range(13)]  # ombros no meio do quadro
    sumiu = [vs.Pose(13 + k, (13 + k) / vs.FPS, None, None) for k in range(40)]
    seq = replace(vs.sequencia(poses + sumiu), altura=480)

    assert fall.detectar_saida_por_baixo(seq) == []
