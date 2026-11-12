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
