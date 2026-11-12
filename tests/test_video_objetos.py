from src.analyzers.video import objetos
from src.analyzers.video.objetos import Objeto

PESSOA = (100.0, 200.0, 200.0, 260.0)  # 100 px de largura


def test_objeto_encostado_ou_perto_e_proximo():
    encostado = Objeto("cama", 0.9, (190.0, 150.0, 400.0, 300.0))
    perto = Objeto("cadeira", 0.6, (240.0, 200.0, 280.0, 260.0))  # a 40 px
    longe = Objeto("sofá", 0.9, (400.0, 200.0, 500.0, 260.0))  # a 200 px

    assert objetos.proximos([longe, perto, encostado], PESSOA) == ["cama", "cadeira"]


def test_mesmo_objeto_repetido_aparece_uma_vez():
    duas = [Objeto("cadeira", 0.5, (150.0, 200.0, 180.0, 250.0))] * 2

    assert objetos.proximos(duas, PESSOA) == ["cadeira"]


def test_sem_objetos_nada_proximo():
    assert objetos.proximos([], PESSOA) == []
