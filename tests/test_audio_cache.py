from dataclasses import dataclass

from src.analyzers.audio.cache import obter_ou_computar


def test_computa_na_primeira_chamada_e_le_do_cache_depois(tmp_path):
    caminho = tmp_path / "c.json"
    chamadas = []

    def computar():
        chamadas.append(1)
        return {"x": 1}

    assert obter_ou_computar(caminho, computar) == {"x": 1}
    assert obter_ou_computar(caminho, computar) == {"x": 1}
    assert len(chamadas) == 1, "a segunda chamada devia ler do cache, não reprocessar"


def test_grava_json_legivel_no_caminho(tmp_path):
    caminho = tmp_path / "sub" / "c.json"
    obter_ou_computar(caminho, lambda: {"a": 1})
    assert caminho.exists()
    assert '"a": 1' in caminho.read_text(encoding="utf-8")


def test_para_json_e_de_json_convertem_tipos_nao_nativos(tmp_path):
    @dataclass(frozen=True)
    class Ponto:
        x: int
        y: int

    caminho = tmp_path / "c.json"

    def computar() -> list[Ponto]:
        return [Ponto(1, 2), Ponto(3, 4)]

    resultado = obter_ou_computar(
        caminho,
        computar,
        para_json=lambda pontos: [{"x": p.x, "y": p.y} for p in pontos],
        de_json=lambda dados: [Ponto(**d) for d in dados],
    )
    assert resultado == [Ponto(1, 2), Ponto(3, 4)]

    def falharia():
        raise AssertionError("não devia computar de novo: já está em cache")

    do_cache = obter_ou_computar(
        caminho,
        falharia,
        para_json=lambda pontos: [{"x": p.x, "y": p.y} for p in pontos],
        de_json=lambda dados: [Ponto(**d) for d in dados],
    )
    assert do_cache == [Ponto(1, 2), Ponto(3, 4)]
