import pytest

from src.analyzers.audio import acoustic, lab, semantic
from src.analyzers.audio.semantic import EntidadeClinica, SentimentoTrecho, Trecho
from src.core.alerts import Origem, Tipo

TRECHOS = [Trecho(0.0, 2.0, "I feel really tired and short of breath")]


@pytest.fixture(autouse=True)
def _mocka_azure(monkeypatch, tmp_path):
    """Nenhum teste deste arquivo chama o Azure de verdade: tudo entra/saí por mock."""
    monkeypatch.setattr(semantic, "converter_wav", lambda origem, destino: origem)
    monkeypatch.setattr(semantic, "transcrever", lambda *a, **kw: TRECHOS)
    monkeypatch.setattr(
        semantic,
        "extrair_entidades_saude",
        lambda *a, **kw: [[EntidadeClinica("short of breath", "SymptomOrSign", 0.9)]],
    )
    monkeypatch.setattr(
        semantic,
        "analisar_sentimento",
        lambda *a, **kw: [SentimentoTrecho(TRECHOS[0], "negative", 0.1, 0.2, 0.7)],
    )
    monkeypatch.setattr(acoustic, "extrair_features", lambda caminho: {"f0_media": 180.0})


def test_analyze_audio_sem_modelo_treinado_nao_tem_probabilidade(tmp_path):
    arquivo = tmp_path / "consulta.wav"
    arquivo.write_bytes(b"\x00")

    resultado = lab.analyze_audio(
        arquivo,
        config=object(),
        cache_dir=tmp_path / "cache",
        caminho_modelo=tmp_path / "nao-existe.joblib",
    )

    assert resultado.probabilidade_alteracao_vocal is None
    assert len(resultado.alertas) == 1
    assert resultado.alertas[0].tipo is Tipo.TERMO_CRITICO
    assert resultado.texto_completo == "I feel really tired and short of breath"


def test_analyze_audio_com_modelo_gera_alerta_de_alteracao_vocal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acoustic,
        "carregar_modelo",
        lambda caminho: {"modelo": object(), "colunas": ["f0_media"], "limiar": 0.5},
    )
    monkeypatch.setattr(acoustic, "prever_com_features", lambda features, modelo: 0.9)

    arquivo = tmp_path / "consulta.wav"
    arquivo.write_bytes(b"\x00")
    caminho_modelo = tmp_path / "modelo.joblib"
    caminho_modelo.write_bytes(b"\x00")

    resultado = lab.analyze_audio(
        arquivo,
        config=object(),
        paciente_id="demo-01",
        cache_dir=tmp_path / "cache",
        caminho_modelo=caminho_modelo,
    )

    assert resultado.probabilidade_alteracao_vocal == 0.9
    tipos = {a.tipo for a in resultado.alertas}
    assert tipos == {Tipo.TERMO_CRITICO, Tipo.ALTERACAO_VOCAL}
    vocal = next(a for a in resultado.alertas if a.tipo is Tipo.ALTERACAO_VOCAL)
    assert vocal.paciente_id == "demo-01"
    assert vocal.origem is Origem.AUDIO


def test_analyze_audio_com_modelo_mas_probabilidade_abaixo_do_limiar_nao_gera_alerta_vocal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        acoustic,
        "carregar_modelo",
        lambda caminho: {"modelo": object(), "colunas": ["f0_media"], "limiar": 0.5},
    )
    monkeypatch.setattr(acoustic, "prever_com_features", lambda features, modelo: 0.1)

    arquivo = tmp_path / "consulta.wav"
    arquivo.write_bytes(b"\x00")
    caminho_modelo = tmp_path / "modelo.joblib"
    caminho_modelo.write_bytes(b"\x00")

    resultado = lab.analyze_audio(
        arquivo, config=object(), cache_dir=tmp_path / "cache", caminho_modelo=caminho_modelo
    )

    assert resultado.probabilidade_alteracao_vocal == 0.1
    assert all(a.tipo is not Tipo.ALTERACAO_VOCAL for a in resultado.alertas)


def test_uploads_de_mesmo_nome_e_conteudo_diferente_nao_compartilham_identidade(
    tmp_path, monkeypatch
):
    identidades = []

    def fake_entidades(trechos, config, cache_dir, consulta_id):
        identidades.append(consulta_id)
        return [[]]

    monkeypatch.setattr(semantic, "extrair_entidades_saude", fake_entidades)
    for pasta, conteudo in (("a", b"\x00"), ("b", b"\x01"), ("c", b"\x00")):
        (tmp_path / pasta).mkdir()
        arquivo = tmp_path / pasta / "gravacao.wav"
        arquivo.write_bytes(conteudo)
        lab.analyze_audio(
            arquivo, config=object(), cache_dir=tmp_path / "cache", caminho_modelo=tmp_path / "x"
        )

    assert identidades[0] != identidades[1]  # conteúdo diferente: não reaproveita cache
    assert identidades[0] == identidades[2]  # mesmo conteúdo: reaproveita, não gasta cota
