import struct
import wave
from datetime import datetime
from pathlib import Path

from src.analyzers.audio import semantic
from src.analyzers.audio.semantic import EntidadeClinica, SentimentoTrecho, Trecho
from src.core.alerts import Origem, Severidade, Tipo

# --- converter_wav (ffmpeg real, sem rede) --------------------------------------------------


def _wav_sintetico(caminho: Path, *, taxa: int = 44_100, canais: int = 2, segundos: float = 0.2):
    n = int(taxa * segundos)
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(canais)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa)
        amostra = struct.pack("<h", 1000) * canais
        arquivo.writeframes(amostra * n)


def test_converter_wav_produz_16khz_mono(tmp_path):
    origem = tmp_path / "origem.wav"
    _wav_sintetico(origem)
    destino = tmp_path / "destino.wav"

    resultado = semantic.converter_wav(origem, destino)

    assert resultado == destino
    with wave.open(str(destino), "rb") as saida:
        assert saida.getframerate() == 16_000
        assert saida.getnchannels() == 1


def test_converter_wav_e_idempotente(tmp_path, monkeypatch):
    origem = tmp_path / "origem.wav"
    _wav_sintetico(origem)
    destino = tmp_path / "destino.wav"
    semantic.converter_wav(origem, destino)
    conteudo_antes = destino.read_bytes()

    def falharia(*a, **kw):
        raise AssertionError("não devia converter de novo: destino já existe")

    monkeypatch.setattr("pydub.AudioSegment.from_file", falharia)
    semantic.converter_wav(origem, destino)
    assert destino.read_bytes() == conteudo_antes


# --- TextGrid --------------------------------------------------------------------------------

TEXTGRID_EXEMPLO = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Patient"
        xmin = 0
        xmax = 10.0
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 2.0
            text = ""
        intervals [2]:
            xmin = 2.0
            xmax = 5.0
            text = "<UNSURE>Oh</UNSURE> hey, I have chest pain."
        intervals [3]:
            xmin = 5.0
            xmax = 9.0
            text = "It hurts when I breathe, <UNIN/>."
"""


def test_ler_textgrid_ignora_intervalos_vazios(tmp_path):
    caminho = tmp_path / "exemplo.TextGrid"
    caminho.write_text(TEXTGRID_EXEMPLO, encoding="utf-8")

    trechos = semantic.ler_textgrid(caminho)

    assert len(trechos) == 2
    assert trechos[0] == Trecho(2.0, 5.0, "<UNSURE>Oh</UNSURE> hey, I have chest pain.")
    assert trechos[1].inicio == 5.0


def test_calcular_wer_identico_e_zero():
    trechos = [Trecho(0, 1, "I have chest pain")]
    assert semantic.calcular_wer(trechos, trechos) == 0.0


def test_calcular_wer_ignora_tags_e_pontuacao_do_textgrid():
    hipotese = [Trecho(0, 1, "oh hey I have chest pain")]
    referencia = [Trecho(0, 1, "<UNSURE>Oh</UNSURE> hey, I have chest pain.")]
    assert semantic.calcular_wer(hipotese, referencia) == 0.0


def test_calcular_wer_conta_substituicoes():
    hipotese = [Trecho(0, 1, "I have chest ache")]
    referencia = [Trecho(0, 1, "I have chest pain")]
    assert semantic.calcular_wer(hipotese, referencia) > 0.0


# --- termos críticos ---------------------------------------------------------------------


def test_termos_criticos_gera_alerta_no_contrato_com_trecho_e_timestamp():
    trechos = [Trecho(83.2, 97.0, "I get really short of breath when I walk")]
    entidades = [
        [EntidadeClinica(texto="short of breath", categoria="SymptomOrSign", confianca=0.9)]
    ]
    inicio = datetime(2026, 10, 1, 8, 0)

    [alerta] = semantic.termos_criticos(
        trechos, entidades, "demo-01", "day1_consultation01", inicio
    )

    assert alerta.origem is Origem.AUDIO
    assert alerta.tipo is Tipo.TERMO_CRITICO
    assert alerta.paciente_id == "demo-01"
    assert alerta.evidencia["trecho"] == trechos[0].texto
    assert alerta.evidencia["inicio"] == 83.2
    assert alerta.evidencia["categoria"] == "respiratorio"
    assert alerta.severidade is Severidade.ALTA
    assert alerta.detectado_em == datetime(2026, 10, 1, 8, 1, 23, 200000)


def test_termos_criticos_ignora_categorias_azure_irrelevantes():
    trechos = [Trecho(0, 1, "taking 81 mg aspirin daily")]
    entidades = [[EntidadeClinica(texto="81 mg", categoria="Dosage", confianca=0.9)]]

    alertas = semantic.termos_criticos(trechos, entidades, "p1", "c1", datetime.now())
    assert alertas == []


def test_termos_criticos_sem_termo_curado_nao_gera_alerta():
    trechos = [Trecho(0, 1, "patient has a mild rash")]
    entidades = [[EntidadeClinica(texto="rash", categoria="SymptomOrSign", confianca=0.9)]]

    assert semantic.termos_criticos(trechos, entidades, "p1", "c1", datetime.now()) == []


def test_termos_criticos_e_deterministico():
    trechos = [Trecho(10.0, 12.0, "I have fatigue")]
    entidades = [[EntidadeClinica(texto="fatigue", categoria="SymptomOrSign", confianca=0.8)]]
    inicio = datetime(2026, 10, 1, 8, 0)

    a1 = semantic.termos_criticos(trechos, entidades, "p1", "c1", inicio)
    a2 = semantic.termos_criticos(trechos, entidades, "p1", "c1", inicio)
    assert [a.id for a in a1] == [a.id for a in a2]
    assert a1[0].severidade is Severidade.MEDIA  # sem categoria de Fusão -> severidade menor
    assert "categoria" not in a1[0].evidencia


# --- transcrever / entidades / sentimento: Azure mockado + cache ---------------------------


def test_transcrever_usa_cache_na_segunda_chamada(tmp_path, monkeypatch):
    chamadas = []

    def fake_reconhecer(caminho_wav, config, idioma=semantic.IDIOMA):
        chamadas.append(1)
        return [Trecho(0.0, 1.0, "hello doctor")]

    monkeypatch.setattr(semantic, "_reconhecer_continuo", fake_reconhecer)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"\x00")
    config = object()

    r1 = semantic.transcrever(audio, config, tmp_path)
    r2 = semantic.transcrever(audio, config, tmp_path)

    assert r1 == r2 == [Trecho(0.0, 1.0, "hello doctor")]
    assert len(chamadas) == 1


def test_extrair_entidades_saude_usa_cache(tmp_path, monkeypatch):
    trechos = [Trecho(0, 1, "I have chest pain")]
    chamadas = []

    def fake_extrair(trechos_, config):
        chamadas.append(1)
        return [[EntidadeClinica("chest pain", "SymptomOrSign", 0.9)]]

    monkeypatch.setattr(semantic, "_extrair_entidades_azure", fake_extrair)
    config = object()

    e1 = semantic.extrair_entidades_saude(trechos, config, tmp_path, "c1")
    e2 = semantic.extrair_entidades_saude(trechos, config, tmp_path, "c1")

    assert e1 == e2 == [[EntidadeClinica("chest pain", "SymptomOrSign", 0.9)]]
    assert len(chamadas) == 1


def test_analisar_sentimento_usa_cache_e_mapeia_por_trecho(tmp_path, monkeypatch):
    trechos = [Trecho(0, 1, "I am worried")]

    def fake_sentimento(trechos_, config):
        return [{"sentimento": "negative", "positivo": 0.1, "neutro": 0.1, "negativo": 0.8}]

    monkeypatch.setattr(semantic, "_analisar_sentimento_azure", fake_sentimento)
    config = object()

    [resultado] = semantic.analisar_sentimento(trechos, config, tmp_path, "c1")
    assert resultado == SentimentoTrecho(trechos[0], "negative", 0.1, 0.1, 0.8)
