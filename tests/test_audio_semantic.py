import struct
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from azure.ai.textanalytics import DocumentError

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


def test_calcular_wer_sem_hesitacoes_ignora_um_e_uh_nas_duas_transcricoes():
    hipotese = [Trecho(0, 1, "I have pain")]
    referencia = [Trecho(0, 1, "Um, I have, uh, pain")]

    assert semantic.calcular_wer(hipotese, referencia) > 0
    assert semantic.calcular_wer(hipotese, referencia, sem_hesitacoes=True) == 0


def test_calcular_wer_sem_hesitacoes_preserva_palavras_parecidas():
    trechos = [Trecho(0, 1, "umbrella error")]

    assert semantic.calcular_wer(trechos, trechos, sem_hesitacoes=True) == 0
    assert semantic.calcular_wer([Trecho(0, 1, "umbrella")], trechos, sem_hesitacoes=True) == 0.5


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


# --- limites e falhas da API do Language ------------------------------------------------------


class _ClienteFalso:
    def __init__(self, respostas=None):
        self.lotes: list[int] = []
        self.respostas = respostas

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def analyze_sentiment(self, documentos, language):
        self.lotes.append(len(documentos))
        if self.respostas is not None:
            return self.respostas
        pontos = SimpleNamespace(positive=0.0, neutral=1.0, negative=0.0)
        return [SimpleNamespace(sentiment="neutral", confidence_scores=pontos) for _ in documentos]


def _trechos(n: int) -> list[Trecho]:
    return [Trecho(i, i + 1, "texto") for i in range(n)]


def test_sentimento_respeita_limite_de_10_documentos_por_chamada(monkeypatch):
    cliente = _ClienteFalso()
    monkeypatch.setattr(semantic, "_cliente_language", lambda _config: cliente)

    resultado = semantic._analisar_sentimento_azure(_trechos(23), object())

    assert cliente.lotes == [10, 10, 3]
    assert len(resultado) == 23


def test_documento_recusado_falha_e_nao_vai_para_o_cache(tmp_path, monkeypatch):
    erro = DocumentError(id="0", error=SimpleNamespace(code="x", message="recusado"))
    cliente = _ClienteFalso(respostas=[erro])
    monkeypatch.setattr(semantic, "_cliente_language", lambda _config: cliente)

    with pytest.raises(RuntimeError, match="recusou"):
        semantic.analisar_sentimento(_trechos(1), object(), tmp_path, "c1")

    assert not (tmp_path / "c1.sentiment.json").exists()


def test_id_do_alerta_de_termo_critico_depende_do_paciente():
    trechos = [Trecho(1.0, 2.0, "chest pain")]
    entidades = [[EntidadeClinica("chest pain", "SymptomOrSign", 0.9)]]
    inicio = datetime(2026, 1, 1)

    [a] = semantic.termos_criticos(trechos, entidades, "avulso", "c1", inicio)
    [b] = semantic.termos_criticos(trechos, entidades, "demo-01", "c1", inicio)

    assert a.id != b.id


# --- negação e sujeito da entidade ------------------------------------------------------------


def _alertas_para(entidade: EntidadeClinica, texto: str = "trecho"):
    return semantic.termos_criticos(
        [Trecho(0.0, 1.0, texto)], [[entidade]], "p1", "c1", datetime(2026, 1, 1)
    )


def test_sintoma_negado_nao_gera_alerta():
    negada = EntidadeClinica("fever", "SymptomOrSign", 0.9, certeza="negative")
    assert _alertas_para(negada) == []


def test_sintoma_de_terceiro_nao_gera_alerta():
    familiar = EntidadeClinica("vomiting", "SymptomOrSign", 0.9, associacao="other")
    assert _alertas_para(familiar) == []


def test_sintoma_afirmado_do_paciente_gera_alerta():
    afirmada = EntidadeClinica("fever", "SymptomOrSign", 0.9, "positive", "subject")
    assert len(_alertas_para(afirmada)) == 1


def test_hay_fever_nao_gera_alerta_de_fever():
    entidade = EntidadeClinica("fever", "SymptomOrSign", 0.9, "positive", "subject")
    assert _alertas_para(entidade, "Sorry, I have hay fever.") == []


def test_extracao_guarda_certeza_e_associacao(monkeypatch):
    assercao = SimpleNamespace(certainty="negative", association="subject")
    entidade = SimpleNamespace(
        text="fever", category="SymptomOrSign", confidence_score=0.9, assertion=assercao
    )
    sem_assercao = SimpleNamespace(
        text="cough", category="SymptomOrSign", confidence_score=0.8, assertion=None
    )

    class Cliente(_ClienteFalso):
        def begin_analyze_healthcare_entities(self, documentos, language):
            doc = SimpleNamespace(entities=[entidade, sem_assercao])
            return SimpleNamespace(result=lambda: [doc])

    monkeypatch.setattr(semantic, "_cliente_language", lambda _config: Cliente())

    [[com, sem]] = semantic._extrair_entidades_azure(_trechos(1), object())

    assert (com.certeza, com.associacao) == ("negative", "subject")
    assert (sem.certeza, sem.associacao) == (None, None)


def test_termo_repetido_no_mesmo_trecho_gera_um_alerta():
    entidade = EntidadeClinica("tired", "SymptomOrSign", 0.9)
    trechos = [Trecho(0.0, 1.0, "tired, so tired")]
    alertas = semantic.termos_criticos(
        trechos, [[entidade, entidade]], "p1", "c1", datetime(2026, 1, 1)
    )
    assert len(alertas) == 1


def test_processar_consulta_do_cache_dispensa_audio_e_azure(tmp_path, monkeypatch):
    """Avaliador sem PriMock57, ffmpeg nem chaves: o cache versionado basta."""
    cache = Path("data/audio/cache_primock57")
    if not (cache / "day3_consultation04_patient_16k.stt.json").exists():
        pytest.skip("cache versionado do PriMock57 ausente")

    def falhar(*_):
        raise AssertionError("não deveria converter nem chamar o Azure")

    monkeypatch.setattr(semantic, "converter_wav", falhar)
    monkeypatch.setattr(semantic, "_reconhecer_continuo", falhar)

    resultado = semantic.processar_consulta(
        "day3_consultation04",
        tmp_path / "primock57-ausente",
        semantic.AzureConfig("", "", "", ""),
        cache,
        inicio=datetime(2026, 10, 1, 10, 0),
    )

    assert resultado.wer is None
    assert any(
        a.evidencia.get("categoria") == "respiratorio" for a in resultado.alertas_termo_critico
    )
