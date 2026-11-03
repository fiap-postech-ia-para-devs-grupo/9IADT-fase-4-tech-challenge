import struct
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analyzers.audio import acoustic
from src.core.alerts import Origem, Severidade, Tipo


def _wav_tom(caminho: Path, *, freq: float = 150.0, taxa: int = 16_000, segundos: float = 1.0):
    """Tom puro (não é voz real, mas basta para exercitar o extrator de features)."""
    n = int(taxa * segundos)
    t = np.arange(n) / taxa
    onda = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa)
        arquivo.writeframes(onda.tobytes())


def _wav_silencio(caminho: Path, *, taxa: int = 16_000, segundos: float = 1.0):
    n = int(taxa * segundos)
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa)
        arquivo.writeframes(struct.pack("<h", 0) * n)


# --- features ----------------------------------------------------------------------------


def test_extrair_features_devolve_as_colunas_esperadas(tmp_path):
    caminho = tmp_path / "tom.wav"
    _wav_tom(caminho)

    features = acoustic.extrair_features(caminho)

    esperadas = {f"mfcc{i}_media" for i in range(1, acoustic.N_MFCC + 1)} | {
        "f0_media",
        "f0_desvio",
        "jitter_local",
        "shimmer_local",
        "hnr_media",
        "proporcao_pausas",
        "taxa_fala",
    }
    assert esperadas <= set(features)
    assert all(isinstance(v, float) for v in features.values())


def test_extrair_features_detecta_f0_proximo_do_tom_puro(tmp_path):
    caminho = tmp_path / "tom.wav"
    _wav_tom(caminho, freq=150.0)

    features = acoustic.extrair_features(caminho)
    assert 120 <= features["f0_media"] <= 180


def test_extrair_features_nao_quebra_em_silencio(tmp_path):
    caminho = tmp_path / "silencio.wav"
    _wav_silencio(caminho)
    features = acoustic.extrair_features(caminho)
    assert features["f0_media"] == 0.0
    assert not np.isnan(features["jitter_local"])


# --- metadados e rótulo -------------------------------------------------------------------


def test_rotulo_sintomatico_por_bd_ou_ftg():
    saudavel_bd = pd.Series({"bd": "True", "ftg": "False", "covid_status": "healthy"})
    saudavel_ftg = pd.Series({"bd": "False", "ftg": "True", "covid_status": "healthy"})
    assert acoustic._rotulo(saudavel_bd) == 1
    assert acoustic._rotulo(saudavel_ftg) == 1


def test_rotulo_saudavel_limpo():
    linha = pd.Series({"bd": "False", "ftg": "False", "covid_status": "healthy"})
    assert acoustic._rotulo(linha) == 0


def test_rotulo_fora_do_subset_e_none():
    linha = pd.Series({"bd": "False", "ftg": "False", "covid_status": "positive_mild"})
    assert acoustic._rotulo(linha) is None


def test_carregar_metadados_calcula_rotulo(tmp_path):
    caminho = tmp_path / "combined_data.csv"
    pd.DataFrame(
        [
            {"id": "p1", "bd": "True", "ftg": "False", "covid_status": "positive_mild"},
            {"id": "p2", "bd": "False", "ftg": "False", "covid_status": "healthy"},
            {"id": "p3", "bd": "False", "ftg": "False", "covid_status": "positive_mild"},
        ]
    ).to_csv(caminho, index=False)

    metadados = acoustic.carregar_metadados(caminho)
    rotulos = metadados.set_index("id")["rotulo"]
    assert rotulos["p1"] == 1
    assert rotulos["p2"] == 0
    assert pd.isna(rotulos["p3"])


# --- dataset a partir de uma pasta extraída -----------------------------------------------


def _montar_pasta_extraida(base: Path, pessoas: dict[str, float]) -> Path:
    """`pessoas`: {id: frequência do tom} — cada pessoa recebe os 2 áudios úteis."""
    extraido = base / "Extracted_data" / "20200101"
    for pessoa_id, freq in pessoas.items():
        pasta = extraido / pessoa_id
        pasta.mkdir(parents=True)
        for nome in acoustic.ARQUIVOS_UTEIS:
            _wav_tom(pasta / nome, freq=freq, segundos=0.3)
    return base / "Extracted_data"


def test_localizar_arquivos_encontra_as_duas_gravacoes_por_pessoa(tmp_path):
    extraido = _montar_pasta_extraida(tmp_path, {"p1": 120, "p2": 200})
    disponiveis = acoustic.localizar_arquivos(extraido)
    assert set(disponiveis) == {"p1", "p2"}
    assert set(disponiveis["p1"]) == set(acoustic.ARQUIVOS_UTEIS)


def test_montar_dataset_uma_linha_por_gravacao(tmp_path):
    extraido = _montar_pasta_extraida(tmp_path, {"p1": 120, "p2": 200})
    metadados = pd.DataFrame(
        [
            {"id": "p1", "rotulo": 1},
            {"id": "p2", "rotulo": 0},
            {"id": "p3", "rotulo": 0},  # sem áudio extraído: fica de fora
        ]
    )

    dataset = acoustic.montar_dataset(metadados, extraido)

    assert len(dataset) == 4  # 2 pessoas x 2 gravações
    assert set(dataset["id"]) == {"p1", "p2"}
    assert set(dataset["arquivo"]) == set(acoustic.ARQUIVOS_UTEIS)


# --- split sem vazamento entre gravações da mesma pessoa -----------------------------------


def _dataset_sintetico(n_por_classe: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    linhas = []
    for rotulo in (0, 1):
        for i in range(n_por_classe):
            pessoa = f"{rotulo}_{i}"
            base = 110 if rotulo == 0 else 220  # classes bem separadas: AUC alto esperado
            for arquivo in acoustic.ARQUIVOS_UTEIS:
                linhas.append(
                    {
                        "id": pessoa,
                        "arquivo": arquivo,
                        "rotulo": rotulo,
                        "f0_media": base + rng.normal(0, 5),
                        "jitter_local": rng.uniform(0, 0.01),
                    }
                )
    return pd.DataFrame(linhas)


def test_dividir_por_pessoa_nao_mistura_gravacoes_da_mesma_pessoa():
    dataset = _dataset_sintetico()
    treino, teste = acoustic.dividir_por_pessoa(dataset, teste=0.3, seed=0)
    assert set(treino["id"]) & set(teste["id"]) == set()
    assert len(treino) + len(teste) == len(dataset)


def test_balancear_por_pessoa_limita_cada_classe():
    metadados = pd.DataFrame(
        {"id": [f"p{i}" for i in range(10)], "rotulo": [1] * 7 + [0] * 3}
    )
    balanceado = acoustic.balancear_por_pessoa(metadados, por_classe=3, seed=0)
    contagem = balanceado["rotulo"].value_counts()
    assert contagem[1] == 3
    assert contagem[0] == 3


# --- treino, avaliação e inferência ---------------------------------------------------------


def test_treinar_avaliar_separa_classes_bem_distintas():
    dataset = _dataset_sintetico(n_por_classe=15)
    treino, teste = acoustic.dividir_por_pessoa(dataset, teste=0.3, seed=0)

    resultados, contexto = acoustic.treinar_avaliar(treino, teste, seed=0)

    assert set(resultados) == {"regressao_logistica", "random_forest"}
    for resultado in resultados.values():
        assert resultado.auc > 0.9  # classes bem separadas na sintética
        assert 0.0 <= resultado.limiar <= 1.0


def test_salvar_e_carregar_modelo_preserva_previsao(tmp_path):
    dataset = _dataset_sintetico(n_por_classe=15)
    treino, teste = acoustic.dividir_por_pessoa(dataset, teste=0.3, seed=0)
    resultados, contexto = acoustic.treinar_avaliar(treino, teste, seed=0)
    melhor = max(resultados, key=lambda nome: resultados[nome].auc)

    caminho = tmp_path / "modelo.joblib"
    acoustic.salvar_modelo(melhor, resultados, contexto, caminho)
    carregado = acoustic.carregar_modelo(caminho)

    linha = teste.iloc[[0]][contexto["colunas"]].to_dict("records")[0]
    assert acoustic.prever_com_features(linha, carregado) == pytest.approx(
        contexto["modelos"][melhor].predict_proba(teste[contexto["colunas"]].iloc[[0]])[0, 1]
    )
    assert carregado["limiar"] == resultados[melhor].limiar


# --- Alerta de Alteração Vocal --------------------------------------------------------------


def test_alerta_alteracao_vocal_none_abaixo_do_limiar():
    assert acoustic.alerta_alteracao_vocal("p1", "c1.wav", 0.3, limiar=0.5) is None


def test_alerta_alteracao_vocal_no_contrato_quando_acima_do_limiar():
    alerta = acoustic.alerta_alteracao_vocal(
        "demo-01", "c1.wav", 0.81, limiar=0.5, trecho="I feel so tired", inicio=12.0,
        detectado_em=datetime(2026, 10, 1, 8, 0),
    )
    assert alerta is not None
    assert alerta.origem is Origem.AUDIO
    assert alerta.tipo is Tipo.ALTERACAO_VOCAL
    assert alerta.severidade is Severidade.ALTA
    assert alerta.evidencia["probabilidade"] == 0.81
    assert alerta.evidencia["trecho"] == "I feel so tired"


def test_alerta_alteracao_vocal_e_deterministico():
    a1 = acoustic.alerta_alteracao_vocal("p1", "c1.wav", 0.9, limiar=0.5)
    a2 = acoustic.alerta_alteracao_vocal("p1", "c1.wav", 0.9, limiar=0.5)
    assert a1.id == a2.id
