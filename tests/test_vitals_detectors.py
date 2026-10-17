import numpy as np
import pandas as pd
import pytest

from src.analyzers.vitals import detectors
from src.core.alerts import Origem, Severidade, Tipo

PACIENTE = "mimic3-100"
MINUTOS_DE_HISTORIA = detectors.MIN_HISTORIA_ZSCORE


def janela_normal(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Sinais vitais de um paciente estável: 1 amostra/min, ruído pequeno em torno da média."""
    rng = np.random.default_rng(seed)
    dados = {
        "FC": 80 + rng.normal(0, 1.0, n),
        "FR": 16 + rng.normal(0, 0.5, n),
        "SpO2": 97 + rng.normal(0, 0.3, n),
        "PAS": 120 + rng.normal(0, 2.0, n),
    }
    indice = pd.date_range("2150-01-01 00:00", periods=n, freq="min")
    return pd.DataFrame(dados, index=indice)


def series_de(alertas, tecnica):
    return [
        (a.evidencia["serie"], a.evidencia["janela"])
        for a in alertas
        if a.evidencia["tecnica"] == tecnica
    ]


# --- contrato do Alerta ---


def test_alertas_seguem_o_contrato_do_alerta():
    janela = janela_normal()
    janela.loc[janela.index[200:205], "FC"] = 135
    alertas = detectors.detect(janela, PACIENTE, "regras_clinicas")

    assert alertas
    for alerta in alertas:
        assert alerta.origem is Origem.SINAIS_VITAIS
        assert alerta.tipo is Tipo.SINAL_VITAL
        assert alerta.paciente_id == PACIENTE
        assert {"tecnica", "serie", "janela", "valores"} <= alerta.evidencia.keys()


def test_ids_sao_deterministicos():
    janela = janela_normal()
    janela.loc[janela.index[200:205], "FC"] = 135
    primeiros = detectors.detect(janela, PACIENTE, "regras_clinicas")
    segundos = detectors.detect(janela, PACIENTE, "regras_clinicas")

    assert [a.id for a in primeiros] == [a.id for a in segundos]


# --- regras clínicas ---


def test_regra_dispara_so_no_episodio_fora_do_limite():
    janela = janela_normal()
    janela.loc[janela.index[200:205], "FC"] = 135  # 5 min acima de 120

    alertas = detectors.regras_clinicas(janela, PACIENTE)

    assert len(alertas) == 1
    alerta = alertas[0]
    assert alerta.evidencia["serie"] == "FC"
    assert alerta.evidencia["janela"] == [200 * 60.0, 204 * 60.0]
    assert alerta.severidade is Severidade.ALTA
    assert alerta.detectado_em == janela.index[204].to_pydatetime()


def test_spo2_abaixo_de_90_e_critica():
    janela = janela_normal()
    janela.loc[janela.index[300:303], "SpO2"] = 85

    alertas = detectors.regras_clinicas(janela, PACIENTE)

    assert [(a.evidencia["serie"], a.severidade) for a in alertas] == [("SpO2", Severidade.CRITICA)]


def test_paciente_estavel_nao_dispara_regras():
    assert detectors.regras_clinicas(janela_normal(), PACIENTE) == []


def test_lacuna_quebra_o_episodio_em_dois_alertas():
    janela = janela_normal()
    janela.loc[janela.index[200:205], "FR"] = 30
    janela.loc[janela.index[205], "FR"] = np.nan  # lacuna no meio do episódio
    janela.loc[janela.index[206:210], "FR"] = 30

    alertas = detectors.regras_clinicas(janela, PACIENTE)

    assert [a.evidencia["janela"] for a in alertas] == [
        [200 * 60.0, 204 * 60.0],
        [206 * 60.0, 209 * 60.0],
    ]


# --- z-score móvel ---


def test_zscore_pega_pico_que_nao_passa_do_limite_clinico():
    janela = janela_normal()
    janela.loc[janela.index[200], "FR"] = 22  # FR 22 não cruza 25, mas é ~12σ acima da média

    alertas = detectors.zscore_movel(janela, PACIENTE)

    assert series_de(alertas, "zscore_movel") == [("FR", [200 * 60.0, 200 * 60.0])]
    assert alertas[0].severidade is Severidade.MEDIA
    assert alertas[0].evidencia["z_pico"] > 3


def test_zscore_ignora_os_primeiros_minutos_sem_historia():
    janela = janela_normal()
    janela.loc[janela.index[5], "FR"] = 40  # antes de haver base estatística

    alertas = detectors.zscore_movel(janela, PACIENTE)

    assert all(a.evidencia["janela"][0] >= MINUTOS_DE_HISTORIA * 60 for a in alertas)


def test_zscore_ignora_variacao_pequena_mesmo_com_sigma_baixo():
    janela = janela_normal()
    janela.loc[janela.index[200], "FR"] = 17.5  # ~3σ, mas só 1.5 rpm acima da média

    assert ("FR", [200 * 60.0, 200 * 60.0]) not in series_de(
        detectors.zscore_movel(janela, PACIENTE), "zscore_movel"
    )


def test_zscore_nao_usa_o_proprio_ponto_na_media():
    janela = janela_normal()
    janela.loc[janela.index[200], "PAS"] = 200  # se o ponto entrasse na média, o z cairia

    alertas = detectors.zscore_movel(janela, PACIENTE)

    assert ("PAS", [200 * 60.0, 200 * 60.0]) in series_de(alertas, "zscore_movel")


# --- Isolation Forest ---


def test_isolation_forest_marca_anomalia_conjunta():
    janela = janela_normal(n=600)
    # Cada sinal dentro da faixa clínica, mas o conjunto (taquicardia + dessaturação
    # + hipotensão) não aparece em nenhum minuto da história.
    trecho = janela.index[400:410]
    janela.loc[trecho, "FC"] = 105
    janela.loc[trecho, "SpO2"] = 92
    janela.loc[trecho, "PAS"] = 95

    alertas = detectors.isolation_forest(janela, PACIENTE)

    assert alertas, "o Isolation Forest deveria marcar o trecho conjunto"
    marcados = set()
    for alerta in alertas:
        ini, fim = alerta.evidencia["janela"]
        marcados.update(range(int(ini // 60), int(fim // 60) + 1))
    assert marcados & set(range(400, 410))
    assert all(a.severidade is Severidade.BAIXA for a in alertas)


def test_isolation_forest_recusa_janela_curta():
    assert detectors.isolation_forest(janela_normal(n=20), PACIENTE) == []


# --- interface ---


def test_tecnica_desconhecida_levanta_erro():
    with pytest.raises(ValueError, match="desconhecida"):
        detectors.detect(janela_normal(), PACIENTE, "magica")


def test_janela_sem_datetime_index_levanta_erro():
    janela = janela_normal().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        detectors.detect(janela, PACIENTE, "zscore_movel")
