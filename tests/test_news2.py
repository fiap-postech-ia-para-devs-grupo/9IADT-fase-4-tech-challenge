import math

import pandas as pd
import pytest

from src.core import news2


@pytest.mark.parametrize(
    ("parametro", "valor", "esperado"),
    [
        ("FR", 8, 3),
        ("FR", 10, 1),
        ("FR", 16, 0),
        ("FR", 22, 2),
        ("FR", 25, 3),
        ("SpO2", 91, 3),
        ("SpO2", 93, 2),
        ("SpO2", 95, 1),
        ("SpO2", 98, 0),
        ("PAS", 90, 3),
        ("PAS", 95, 2),
        ("PAS", 105, 1),
        ("PAS", 130, 0),
        ("PAS", 220, 3),
        ("FC", 40, 3),
        ("FC", 45, 1),
        ("FC", 75, 0),
        ("FC", 100, 1),
        ("FC", 120, 2),
        ("FC", 131, 3),
    ],
)
def test_pontos_por_faixa(parametro, valor, esperado):
    assert news2.pontos(parametro, valor) == esperado


def test_paciente_estavel_e_risco_baixo():
    escore = news2.calcular({"FR": 16, "SpO2": 97, "PAS": 125, "FC": 80})

    assert escore.total == 0
    assert escore.risco == "baixo"


def test_um_parametro_vermelho_e_baixo_medio():
    escore = news2.calcular({"FR": 16, "SpO2": 90, "PAS": 125, "FC": 80})

    assert escore.total == 3
    assert escore.tem_parametro_vermelho
    assert escore.risco == "baixo-médio"


def test_piora_respiratoria_e_risco_alto():
    escore = news2.calcular({"FR": 26, "SpO2": 88, "PAS": 100, "FC": 115})

    assert escore.total == 3 + 3 + 2 + 2
    assert escore.risco == "alto"
    assert "emergência" in escore.resposta


def test_risco_medio_entre_5_e_6():
    assert news2.calcular({"FR": 22, "SpO2": 94, "PAS": 105, "FC": 95}).risco == "médio"


def test_parametro_sem_medida_fica_de_fora():
    escore = news2.calcular({"FR": math.nan, "SpO2": 97})

    assert escore.pontos == {"SpO2": 0}


def test_serie_preenche_lacunas_com_o_ultimo_valor():
    sinais = pd.DataFrame(
        {"FR": [16, math.nan, 30], "SpO2": [97, 97, 97], "PAS": [120, 120, 120], "FC": [80, 80, 80]}
    )

    assert list(news2.serie(sinais)) == [0, 0, 3]


def test_tendencia_gera_um_alerta_por_episodio_de_piora():
    indice = pd.date_range("2026-10-01 08:00", periods=90, freq="1min")
    sinais = pd.DataFrame({"FR": 16.0, "SpO2": 97.0, "PAS": 125.0, "FC": 80.0}, index=indice)
    sinais.iloc[30:, sinais.columns.get_loc("FR")] = 26.0  # +3
    sinais.iloc[30:, sinais.columns.get_loc("SpO2")] = 93.0  # +2 -> NEWS2 5

    alertas = news2.alertas_de_tendencia(sinais, "demo-x")

    assert len(alertas) == 1
    alerta = alertas[0]
    assert alerta.severidade.value == "alta"
    assert alerta.evidencia["tecnica"] == "news2_tendencia"
    assert alerta.evidencia["news2"] == 5
    assert "FR" in alerta.evidencia["serie"]


def test_escore_alto_estavel_desde_o_inicio_nao_e_tendencia():
    indice = pd.date_range("2026-10-01 08:00", periods=60, freq="1min")
    sinais = pd.DataFrame({"FR": 26.0, "SpO2": 93.0, "PAS": 125.0, "FC": 80.0}, index=indice)

    assert news2.alertas_de_tendencia(sinais, "demo-x") == []
