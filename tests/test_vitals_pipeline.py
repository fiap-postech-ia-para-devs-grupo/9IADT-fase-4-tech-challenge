import numpy as np
import pandas as pd
import pytest

from src.analyzers.vitals import evaluate, inject, load, stream
from src.core.alerts import Alerta, Origem, Severidade, Tipo

PACIENTE = "mimic3-100"


def janela_normal(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indice = pd.date_range("2150-01-01 00:00", periods=n, freq="min")
    return pd.DataFrame(
        {
            "FC": 80 + rng.normal(0, 1.0, n),
            "FR": 16 + rng.normal(0, 0.5, n),
            "SpO2": 97 + rng.normal(0, 0.3, n),
            "PAS": 120 + rng.normal(0, 2.0, n),
        },
        index=indice,
    )


# --- inject ---


def test_aplicar_registra_gabarito_no_trecho_certo():
    janela = janela_normal()
    injetada, gabarito = inject.aplicar(janela, [("pico_fc", 100)], duracao_min=10)

    assert (injetada["FC"].iloc[100:110] == janela["FC"].iloc[100:110] + 50).all()
    assert injetada["FC"].iloc[99] == janela["FC"].iloc[99]
    assert gabarito == [
        inject.Injecao(
            tipo="pico_fc",
            serie="FC",
            inicio=janela.index[100].to_pydatetime(),
            fim=janela.index[109].to_pydatetime(),
        )
    ]


def test_queda_spo2_chega_a_86_no_fim_do_trecho():
    janela = janela_normal()
    injetada, _ = inject.aplicar(janela, [("queda_spo2", 50)], duracao_min=20)

    assert injetada["SpO2"].iloc[69] == pytest.approx(86.0)


def test_planejar_respeita_trechos_proibidos_e_espacamento():
    janela = janela_normal(n=600)
    proibido = pd.Series(False, index=janela.index)
    proibido.iloc[:200] = True

    planos = inject.planejar(
        janela,
        ["pico_fc", "hipotensao_subita"],
        por_tipo=3,
        duracao_min=10,
        excluir=proibido,
        seed=1,
    )

    assert len(planos) == 6
    assert all(inicio >= 200 for _, inicio in planos)
    inicios = sorted(inicio for _, inicio in planos)
    assert all(
        b - a >= 10 + inject.ESPACAMENTO_MIN for a, b in zip(inicios, inicios[1:], strict=False)
    )


def test_planejar_pula_trechos_com_lacuna():
    janela = janela_normal(n=200)
    janela.loc[janela.index[:150], "FC"] = np.nan

    planos = inject.planejar(janela, ["pico_fc"], por_tipo=2, duracao_min=10, seed=2)

    assert all(inicio >= 150 for _, inicio in planos)


# --- evaluate ---


def test_avaliar_recall_e_precisao_com_regras_clinicas():
    janela = janela_normal(n=600)
    planos = [("pico_fc", 100), ("queda_spo2", 300), ("hipotensao_subita", 480)]
    injetada, gabarito = inject.aplicar(janela, planos, duracao_min=20)

    from src.analyzers.vitals import detectors

    alertas = {"regras_clinicas": detectors.detect(injetada, PACIENTE, "regras_clinicas")}
    tabela = evaluate.avaliar(alertas, gabarito, janela.index[0].to_pydatetime())

    linha = tabela.iloc[0]
    assert linha["tecnica"] == "regras_clinicas"
    assert linha["recall"] == pytest.approx(1.0)
    assert linha["precisao"] == pytest.approx(1.0)
    assert 0 <= linha["atraso_mediano_min"] <= 20


def test_avaliar_sem_alertas_tem_precisao_nan():
    janela = janela_normal(n=200)
    _, gabarito = inject.aplicar(janela, [("pico_fc", 50)], duracao_min=10)

    tabela = evaluate.avaliar({"zscore_movel": []}, gabarito, janela.index[0].to_pydatetime())

    assert tabela.iloc[0]["recall"] == 0
    assert np.isnan(tabela.iloc[0]["precisao"])


# --- stream ---


def test_replay_emite_cada_anomalia_uma_vez_no_minuto_em_que_aparece():
    janela = janela_normal(n=300)
    janela.loc[janela.index[200:210], "FC"] = 135
    esperados = []

    alertas = list(
        stream.replay(
            janela,
            PACIENTE,
            tecnicas=("regras_clinicas",),
            minutos_por_segundo=0,
            ao_detectar=esperados.append,
        )
    )

    assert len(alertas) == 1
    assert alertas[0].evidencia["serie"] == "FC"
    assert esperados == alertas


def test_replay_espera_um_segundo_por_minuto_quando_pedido():
    janela = janela_normal(n=5)
    pausas: list[float] = []

    list(
        stream.replay(janela, PACIENTE, tecnicas=(), minutos_por_segundo=1.0, esperar=pausas.append)
    )

    assert pausas == [1.0] * 5


# --- load ---


def test_de_sinais_escolhe_abp_antes_de_nbp_e_reamostra_por_minuto():
    indice = pd.date_range("2150-01-01 00:00", periods=240, freq="30s")
    brutos = pd.DataFrame(
        {
            "HR": 80.0,
            "RESP": 16.0,
            "SpO2": 97.0,
            "ABP Sys": 120.0,
            "NBP Sys": 150.0,
        },
        index=indice,
    )

    quadro = load.de_sinais(brutos)

    assert list(quadro.columns) == ["FC", "FR", "SpO2", "PAS"]
    assert len(quadro) == 120  # 240 amostras de 30 s = 120 minutos
    assert quadro["PAS"].iloc[0] == pytest.approx(120.0)


def test_de_sinais_coluna_ausente_vira_nan():
    indice = pd.date_range("2150-01-01 00:00", periods=120, freq="30s")
    brutos = pd.DataFrame({"HR": 80.0}, index=indice)

    quadro = load.de_sinais(brutos)

    assert quadro["SpO2"].isna().all()
    assert quadro["FC"].notna().all()


def test_reamostrar_interpola_lacuna_curta_e_mantem_longa():
    minutos = pd.date_range("2150-01-01 00:00", periods=30, freq="min")
    serie = pd.Series(np.arange(30, dtype=float), index=minutos)
    serie.iloc[5:8] = np.nan  # 3 min: interpolado
    serie.iloc[15:25] = np.nan  # 10 min: fica NaN

    saida = load.reamostrar(serie)

    assert not saida.iloc[5:8].isna().any()
    assert saida.iloc[15:25].isna().all()


# --- Alerta de sinal vital serializa bem para o replay ---


def test_alerta_de_sinal_vital_eh_valido_no_contrato():
    alerta = Alerta(
        paciente_id=PACIENTE,
        origem=Origem.SINAIS_VITAIS,
        tipo=Tipo.SINAL_VITAL,
        severidade=Severidade.ALTA,
        descricao="teste",
        evidencia={"serie": "FC", "janela": [0.0, 60.0], "valores": [130.0]},
        detectado_em=pd.Timestamp("2150-01-01 01:00").to_pydatetime(),
    )
    assert alerta.para_dict()["origem"] == "sinais_vitais"
