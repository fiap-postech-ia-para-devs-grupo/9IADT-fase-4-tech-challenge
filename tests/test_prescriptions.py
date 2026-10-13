from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.analyzers.prescriptions import listas, load, rules
from src.core.alerts import Origem, Severidade, Tipo

T0 = datetime(2150, 1, 1, 8, 0)


def prescricoes(*linhas: dict) -> pd.DataFrame:
    """Prescrições mínimas no formato de `load.carregar`; `h` é o offset de início em horas."""
    registros = []
    for linha in linhas:
        inicio = T0 + timedelta(hours=linha["h"])
        registros.append(
            {
                "subject_id": linha.get("subject_id", 1),
                "hadm_id": linha.get("hadm_id", 10),
                "starttime": inicio,
                "stoptime": inicio + timedelta(hours=linha.get("duracao_h", 24)),
                "drug": linha["drug"],
                "dose": linha.get("dose"),
                "dose_unit_rx": linha.get("unidade", "mg"),
                "route": linha.get("route", "IV"),
            }
        )
    return pd.DataFrame(registros)


def regras_de(alertas):
    return [a.evidencia["regra"] for a in alertas]


# --- listas ---


def test_identificar_ignora_maiusculas_e_nomes_comerciais():
    assert listas.alta_vigilancia("HYDROmorphone (Dilaudid)") == ("opioide", "hydromorphone")
    assert listas.alta_vigilancia("Heparin Sodium") == ("anticoagulante", "heparin")
    assert listas.alta_vigilancia("Acetaminophen") is None


def test_heparin_flush_nao_e_anticoagulacao():
    assert listas.alta_vigilancia("Heparin Flush (10 units/ml)") is None
    assert listas.classe_terapeutica("Sodium Chloride 0.9%  Flush") is None


def test_metoprolol_tartrato_e_succinato_sao_o_mesmo_principio_ativo():
    a = listas.classe_terapeutica("Metoprolol Tartrate")
    b = listas.classe_terapeutica("Metoprolol Succinate XL")
    assert a == b == ("beta_bloqueador", "metoprolol")


# --- carga ---


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [("25", 25.0), ("0.5", 0.5), ("25,000", 25000.0), ("5-10", 10.0), ("0.5-1", 1.0)],
)
def test_dose_numerica(valor, esperado):
    assert load.dose_numerica(valor) == esperado


@pytest.mark.parametrize("valor", [None, float("nan"), "", "UNIT"])
def test_dose_numerica_sem_numero(valor):
    assert load.dose_numerica(valor) is None


# --- salto de dose ---


def test_salto_de_dose_acima_de_50_por_cento_em_menos_de_24h():
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Metoprolol Tartrate", "dose": 25},
            {"h": 12, "drug": "Metoprolol Tartrate", "dose": 50},
        )
    )

    assert len(alertas) == 1
    alerta = alertas[0]
    assert alerta.origem is Origem.PRESCRICAO
    assert alerta.tipo is Tipo.MUDANCA_TERAPEUTICA
    assert alerta.severidade is Severidade.MEDIA
    assert alerta.detectado_em == T0 + timedelta(hours=12)
    assert alerta.evidencia["regra"] == "salto_de_dose"
    assert (alerta.evidencia["dose_anterior"], alerta.evidencia["dose_nova"]) == (25, 50)
    assert alerta.paciente_id == "mimic4-1"


def test_salto_de_dose_em_alta_vigilancia_e_mais_grave():
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Morphine Sulfate", "dose": 2},
            {"h": 6, "drug": "Morphine Sulfate", "dose": 4},
        )
    )
    assert [a.severidade for a in alertas] == [Severidade.ALTA]


@pytest.mark.parametrize(
    ("segundo", "motivo"),
    [
        ({"h": 12, "dose": 37}, "aumento de 48%, abaixo do limite"),
        ({"h": 12, "dose": 50, "route": "PO"}, "via diferente"),
        ({"h": 12, "dose": 50, "unidade": "mcg"}, "unidade diferente"),
        ({"h": 30, "dose": 50}, "mais de 24h entre as prescrições"),
        ({"h": 12, "dose": 10}, "redução de dose"),
    ],
)
def test_salto_de_dose_nao_dispara(segundo, motivo):
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Metoprolol Tartrate", "dose": 25},
            {"drug": "Metoprolol Tartrate", **segundo},
        )
    )
    assert alertas == [], motivo


def test_salto_de_dose_ignora_reposicao_de_eletrolitos():
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Magnesium Sulfate", "dose": 2, "unidade": "g"},
            {"h": 6, "drug": "Magnesium Sulfate", "dose": 4, "unidade": "g"},
        )
    )
    assert alertas == []


def test_salto_de_dose_ignora_dose_zero_ou_ausente():
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Insulin", "dose": 0},
            {"h": 2, "drug": "Insulin", "dose": 10},
            {"h": 4, "drug": "Insulin", "dose": None},
        )
    )
    assert alertas == []


def test_salto_de_dose_nao_cruza_internacoes():
    alertas = rules.salto_de_dose(
        prescricoes(
            {"h": 0, "drug": "Metoprolol Tartrate", "dose": 25, "hadm_id": 10},
            {"h": 2, "drug": "Metoprolol Tartrate", "dose": 50, "hadm_id": 11},
        )
    )
    assert alertas == []


# --- alta vigilância ---


def test_alta_vigilancia_alerta_so_a_primeira_droga_de_cada_classe():
    alertas = rules.alta_vigilancia(
        prescricoes(
            {"h": 0, "drug": "Acetaminophen", "dose": 500},
            {"h": 1, "drug": "Morphine Sulfate", "dose": 4},
            {"h": 5, "drug": "Fentanyl Citrate", "dose": 50},
            {"h": 8, "drug": "Insulin", "dose": 5},
        )
    )

    assert [(a.evidencia["classe"], a.evidencia["droga"]) for a in alertas] == [
        ("opioide", "Morphine Sulfate"),
        ("insulina", "Insulin"),
    ]
    assert [a.severidade for a in alertas] == [Severidade.ALTA, Severidade.MEDIA]
    assert alertas[0].detectado_em == T0 + timedelta(hours=1)


def test_alta_vigilancia_ignora_heparin_flush():
    alertas = rules.alta_vigilancia(
        prescricoes({"h": 0, "drug": "Heparin Flush (10 units/ml)", "dose": 1})
    )
    assert alertas == []


def test_alta_vigilancia_confirma_a_primeira_administracao_no_emar():
    emar = pd.DataFrame(
        {
            "hadm_id": [10, 10, 10, 99],
            "medication": ["Morphine Sulfate", "Morphine Sulfate", "Insulin", "Morphine Sulfate"],
            "event_txt": ["Not Given", "Administered", "Administered", "Administered"],
            "charttime": [T0 + timedelta(hours=h) for h in (1, 3, 4, 0)],
        }
    )

    [alerta] = rules.alta_vigilancia(
        prescricoes({"h": 1, "drug": "Morphine Sulfate", "dose": 4}), emar
    )

    assert alerta.evidencia["primeira_administracao"] == (T0 + timedelta(hours=3)).isoformat()


# --- trocas excessivas ---


def _trocas(n: int, espacamento_h: float = 6):
    """n trocas consecutivas de beta-bloqueador (n + 1 drogas distintas)."""
    drogas = ["Metoprolol", "Atenolol", "Carvedilol", "Labetalol", "Propranolol", "Metoprolol"]
    return prescricoes(
        *({"h": i * espacamento_h, "drug": drogas[i % len(drogas)]} for i in range(n + 1))
    )


def test_mais_de_quatro_trocas_em_48h_dispara_um_alerta_por_episodio():
    alertas = rules.trocas_excessivas(_trocas(5))

    assert len(alertas) == 1
    assert alertas[0].evidencia["n_trocas"] == 5
    assert alertas[0].severidade is Severidade.BAIXA


def test_quatro_trocas_em_48h_nao_dispara():
    assert rules.trocas_excessivas(_trocas(4)) == []


def test_trocas_espalhadas_alem_de_48h_nao_disparam():
    assert rules.trocas_excessivas(_trocas(6, espacamento_h=24)) == []


def test_reordenar_o_mesmo_principio_ativo_nao_e_troca():
    p = prescricoes(*({"h": i * 4, "drug": "Metoprolol Tartrate"} for i in range(10)))
    assert rules.trocas_excessivas(p) == []


# --- duplicidade ---


def test_duplicidade_de_duas_drogas_da_mesma_classe_ativas_juntas():
    [alerta] = rules.duplicidade(
        prescricoes(
            {"h": 0, "drug": "Morphine Sulfate", "duracao_h": 48},
            {"h": 10, "drug": "HYDROmorphone (Dilaudid)", "duracao_h": 24},
        )
    )

    assert alerta.evidencia["regra"] == "duplicidade"
    assert alerta.evidencia["classe"] == "opioide"
    assert alerta.evidencia["drogas"] == ["Morphine Sulfate", "HYDROmorphone (Dilaudid)"]
    assert alerta.severidade is Severidade.ALTA
    assert alerta.detectado_em == T0 + timedelta(hours=10)


def test_duplicidade_nao_dispara_quando_a_primeira_ja_terminou():
    alertas = rules.duplicidade(
        prescricoes(
            {"h": 0, "drug": "Famotidine", "duracao_h": 10},
            {"h": 12, "drug": "Ranitidine"},
        )
    )
    assert alertas == []


def test_duplicidade_nao_dispara_para_o_mesmo_principio_ativo():
    alertas = rules.duplicidade(
        prescricoes(
            {"h": 0, "drug": "Metoprolol Tartrate"},
            {"h": 2, "drug": "Metoprolol Succinate XL"},
        )
    )
    assert alertas == []


def test_duplicidade_nao_dispara_para_classes_diferentes():
    alertas = rules.duplicidade(
        prescricoes({"h": 0, "drug": "Furosemide"}, {"h": 1, "drug": "Atorvastatin"})
    )
    assert alertas == []


def test_heparina_com_varfarina_e_ponte_de_anticoagulacao_e_nao_duplicidade():
    alertas = rules.duplicidade(
        prescricoes({"h": 0, "drug": "Heparin", "duracao_h": 72}, {"h": 6, "drug": "Warfarin"})
    )
    assert alertas == []


def test_duplicidade_alerta_o_par_so_uma_vez():
    alertas = rules.duplicidade(
        prescricoes(
            {"h": 0, "drug": "Lorazepam", "duracao_h": 100},
            {"h": 1, "drug": "Midazolam", "duracao_h": 100},
            {"h": 2, "drug": "Midazolam", "duracao_h": 100},
        )
    )
    assert len(alertas) == 1


# --- analisar ---


def test_analisar_junta_as_regras_em_ordem_cronologica_e_conta_por_regra():
    alertas = rules.analisar(
        prescricoes(
            {"h": 0, "drug": "Morphine Sulfate", "dose": 2, "duracao_h": 48},
            {"h": 6, "drug": "Morphine Sulfate", "dose": 4, "duracao_h": 48},
            {"h": 8, "drug": "Fentanyl Citrate", "dose": 50},
        )
    )

    assert [a.detectado_em for a in alertas] == sorted(a.detectado_em for a in alertas)
    assert rules.contar_por_regra(alertas) == {
        "salto_de_dose": 1,
        "alta_vigilancia": 1,
        "trocas_excessivas": 0,
        "duplicidade": 1,
    }


def test_analisar_e_deterministico():
    p = prescricoes(
        {"h": 0, "drug": "Morphine Sulfate", "dose": 2},
        {"h": 6, "drug": "Morphine Sulfate", "dose": 4},
    )
    assert [a.id for a in rules.analisar(p)] == [a.id for a in rules.analisar(p)]


# --- dados reais (só se o MIMIC-IV Demo foi baixado) ---

requer_mimic = pytest.mark.skipif(
    not (load.PASTA_PADRAO / "hosp" / "prescriptions.csv.gz").exists(),
    reason="rode `uv run python scripts/download_data.py mimic4demo`",
)


@requer_mimic
def test_mimic4_demo_gera_alertas_para_as_quatro_regras():
    dados = load.carregar()

    contagem = rules.contar_por_regra(rules.analisar(dados.prescricoes, dados.emar))

    assert dados.prescricoes["subject_id"].nunique() == 100
    assert all(n > 0 for n in contagem.values()), contagem
