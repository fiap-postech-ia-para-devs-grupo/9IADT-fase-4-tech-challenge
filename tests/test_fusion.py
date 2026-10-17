from datetime import datetime, timedelta

from src.core import fusion
from src.core.alerts import Alerta, Origem, Severidade, Tipo

T0 = datetime(2150, 1, 1, 8, 0)
PACIENTE = "mimic3-100"


def sinal(serie: str, quando: datetime = T0, paciente: str = PACIENTE) -> Alerta:
    return Alerta(
        paciente_id=paciente,
        origem=Origem.SINAIS_VITAIS,
        tipo=Tipo.SINAL_VITAL,
        severidade=Severidade.ALTA,
        descricao="sinal",
        evidencia={"serie": serie, "janela": [0.0, 60.0], "valores": [1.0]},
        detectado_em=quando,
    )


def vocal(quando: datetime = T0, tipo: Tipo = Tipo.ALTERACAO_VOCAL, **extra) -> Alerta:
    return Alerta(
        paciente_id=PACIENTE,
        origem=Origem.AUDIO,
        tipo=tipo,
        severidade=Severidade.MEDIA,
        descricao="voz",
        evidencia={"trecho": "...", "inicio": 0.0, **extra},
        detectado_em=quando,
    )


def queda(quando: datetime = T0) -> Alerta:
    return Alerta(
        paciente_id=PACIENTE,
        origem=Origem.VIDEO,
        tipo=Tipo.QUEDA,
        severidade=Severidade.CRITICA,
        descricao="queda",
        evidencia={"frame": "f.png", "t": 1.0},
        detectado_em=quando,
    )


def desvio(quando: datetime = T0) -> Alerta:
    return Alerta(
        paciente_id=PACIENTE,
        origem=Origem.VIDEO,
        tipo=Tipo.DESVIO_EXECUCAO,
        severidade=Severidade.MEDIA,
        descricao="desvio",
        evidencia={"frame": "f.png", "t": 1.0},
        detectado_em=quando,
    )


def prescricao(droga: str, regra: str, quando: datetime = T0) -> Alerta:
    return Alerta(
        paciente_id=PACIENTE,
        origem=Origem.PRESCRICAO,
        tipo=Tipo.MUDANCA_TERAPEUTICA,
        severidade=Severidade.ALTA,
        descricao="prescricao",
        evidencia={"droga": droga, "regra": regra},
        detectado_em=quando,
    )


def regras(compostos):
    return [c.evidencia["regra"] for c in compostos]


def test_sinal_respiratorio_com_alteracao_vocal_vira_insuficiencia_critica():
    origens = [sinal("SpO2"), vocal()]

    compostos = fusion.fundir(origens)

    assert regras(compostos) == ["insuficiencia_respiratoria"]
    assert compostos[0].severidade is Severidade.CRITICA
    assert compostos[0].origem is Origem.FUSAO
    assert compostos[0].tipo is Tipo.COMPOSTO
    assert sorted(compostos[0].alertas_origem) == sorted(a.id for a in origens)


def test_termo_critico_so_conta_se_for_respiratorio():
    generico = vocal(tipo=Tipo.TERMO_CRITICO, categoria="cardio")
    respiratorio = vocal(tipo=Tipo.TERMO_CRITICO, categoria="respiratorio")

    assert fusion.fundir([sinal("FR"), generico]) == []
    assert regras(fusion.fundir([sinal("FR"), respiratorio])) == ["insuficiencia_respiratoria"]


def test_origens_fora_da_janela_de_24h_nao_fundem():
    tarde = T0 + timedelta(hours=25)

    assert fusion.fundir([sinal("SpO2"), vocal(quando=tarde)]) == []


def test_pacientes_diferentes_nao_fundem():
    assert fusion.fundir([sinal("SpO2", paciente="outro"), vocal()]) == []


def test_queda_com_sedativo_vira_alerta_alto():
    compostos = fusion.fundir([queda(), prescricao("Midazolam (Versed)", "alta_vigilancia")])

    assert regras(compostos) == ["queda_com_sedativo"]
    assert compostos[0].severidade is Severidade.ALTA


def test_queda_com_droga_nao_vigilante_nao_funde():
    assert fusion.fundir([queda(), prescricao("Acetaminophen", "alta_vigilancia")]) == []


def test_pressao_com_anti_hipertensivo_vira_efeito_adverso():
    compostos = fusion.fundir([sinal("PAS"), prescricao("Enalapril Maleate", "trocas_excessivas")])

    assert regras(compostos) == ["efeito_adverso_cardiovascular"]


def test_desvio_recorrente_com_dor_vira_agravamento():
    origens = [
        desvio(),
        desvio(T0 + timedelta(hours=1)),
        vocal(tipo=Tipo.TERMO_CRITICO, categoria="dor"),
    ]

    compostos = fusion.fundir(origens)

    assert regras(compostos) == ["agravamento_em_reabilitacao"]
    assert compostos[0].severidade is Severidade.MEDIA


def test_um_desvio_so_nao_basta_para_recorrencia():
    assert fusion.fundir([desvio(), vocal(tipo=Tipo.TERMO_CRITICO, categoria="dor")]) == []


def test_ids_sao_deterministicos():
    origens = [sinal("SpO2"), vocal()]

    assert [c.id for c in fusion.fundir(origens)] == [c.id for c in fusion.fundir(origens)]
