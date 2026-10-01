from datetime import datetime
from typing import Any

import pytest

from src.core.alerts import (
    Alerta,
    Origem,
    Severidade,
    Status,
    Tipo,
    TransicaoInvalida,
    reconhecer,
    resolver,
)

T0 = datetime(2026, 10, 1, 8, 0)


def alerta_spo2(**kw):
    campos: dict[str, Any] = dict(
        paciente_id="demo-01",
        origem="sinais_vitais",
        tipo="sinal_vital",
        severidade="alta",
        descricao="SpO2 abaixo de 90% por 5 minutos",
        evidencia={"serie": "SpO2", "janela": [0, 300], "valores": [89, 88, 87]},
        detectado_em=T0,
    )
    campos.update(kw)
    return Alerta(**campos)


def test_novo_alerta_nasce_com_id_e_status_novo():
    alerta = alerta_spo2()

    assert alerta.status is Status.NOVO
    assert alerta.origem is Origem.SINAIS_VITAIS
    assert alerta.tipo is Tipo.SINAL_VITAL
    assert alerta.severidade is Severidade.ALTA
    assert len(alerta.id) == 36
    assert alerta.id != alerta_spo2().id
    assert alerta.alertas_origem == []
    assert alerta.reconhecido_por is None and alerta.reconhecido_em is None


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("origem", "raio_x"),
        ("tipo", "pressentimento"),
        ("severidade", "urgente"),
        ("status", "arquivado"),
    ],
)
def test_valores_fora_dos_enums_sao_rejeitados(campo, valor):
    with pytest.raises(ValueError):
        alerta_spo2(**{campo: valor})


def test_tipo_precisa_pertencer_a_origem():
    with pytest.raises(ValueError, match="queda"):
        alerta_spo2(origem="sinais_vitais", tipo="queda")


def test_alerta_composto_exige_alertas_de_origem():
    with pytest.raises(ValueError, match="alertas_origem"):
        alerta_spo2(origem="fusao", tipo="composto", alertas_origem=[])
    with pytest.raises(ValueError, match="alertas_origem"):
        alerta_spo2(origem="fusao", tipo="composto", alertas_origem=["a"])

    composto = alerta_spo2(origem="fusao", tipo="composto", alertas_origem=["a", "b"])
    assert composto.alertas_origem == ["a", "b"]


def test_so_alerta_composto_tem_alertas_de_origem():
    with pytest.raises(ValueError, match="alertas_origem"):
        alerta_spo2(alertas_origem=["a"])


def test_evidencia_precisa_ser_objeto_json():
    with pytest.raises(ValueError, match="evidencia"):
        alerta_spo2(evidencia=["não", "é", "objeto"])
    with pytest.raises(ValueError, match="evidencia"):
        alerta_spo2(evidencia={"quando": T0})


def test_reconhecimento_registra_quem_e_quando():
    T1 = datetime(2026, 10, 1, 8, 5)

    reconhecido = reconhecer(alerta_spo2(), por="enf. Ana", em=T1)

    assert reconhecido.status is Status.RECONHECIDO
    assert reconhecido.reconhecido_por == "enf. Ana"
    assert reconhecido.reconhecido_em == T1


def test_resolucao_vem_depois_do_reconhecimento():
    reconhecido = reconhecer(alerta_spo2(), por="enf. Ana", em=T0)

    resolvido = resolver(reconhecido)

    assert resolvido.status is Status.RESOLVIDO
    assert resolvido.reconhecido_por == "enf. Ana"


def test_transicoes_fora_de_ordem_sao_rejeitadas():
    novo = alerta_spo2()
    reconhecido = reconhecer(novo, por="enf. Ana", em=T0)

    with pytest.raises(TransicaoInvalida):
        resolver(novo)
    with pytest.raises(TransicaoInvalida):
        reconhecer(reconhecido, por="dr. Bruno", em=T0)
    with pytest.raises(TransicaoInvalida):
        resolver(resolver(reconhecido))


def test_alerta_vira_dict_json_e_volta_igual():
    import json

    composto = reconhecer(
        alerta_spo2(
            origem="fusao",
            tipo="composto",
            severidade="critica",
            alertas_origem=["a", "b"],
        ),
        por="dr. Bruno",
        em=datetime(2026, 10, 1, 9, 0),
    )

    como_dict = composto.para_dict()

    assert como_dict["origem"] == "fusao"
    assert como_dict["detectado_em"] == "2026-10-01T08:00:00"
    assert como_dict["reconhecido_em"] == "2026-10-01T09:00:00"
    assert Alerta.de_dict(json.loads(json.dumps(como_dict))) == composto
