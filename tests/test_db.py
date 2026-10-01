from datetime import datetime, timedelta

import pytest

from src.core import db
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo, TransicaoInvalida

T0 = datetime(2026, 10, 1, 8, 0)


@pytest.fixture
def conn(tmp_path):
    conexao = db.conectar(tmp_path / "alertas.db")
    yield conexao
    conexao.close()


def alerta(
    paciente_id="demo-01",
    origem="sinais_vitais",
    tipo="sinal_vital",
    severidade="alta",
    minutos=0,
    **kw,
):
    return Alerta(
        paciente_id=paciente_id,
        origem=Origem(origem),
        tipo=Tipo(tipo),
        severidade=Severidade(severidade),
        descricao=f"{tipo} em {paciente_id}",
        evidencia={"serie": "SpO2", "janela": [0, 300], "valores": [89, 88]},
        detectado_em=T0 + timedelta(minutes=minutos),
        **kw,
    )


def test_alerta_gravado_e_lido_de_volta_igual(conn):
    composto = alerta(
        origem="fusao", tipo="composto", severidade="critica", alertas_origem=["x", "y"]
    )

    db.salvar(conn, composto)

    assert db.obter(conn, composto.id) == composto


def test_banco_persiste_entre_conexoes(tmp_path):
    caminho = tmp_path / "alertas.db"
    original = alerta()
    with db.conectar(caminho) as primeira:
        db.salvar(primeira, original)

    with db.conectar(caminho) as segunda:
        assert db.obter(segunda, original.id) == original


def test_obter_id_inexistente_devolve_none(conn):
    assert db.obter(conn, "nao-existe") is None


def test_listar_ordena_por_severidade_e_depois_mais_recente(conn):
    media_antiga = alerta(severidade="media", minutos=0)
    critica = alerta(severidade="critica", minutos=5)
    media_recente = alerta(severidade="media", minutos=10)
    baixa = alerta(severidade="baixa", minutos=20)
    alta = alerta(severidade="alta", minutos=1)
    for a in (media_antiga, critica, media_recente, baixa, alta):
        db.salvar(conn, a)

    assert [a.id for a in db.listar(conn)] == [
        critica.id,
        alta.id,
        media_recente.id,
        media_antiga.id,
        baixa.id,
    ]


def test_listar_filtra_por_origem_status_e_paciente(conn):
    video_demo1 = alerta(origem="video", tipo="queda", paciente_id="demo-01")
    audio_demo1 = alerta(origem="audio", tipo="termo_critico", paciente_id="demo-01")
    video_demo2 = alerta(
        origem="video",
        tipo="queda",
        paciente_id="demo-02",
        status="reconhecido",
        reconhecido_por="enf. Ana",
        reconhecido_em=T0,
    )
    for a in (video_demo1, audio_demo1, video_demo2):
        db.salvar(conn, a)

    def ids(**filtros):
        return {a.id for a in db.listar(conn, **filtros)}

    assert ids(origem="video") == {video_demo1.id, video_demo2.id}
    assert ids(paciente_id="demo-01") == {video_demo1.id, audio_demo1.id}
    assert ids(status=Status.NOVO) == {video_demo1.id, audio_demo1.id}
    assert ids(origem="video", status="reconhecido") == {video_demo2.id}
    assert ids(origem=["video", "audio"], paciente_id=["demo-01"]) == {
        video_demo1.id,
        audio_demo1.id,
    }
    assert ids(origem=[]) == set()


def test_reconhecer_e_resolver_atualizam_o_status_gravado(conn):
    original = alerta()
    db.salvar(conn, original)
    T1 = T0 + timedelta(minutes=3)

    reconhecido = db.reconhecer(conn, original.id, por="enf. Ana", em=T1)

    assert reconhecido == db.obter(conn, original.id)
    assert (reconhecido.status, reconhecido.reconhecido_por, reconhecido.reconhecido_em) == (
        Status.RECONHECIDO,
        "enf. Ana",
        T1,
    )
    assert db.listar(conn, status="novo") == []

    resolvido = db.resolver(conn, original.id)

    assert resolvido == db.obter(conn, original.id)
    assert resolvido.status is Status.RESOLVIDO
    assert resolvido.reconhecido_por == "enf. Ana"


def test_reconhecer_sem_horario_usa_o_momento_atual(conn):
    original = alerta()
    db.salvar(conn, original)
    antes = datetime.now()

    reconhecido = db.reconhecer(conn, original.id, por="enf. Ana")

    assert reconhecido.reconhecido_em is not None
    assert antes <= reconhecido.reconhecido_em <= datetime.now()


def test_transicao_invalida_nao_altera_o_banco(conn):
    original = alerta()
    db.salvar(conn, original)

    with pytest.raises(TransicaoInvalida):
        db.resolver(conn, original.id)
    assert db.obter(conn, original.id) == original


def test_atualizar_alerta_inexistente_falha(conn):
    with pytest.raises(KeyError):
        db.reconhecer(conn, "nao-existe", por="enf. Ana")
