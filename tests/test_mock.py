from src.core import db
from src.core.alerts import Origem, Severidade, Status, Tipo
from src.core.mock import gerar_alertas_mock, popular_mock


def test_mock_cobre_todas_as_origens_tipos_severidades_e_status():
    alertas = gerar_alertas_mock()

    assert {a.origem for a in alertas} == set(Origem)
    assert {a.tipo for a in alertas} == set(Tipo)
    assert {a.severidade for a in alertas} == set(Severidade)
    assert {a.status for a in alertas} == set(Status)


def test_alertas_compostos_fundem_alertas_do_mesmo_paciente_de_origens_diferentes():
    alertas = gerar_alertas_mock()
    por_id = {a.id: a for a in alertas}
    compostos = [a for a in alertas if a.origem is Origem.FUSAO]

    assert any(c.severidade is Severidade.CRITICA for c in compostos)
    for composto in compostos:
        fundidos = [por_id[i] for i in composto.alertas_origem]
        assert {a.paciente_id for a in fundidos} == {composto.paciente_id}
        assert len({a.origem for a in fundidos}) >= 2
        assert all(a.detectado_em <= composto.detectado_em for a in fundidos)


def test_mock_e_deterministico():
    assert gerar_alertas_mock() == gerar_alertas_mock()


def test_popular_mock_grava_no_banco_sem_duplicar(tmp_path):
    with db.conectar(tmp_path / "alertas.db") as conn:
        popular_mock(conn)
        popular_mock(conn)

        assert sorted(a.id for a in db.listar(conn)) == sorted(a.id for a in gerar_alertas_mock())


def test_popular_mock_de_novo_preserva_reconhecimentos_feitos_na_central(tmp_path):
    with db.conectar(tmp_path / "alertas.db") as conn:
        popular_mock(conn)
        novo = db.listar(conn, status="novo")[0]
        db.reconhecer(conn, novo.id, por="enf. Ana")

        popular_mock(conn)

        assert db.obter(conn, novo.id).status is Status.RECONHECIDO
