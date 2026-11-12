import os
from datetime import datetime

import pytest
from streamlit.testing.v1 import AppTest

from src.core import db, mock
from src.core.alerts import Origem, Severidade, Status
from src.ui import central


@pytest.fixture
def conn(tmp_path):
    conexao = db.conectar(tmp_path / "alertas.db")
    mock.popular_mock(conexao)
    yield conexao
    conexao.close()


def test_tabela_segue_a_ordem_por_severidade_e_horario(conn):
    tabela = central.tabela_alertas(db.listar(conn))

    assert tabela["Severidade"].iloc[0].endswith("Crítica")
    assert list(tabela.columns) == [
        "Severidade",
        "Horário",
        "Paciente",
        "Origem",
        "Tipo",
        "Status",
        "Descrição",
    ]
    assert len(tabela) == len(mock.gerar_alertas_mock())


def test_central_abre_sem_erro_e_carrega_mock_quando_banco_vazio(tmp_path, monkeypatch):
    def roteiro():
        import os

        from src.core import db
        from src.ui import central

        central.pagina(db.conectar(os.environ["TESTE_DB"]))

    monkeypatch.setenv("TESTE_DB", str(tmp_path / "vazio.db"))
    app = AppTest.from_function(roteiro, default_timeout=60).run()

    assert not app.exception
    assert any("mock" in i.value for i in app.info)


def test_filtros_reduzem_a_lista(conn):
    so_prescricao = db.listar(conn, origem=[Origem.PRESCRICAO])
    assert so_prescricao
    assert {a.origem for a in so_prescricao} == {Origem.PRESCRICAO}


def test_reconhecer_e_resolver_gravam_no_banco(conn):
    novo = next(a for a in db.listar(conn) if a.status is Status.NOVO)

    db.reconhecer(conn, novo.id, por="enf. Ana", em=datetime(2026, 10, 1, 12, 0))
    reconhecido = db.listar(conn, status=Status.RECONHECIDO)
    gravado = next(a for a in reconhecido if a.id == novo.id)
    assert gravado.reconhecido_por == "enf. Ana"
    assert gravado.reconhecido_em == datetime(2026, 10, 1, 12, 0)

    db.resolver(conn, novo.id)
    assert novo.id in {a.id for a in db.listar(conn, status=Status.RESOLVIDO)}


def test_composto_critico_referencia_alertas_existentes(conn):
    composto = next(a for a in db.listar(conn) if a.severidade is Severidade.CRITICA)
    assert all(db.obter(conn, i) is not None for i in composto.alertas_origem)


def test_evidencia_em_linguagem_clinica():
    fatos = dict(
        central.fatos_da_evidencia(
            {
                "frame": "results/video/x.jpg",
                "regra": "valgo_dinamico",
                "lado": "dir",
                "altura_relativa": 0.48,
                "tecnica": "regras_clinicas",
                "valores": [[88.0]],
            }
        )
    )

    assert fatos == {
        "Regra": "valgo dinâmico do joelho",
        "Lado": "direito",
        "Altura vs. em pé": "48%",
        "Técnica": "regra clínica",
    }


def test_rotulos_humanos_na_tabela(conn):
    tabela = central.tabela_alertas(db.listar(conn, origem=[Origem.SINAIS_VITAIS]))

    assert set(tabela["Origem"]) == {"Sinais vitais"}
    assert "sinais_vitais" not in tabela.to_string()


def test_central_abre_alerta_selecionado_com_acoes(tmp_path):
    def roteiro():
        import os

        from src.core import db
        from src.ui import central

        central.pagina(db.conectar(os.environ["TESTE_DB"]))

    caminho = tmp_path / "alertas.db"
    conexao = db.conectar(caminho)
    mock.popular_mock(conexao)
    critico = next(a for a in db.listar(conexao) if a.severidade is Severidade.CRITICA)
    os.environ["TESTE_DB"] = str(caminho)

    app = AppTest.from_function(roteiro, default_timeout=30)
    app.session_state["alerta_aberto"] = critico.id
    app.session_state["central_escopo"] = "todos"
    app.run()

    assert not app.exception
    assert any("Reconhecer" in b.label for b in app.button)
    assert any("Críticos sem reconhecimento" in m.label for m in app.metric)
