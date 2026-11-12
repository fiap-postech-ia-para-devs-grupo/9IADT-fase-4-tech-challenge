# pyright: reportAttributeAccessIssue=false

from datetime import datetime, timedelta

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.analyzers.audio.semantic import EntidadeClinica, Trecho
from src.analyzers.prescriptions import load
from src.core import db, demo, mock
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo
from src.ui import painel

T0 = datetime(2026, 10, 1, 8, 0)


def paciente(**kw) -> demo.PacienteDemo:
    return demo.PacienteDemo(
        id=kw.get("id", "demo-x"),
        nome="Paciente X",
        resumo="",
        internacao_inicio=T0,
        registro="r",
        prescricoes=demo.FontePrescricoes(1, 10, datetime(2150, 1, 1, 12, 0), 60),
        consulta=demo.FonteConsulta("c1", 100),
        video=demo.FonteVideo(None, 150),
    )


def janela_com_spo2_baixa(minutos: int = 200, queda_em: int = 120) -> pd.DataFrame:
    indice = pd.date_range(T0, periods=minutos, freq="1min")
    janela = pd.DataFrame({"FC": 80.0, "FR": 16.0, "SpO2": 97.0, "PAS": 120.0}, index=indice)
    janela.iloc[queda_em, janela.columns.get_loc("SpO2")] = 85.0
    return janela


def prescricoes_com_opioide() -> pd.DataFrame:
    inicio = T0 + timedelta(minutes=60)
    return pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [10],
            "starttime": [inicio],
            "stoptime": [inicio + timedelta(hours=24)],
            "drug": ["Morphine Sulfate"],
            "dose": [4.0],
            "dose_unit_rx": ["mg"],
            "route": ["IV"],
        }
    )


EMAR_VAZIO = pd.DataFrame(columns=["hadm_id", "medication", "event_txt", "charttime"])


def internacao(p=None, janela=None, com_falta_de_ar=True) -> demo.Internacao:
    texto = "I feel short of breath" if com_falta_de_ar else "I am fine"
    entidades = (
        [[EntidadeClinica("short of breath", "SymptomOrSign", 0.9)]] if com_falta_de_ar else [[]]
    )
    return demo.montar_de_dados(
        p or paciente(),
        janela_com_spo2_baixa() if janela is None else janela,
        prescricoes_com_opioide(),
        EMAR_VAZIO,
        [Trecho(1200.0, 1205.0, texto)],
        entidades,
    )


def queda(minuto: float) -> Alerta:
    return Alerta(
        paciente_id="demo-x",
        origem=Origem.VIDEO,
        tipo=Tipo.QUEDA,
        severidade=Severidade.ALTA,
        descricao="Queda",
        evidencia={"t": 1.0},
        detectado_em=T0 + timedelta(minutes=minuto),
    )


def compostos(eventos):
    return [e.alerta for e in eventos if e.alerta.origem is Origem.FUSAO]


def test_json_liga_as_quatro_fontes_dos_tres_pacientes():
    pacientes = demo.carregar_pacientes()

    assert [p.id for p in pacientes] == ["demo-01", "demo-02", "demo-03"]
    for p in pacientes:
        assert p.registro and p.prescricoes.hadm_id and p.consulta.id
        assert p.video.inicio_min > 0


def test_paciente_do_composto_critico_usa_consulta_indicada_na_issue():
    assert demo.carregar_pacientes()[0].consulta.id == "day3_consultation04"


def test_prescricoes_sao_deslocadas_para_a_ancora_da_internacao():
    p = paciente()
    prescricoes = prescricoes_com_opioide().assign(
        starttime=[datetime(2150, 1, 1, 12, 0)], stoptime=[datetime(2150, 1, 2, 12, 0)]
    )
    emar = pd.DataFrame(
        {
            "hadm_id": [10, 99],
            "medication": ["Morphine", "Morphine"],
            "event_txt": ["Administered", "Administered"],
            "charttime": [datetime(2150, 1, 1, 13, 0), datetime(2150, 1, 1, 13, 0)],
        }
    )

    deslocadas, administracoes = demo._deslocar_prescricoes(p, prescricoes, emar)

    assert deslocadas["starttime"].iloc[0] == T0 + timedelta(minutes=60)
    assert len(administracoes) == 1
    assert administracoes["charttime"].iloc[0] == T0 + timedelta(minutes=120)


def test_alertas_saem_com_o_id_do_paciente_de_demonstracao():
    i = internacao()

    assert {e.alerta.paciente_id for e in i.eventos} == {"demo-x"}
    assert {e.alerta.origem for e in i.eventos} >= {
        Origem.SINAIS_VITAIS,
        Origem.PRESCRICAO,
        Origem.AUDIO,
    }


def test_prescricao_fora_da_internacao_nao_gera_alerta():
    p = paciente()
    tarde = prescricoes_com_opioide().assign(starttime=[T0 + timedelta(days=3)])

    assert demo.eventos_de_prescricoes(p, tarde, EMAR_VAZIO, T0, T0 + timedelta(hours=4)) == []


def test_sinal_vital_aparece_no_minuto_em_que_o_replay_o_emite():
    eventos = demo.eventos_de_sinais(janela_com_spo2_baixa(queda_em=120), "demo-x")

    spo2 = [e for e in eventos if e.alerta.evidencia["tecnica"] == "regras_clinicas"]
    assert [e.aparece_em for e in spo2] == [T0 + timedelta(minutes=120)]


def test_episodio_que_sai_da_historia_do_replay_nao_vira_alerta_novo():
    janela = janela_com_spo2_baixa(minutos=260, queda_em=5)

    eventos = demo.eventos_de_sinais(janela, "demo-x")

    regras = [e for e in eventos if e.alerta.evidencia["tecnica"] == "regras_clinicas"]
    assert len(regras) == 1


def test_sinal_respiratorio_com_termo_critico_gera_composto_critico():
    i = internacao()

    [composto] = compostos(demo.fundir_eventos(i.eventos))

    assert composto.severidade is Severidade.CRITICA
    assert composto.descricao == "Possível insuficiência respiratória"


def test_sem_termo_critico_respiratorio_nao_ha_composto():
    assert compostos(demo.fundir_eventos(internacao(com_falta_de_ar=False).eventos)) == []


def test_composto_aparece_quando_a_ultima_origem_aparece():
    eventos = demo.fundir_eventos(internacao().eventos)

    evento = next(e for e in eventos if e.alerta.origem is Origem.FUSAO)
    por_id = {e.alerta.id: e for e in eventos}
    assert evento.aparece_em == max(por_id[i].aparece_em for i in evento.alerta.alertas_origem)


def test_fusao_mantem_um_composto_por_regra_com_o_par_mais_proximo():
    janela = janela_com_spo2_baixa()
    janela.iloc[30, janela.columns.get_loc("SpO2")] = 85.0
    i = internacao(janela=janela)

    [composto] = compostos(demo.fundir_eventos(i.eventos))

    por_id = {e.alerta.id: e.alerta for e in i.eventos}
    sinal = next(
        por_id[o] for o in composto.alertas_origem if por_id[o].origem is Origem.SINAIS_VITAIS
    )
    assert sinal.detectado_em == T0 + timedelta(minutes=120)


def test_queda_do_analisador_de_video_entra_na_fusao():
    eventos = demo.fundir_eventos(internacao(com_falta_de_ar=False).eventos, [queda(150)])

    [composto] = compostos(eventos)
    assert composto.descricao == "Queda possivelmente associada a medicação"


def test_visiveis_respeita_o_relogio_do_replay():
    eventos = demo.fundir_eventos(internacao().eventos)

    assert compostos(demo.visiveis(eventos, T0 + timedelta(minutes=119))) == []
    assert compostos(demo.visiveis(eventos, T0 + timedelta(minutes=125)))


def test_gravar_tira_o_mock_e_preserva_reconhecimento(tmp_path):
    with db.conectar(tmp_path / "a.db") as conn:
        mock.popular_mock(conn)
        alertas = [e.alerta for e in internacao().eventos]

        assert demo.gravar(conn, alertas) == len(alertas)
        assert {a.id for a in db.listar(conn)} == {a.id for a in alertas}

        db.reconhecer(conn, alertas[0].id, por="enf. Ana")
        assert demo.gravar(conn, alertas) == 0
        gravado = db.obter(conn, alertas[0].id)
        assert gravado is not None
        assert gravado.status is Status.RECONHECIDO


def test_figuras_mostram_so_o_que_ja_surgiu():
    i = internacao()
    eventos = demo.fundir_eventos(i.eventos)
    agora = T0 + timedelta(minutes=70)
    surgidos = demo.visiveis(eventos, agora)

    sinais = painel.figura_sinais(i, surgidos, agora)
    linha = painel.figura_linha_do_tempo(i, surgidos, agora)

    assert len(sinais.data[0].x) == 71
    nomes = {t.name for t in linha.data if t.type == "scatter"}
    assert nomes == {"prescricao"}
    assert {"Consulta", "Vídeo", "Sinais Vitais", "Morphine Sulfate"} <= {
        t.y[0] for t in linha.data if t.type == "bar"
    }


def test_painel_abre_sem_erro(tmp_path, monkeypatch):
    def roteiro():
        from src.ui import painel

        painel.pagina()

    monkeypatch.setattr(demo, "montar", lambda p: internacao(p))
    app = AppTest.from_function(roteiro, default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "Painel do Paciente"
    assert len(app.selectbox[0].options) == 3


requer_dados = pytest.mark.skipif(
    not (demo.PASTA_VITALS / "30" / "3002751" / "3002751n.hea").exists()
    or not (load.PASTA_PADRAO / "hosp" / "prescriptions.csv.gz").exists(),
    reason="rode `uv run python scripts/download_data.py mimic3wdb mimic4demo`",
)


@requer_dados
def test_demo_01_dispara_composto_critico_com_falta_de_ar_na_queda_de_spo2():
    p = demo.carregar_pacientes()[0]
    i = demo.montar(p)

    eventos = demo.fundir_eventos(i.eventos)

    criticos = [c for c in compostos(eventos) if c.severidade is Severidade.CRITICA]
    assert len(criticos) == 1
    por_id = {e.alerta.id: e.alerta for e in eventos}
    origens = [por_id[o] for o in criticos[0].alertas_origem]
    sinal = next(a for a in origens if a.origem is Origem.SINAIS_VITAIS)
    termo = next(a for a in origens if a.origem is Origem.AUDIO)
    assert sinal.evidencia["serie"] == "SpO2"
    assert termo.evidencia["termo"] == "short of breath"
    assert abs(sinal.detectado_em - termo.detectado_em) < timedelta(minutes=1)


@requer_dados
def test_demo_02_dispara_efeito_adverso_e_demo_03_espera_o_video():
    _, segundo, terceiro = demo.carregar_pacientes()

    assert [c.descricao for c in compostos(demo.fundir_eventos(demo.montar(segundo).eventos))] == [
        "Possível efeito adverso de medicação"
    ]
    assert compostos(demo.fundir_eventos(demo.montar(terceiro).eventos)) == []


def test_monitor_usa_o_ultimo_valor_medido_ate_agora():
    from src.ui import painel

    janela = janela_com_spo2_baixa()
    janela.iloc[121, janela.columns.get_loc("FC")] = float("nan")
    i = internacao(janela=janela)

    no_minuto_121 = painel.ultimos_sinais(i, T0 + timedelta(minutes=121))

    assert no_minuto_121["FC"] == 80.0  # lacuna preenchida com o último valor
    assert painel.ultimos_sinais(i, T0 + timedelta(minutes=120))["SpO2"] == 85.0


def test_painel_mostra_monitor_com_news2(monkeypatch):
    def roteiro():
        from src.ui import painel

        painel.pagina()

    monkeypatch.setattr(demo, "montar", lambda p: internacao(p))
    app = AppTest.from_function(roteiro, default_timeout=60).run()

    rotulos = [m.label for m in app.metric]
    assert any("NEWS2" in r for r in rotulos)
    assert any("Saturação" in r for r in rotulos)
