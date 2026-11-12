from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from src.analyzers.video import lab
from src.analyzers.video import pose as p
from src.core import db
from src.core.alerts import Origem, Severidade, Tipo
from tests import video_sintetico as vs
from tests.test_video_fall import EM_PE, NO_CHAO
from tests.test_video_rules import _unipodal

INICIO = datetime(2026, 10, 1, 11, 30)


def _video(caminho, quadros: int):
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter.fourcc(*"MJPG"), vs.FPS, (320, 240))
    for _ in range(quadros):
        escritor.write(np.full((240, 320, 3), 128, np.uint8))
    escritor.release()
    return caminho


@pytest.fixture
def sem_yolo(monkeypatch):
    """`pose.estimar` devolve a sequência sintética da vez; o vídeo só fornece os frames."""
    estado = {}

    def estimar(caminho, *, passo, recorte, modelo):
        return estado["sequencia"]

    monkeypatch.setattr(p, "estimar", estimar)
    return estado


def test_sessao_com_valgo_gera_desvios_relatorio_e_alertas(tmp_path, sem_yolo):
    frames = _unipodal([18, 15])
    sem_yolo["sequencia"] = vs.de_keypoints(frames)
    arquivo = _video(tmp_path / "sessao.avi", len(frames))

    resultado = lab.analyze_video(
        arquivo,
        exercicio="agachamento_unipodal",
        vista="frontal",
        paciente_id="demo-01",
        clipe_id="sessao",
        inicio=INICIO,
        saida=tmp_path / "results",
    )

    assert len(resultado.desvios) == 2
    assert resultado.quedas == []
    assert "valgo" in resultado.serie
    pasta = tmp_path / "results" / "sessao"
    assert (pasta / "relatorio.md").read_text(encoding="utf-8").count("valgo dinâmico") >= 2
    assert (pasta / "angulos.png").is_file()
    assert sorted(f.name for f in pasta.glob("desvio_*.jpg")) == ["desvio_01.jpg", "desvio_02.jpg"]

    alerta = resultado.alertas[0]
    assert (alerta.origem, alerta.tipo, alerta.severidade) == (
        Origem.VIDEO,
        Tipo.DESVIO_EXECUCAO,
        Severidade.MEDIA,
    )
    assert alerta.paciente_id == "demo-01"
    assert alerta.detectado_em == INICIO + timedelta(seconds=resultado.desvios[0].t)
    assert alerta.evidencia["regra"] == "valgo_dinamico"
    assert (tmp_path / "results" / "sessao" / "desvio_01.jpg").samefile(alerta.evidencia["frame"])


def test_video_de_quarto_gera_alerta_de_queda(tmp_path, sem_yolo):
    sem_yolo["sequencia"] = vs.de_caixas([EM_PE] * 10 + [NO_CHAO] * 12)
    arquivo = _video(tmp_path / "quarto.avi", 22)

    resultado = lab.analyze_video(arquivo, inicio=INICIO, saida=tmp_path / "results")

    assert resultado.exercicio is None
    assert [a.tipo for a in resultado.alertas] == [Tipo.QUEDA]
    queda = resultado.alertas[0]
    assert queda.severidade is Severidade.ALTA
    assert queda.paciente_id == "avulso"
    assert queda.evidencia["altura_relativa"] == pytest.approx(70 / 170, abs=0.01)
    assert "Nenhum Desvio" not in resultado.relatorio.read_text(encoding="utf-8")


def test_reprocessar_nao_duplica_alertas(tmp_path, sem_yolo):
    sem_yolo["sequencia"] = vs.de_caixas([EM_PE] * 10 + [NO_CHAO] * 12)
    arquivo = _video(tmp_path / "quarto.avi", 22)
    conn = db.conectar(tmp_path / "alertas.db")

    for _ in range(2):
        resultado = lab.analyze_video(arquivo, inicio=INICIO, saida=tmp_path / "results")
        for alerta in resultado.alertas:
            db.salvar(conn, alerta)

    assert len(db.listar(conn)) == 1


def test_uploads_de_mesmo_nome_e_conteudo_diferente_tem_ids_diferentes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    um = _video(tmp_path / "a" / "video.avi", 3)
    outro = _video(tmp_path / "b" / "video.avi", 5)

    assert lab.identificador(um) != lab.identificador(outro)
    assert lab.identificador(um).startswith("video-")


def test_exercicio_desconhecido(tmp_path):
    with pytest.raises(ValueError, match="exercício desconhecido"):
        lab.analyze_video(tmp_path / "x.mp4", exercicio="polichinelo")
