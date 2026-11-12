from streamlit.testing.v1 import AppTest


def test_laboratorio_abre_sem_erro_sem_upload():
    def roteiro():
        from src.ui import laboratorio

        laboratorio.pagina()

    app = AppTest.from_function(roteiro, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Laboratório de Análise"


def test_laboratorio_tem_abas_de_video_e_audio():
    def roteiro():
        from src.ui import laboratorio

        laboratorio.pagina()

    app = AppTest.from_function(roteiro, default_timeout=30).run()

    assert [aba.label for aba in app.tabs] == ["🎥 Vídeo", "🎙️ Áudio"]
    assert "Quarto: monitoramento de Queda" in app.selectbox(key="cenario_video").options


def test_laboratorio_mostra_resultado_de_video(tmp_path, monkeypatch):
    from src.analyzers.video import lab
    from src.analyzers.video import pose as p
    from tests import video_sintetico as vs
    from tests.test_video_lab import _video
    from tests.test_video_rules import _unipodal

    frames = _unipodal([18])
    monkeypatch.setattr(p, "estimar", lambda *a, **kw: vs.de_keypoints(frames))
    resultado = lab.analyze_video(
        _video(tmp_path / "sessao.avi", len(frames)),
        exercicio="agachamento_unipodal",
        vista="frontal",
        saida=tmp_path,
    )

    def roteiro():
        from src.ui import laboratorio

        laboratorio.pagina()

    app = AppTest.from_function(roteiro, default_timeout=30)
    app.session_state["resultado_video"] = resultado
    app.run()

    assert not app.exception
    assert app.metric[1].value == "1"  # Desvios de Execução
    assert any("Gravar Alertas" in b.label for b in app.button)
