from streamlit.testing.v1 import AppTest


def test_laboratorio_abre_sem_erro_sem_upload():
    def roteiro():
        from src.ui import laboratorio

        laboratorio.pagina()

    app = AppTest.from_function(roteiro, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Laboratório de Análise"
