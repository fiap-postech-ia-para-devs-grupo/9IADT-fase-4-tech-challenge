from datetime import datetime, timedelta

from src.core import inteligencia
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo

T0 = datetime(2026, 10, 1, 8, 0)


def sinal(minuto: int, serie: str = "SpO2", sev=Severidade.ALTA, paciente="demo-01", **kw):
    return Alerta(
        paciente_id=paciente,
        origem=Origem.SINAIS_VITAIS,
        tipo=Tipo.SINAL_VITAL,
        severidade=sev,
        descricao=f"{serie} fora da faixa",
        evidencia={"tecnica": "regras_clinicas", "serie": serie},
        detectado_em=T0 + timedelta(minutes=minuto),
        **kw,
    )


def test_repetidos_proximos_viram_um_episodio():
    episodios = inteligencia.agrupar([sinal(0), sinal(10), sinal(25)])

    assert len(episodios) == 1
    assert len(episodios[0].alertas) == 3
    assert episodios[0].fim - episodios[0].inicio == timedelta(minutes=25)


def test_intervalo_longo_abre_outro_episodio():
    assert len(inteligencia.agrupar([sinal(0), sinal(90)])) == 2


def test_sinais_ou_pacientes_diferentes_nao_se_misturam():
    episodios = inteligencia.agrupar([sinal(0), sinal(5, "FC"), sinal(5, paciente="demo-02")])

    assert len(episodios) == 3


def test_episodio_mostra_o_mais_grave_e_vem_primeiro_na_fila():
    episodios = inteligencia.agrupar(
        [sinal(0), sinal(5, sev=Severidade.CRITICA), sinal(200, "FC", sev=Severidade.MEDIA)]
    )

    assert episodios[0].principal.severidade is Severidade.CRITICA
    assert episodios[-1].principal.severidade is Severidade.MEDIA


def test_critico_sem_reconhecimento_escalona_depois_do_prazo():
    critico = sinal(0, sev=Severidade.CRITICA)

    assert not inteligencia.escalonado(critico, T0 + timedelta(minutes=4))
    assert inteligencia.escalonado(critico, T0 + timedelta(minutes=5))
    assert inteligencia.minutos_sem_reconhecimento(critico, T0 + timedelta(minutes=7)) == 7


def test_reconhecido_ou_alto_nao_escalona():
    reconhecido = sinal(0, sev=Severidade.CRITICA, status=Status.RECONHECIDO)
    alto = sinal(0)

    assert not inteligencia.escalonado(reconhecido, T0 + timedelta(minutes=30))
    assert not inteligencia.escalonado(alto, T0 + timedelta(minutes=30))
