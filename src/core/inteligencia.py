"""Triagem da fila de Alertas contra a fadiga de alarme.

- **Agrupamento:** Alertas repetidos do mesmo paciente, mesma regra e mesmo sinal, a menos de
  `INTERVALO_EPISODIO` um do outro, viram um **episódio**: um cartão com a contagem, e não dez.
- **Escalonamento:** Alerta que pede chamada (crítico ou Queda) sem Reconhecimento há
  `notificacao.MINUTOS_ESCALONAMENTO` minutos do plantão é marcado como escalonado (e a equipe
  médica é acionada no WhatsApp, em `notificacao`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.core import notificacao
from src.core.alerts import Alerta, Severidade, Status

INTERVALO_EPISODIO = timedelta(minutes=30)
_PESO = {s: i for i, s in enumerate(Severidade)}


@dataclass(frozen=True)
class Episodio:
    alertas: tuple[Alerta, ...]  # mais recente primeiro

    @property
    def principal(self) -> Alerta:
        """O mais grave e, entre iguais, o mais recente: é o que o cartão mostra."""
        return max(self.alertas, key=lambda a: (_PESO[a.severidade], a.detectado_em))

    @property
    def inicio(self) -> datetime:
        return min(a.detectado_em for a in self.alertas)

    @property
    def fim(self) -> datetime:
        return max(a.detectado_em for a in self.alertas)

    @property
    def pendentes(self) -> list[Alerta]:
        return [a for a in self.alertas if a.status is Status.NOVO]


def chave(alerta: Alerta) -> tuple:
    ev = alerta.evidencia
    regra = ev.get("tecnica") or ev.get("regra") or ""
    sinal = ev.get("serie") or ev.get("termo") or ev.get("droga") or ""
    return (alerta.paciente_id, alerta.origem, alerta.tipo, regra, sinal)


def agrupar(alertas: list[Alerta]) -> list[Episodio]:
    """Episódios na ordem da fila: mais grave primeiro, depois mais recente."""
    por_chave: dict[tuple, list[list[Alerta]]] = {}
    for alerta in sorted(alertas, key=lambda a: a.detectado_em):
        grupos = por_chave.setdefault(chave(alerta), [])
        if grupos and alerta.detectado_em - grupos[-1][-1].detectado_em <= INTERVALO_EPISODIO:
            grupos[-1].append(alerta)
        else:
            grupos.append([alerta])
    episodios = [
        Episodio(tuple(sorted(g, key=lambda a: a.detectado_em, reverse=True)))
        for grupos in por_chave.values()
        for g in grupos
    ]
    return sorted(episodios, key=lambda e: (_PESO[e.principal.severidade], e.fim), reverse=True)


def minutos_sem_reconhecimento(alerta: Alerta, agora: datetime) -> int | None:
    if alerta.status is not Status.NOVO or alerta.detectado_em > agora:
        return None
    return int((agora - alerta.detectado_em).total_seconds() // 60)


def escalonado(alerta: Alerta, agora: datetime) -> bool:
    minutos = minutos_sem_reconhecimento(alerta, agora)
    return (
        minutos is not None
        and notificacao.exige_chamada(alerta)
        and minutos >= notificacao.MINUTOS_ESCALONAMENTO
    )
