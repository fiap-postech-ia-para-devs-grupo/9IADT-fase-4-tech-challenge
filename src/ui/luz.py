"""Estado da luz de chamada de um leito, a partir dos Alertas pendentes do paciente."""

from __future__ import annotations

from collections.abc import Iterable

from src.core import notificacao
from src.core.alerts import Alerta, Severidade, Status


def estado(pendentes: Iterable[Alerta]) -> str:
    """'critico' (chamado aberto: sirene vermelha + WhatsApp), 'alto' (atenção, âmbar) ou
    'apagada'. Só conta o que ainda não foi reconhecido."""
    novos = [a for a in pendentes if a.status is Status.NOVO]
    if any(notificacao.exige_chamada(a) for a in novos):
        return "critico"
    if any(a.severidade is Severidade.ALTA for a in novos):
        return "alto"
    return "apagada"
