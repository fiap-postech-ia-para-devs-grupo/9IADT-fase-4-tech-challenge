"""Monitoramento Contínuo por replay (ESTRATEGIA.md, seção 6).

`replay` alimenta os detectores minuto a minuto, como se os sinais chegassem ao vivo: a cada
minuto roda as técnicas sobre a história recente (`janela_analise` minutos) e emite cada
Anomalia no instante em que ela aparece pela primeira vez. Anomalias já emitidas não se
repetem, mesmo quando o episódio cresce nos minutos seguintes.

O ritmo é controlado por `minutos_por_segundo`: 1.0 significa 1 minuto de dado por segundo de
parede (o exemplo da especificação). `esperar` é injetável para que testes não durmam.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence

import pandas as pd

from src.analyzers.vitals import detectors
from src.core.alerts import Alerta

TECNICAS_PADRAO: tuple[str, ...] = ("regras_clinicas", "zscore_movel")


def replay(
    janela: pd.DataFrame,
    paciente_id: str,
    tecnicas: Sequence[str] = TECNICAS_PADRAO,
    minutos_por_segundo: float = 1.0,
    janela_analise: int = 180,
    esperar: Callable[[float], None] = time.sleep,
    ao_detectar: Callable[[Alerta], None] | None = None,
) -> Iterator[Alerta]:
    vistos: set[tuple[str, str, str]] = set()

    for t in range(len(janela)):
        if minutos_por_segundo > 0:
            esperar(1 / minutos_por_segundo)

        trecho = janela.iloc[max(0, t - janela_analise + 1) : t + 1]
        for tecnica in tecnicas:
            for alerta in detectors.detect(trecho, paciente_id, tecnica):
                inicio_abs = trecho.index[0] + pd.Timedelta(seconds=alerta.evidencia["janela"][0])
                chave = (tecnica, alerta.evidencia["serie"], inicio_abs.isoformat())
                if chave in vistos:
                    continue
                vistos.add(chave)
                if ao_detectar is not None:
                    ao_detectar(alerta)
                yield alerta
