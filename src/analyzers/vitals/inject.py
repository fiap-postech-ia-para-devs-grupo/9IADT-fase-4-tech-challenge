"""Injeção de Anomalias conhecidas em trechos limpos de Sinais Vitais (ESTRATEGIA.md, seção 6).

Como não há rótulo de anomalia no MIMIC-III, a avaliação injeta eventos com início e duração
conhecidos num registro. Os três tipos de evento são:

    pico_fc            FC sobe 50 bpm de forma abrupta e volta ao normal ao fim do trecho
    queda_spo2         SpO2 cai gradualmente até 86% ao longo do trecho
    hipotensao_subita  PAS cai de forma abrupta para 75 mmHg durante o trecho

`planejar` escolhe os locais de injeção fora de trechos proibidos (ex.: onde já há ruído,
lacunas ou Anomalias naturais), com espaçamento mínimo. `aplicar` devolve uma cópia da janela
com os eventos injetados e a lista de `Injecao`, que é o gabarito para `evaluate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Injecao:
    tipo: str
    serie: str
    inicio: datetime
    fim: datetime


def _pico_fc(valor: np.ndarray) -> np.ndarray:
    return valor + 50


def _queda_spo2(valor: np.ndarray) -> np.ndarray:
    return np.linspace(valor[0], 86.0, len(valor))


def _hipotensao_subita(valor: np.ndarray) -> np.ndarray:
    return np.full(len(valor), 75.0)


# tipo -> (série afetada, transformação do trecho)
TIPOS: dict[str, tuple[str, object]] = {
    "pico_fc": ("FC", _pico_fc),
    "queda_spo2": ("SpO2", _queda_spo2),
    "hipotensao_subita": ("PAS", _hipotensao_subita),
}

ESPACAMENTO_MIN = 30  # minutos entre dois eventos injetados (e entre eventos e trechos proibidos)


def planejar(
    janela: pd.DataFrame,
    tipos: list[str],
    por_tipo: int,
    duracao_min: int,
    excluir: pd.Series | None = None,
    seed: int = 0,
) -> list[tuple[str, int]]:
    """Posições de início (em minutos desde o começo da janela) para cada tipo de evento.

    `excluir` é uma máscara booleana por minuto: True marca trechos que não podem ser usados.
    Trechos com NaN na série afetada também são descartados.
    """
    rng = np.random.default_rng(seed)
    proibido = np.zeros(len(janela), dtype=bool) if excluir is None else excluir.to_numpy(bool)
    escolhidos: list[tuple[str, int]] = []
    ocupado = proibido.copy()

    for tipo in tipos:
        serie = TIPOS[tipo][0]
        sem_lacuna = janela[serie].notna().to_numpy()
        tentativas = 0
        feitos = 0
        while feitos < por_tipo and tentativas < 10_000:
            tentativas += 1
            inicio = int(rng.integers(0, len(janela) - duracao_min))
            trecho = slice(inicio, inicio + duracao_min)
            if ocupado[trecho].any() or not sem_lacuna[trecho].all():
                continue
            escolhidos.append((tipo, inicio))
            margem = slice(max(0, inicio - ESPACAMENTO_MIN), inicio + duracao_min + ESPACAMENTO_MIN)
            ocupado[margem] = True
            feitos += 1
    return escolhidos


def aplicar(
    janela: pd.DataFrame, planos: list[tuple[str, int]], duracao_min: int
) -> tuple[pd.DataFrame, list[Injecao]]:
    """Cópia da janela com cada evento aplicado e o gabarito correspondente."""
    injetada = janela.copy()
    gabarito: list[Injecao] = []
    for tipo, inicio in planos:
        serie, transformar = TIPOS[tipo]
        trecho = slice(inicio, inicio + duracao_min)
        valor = injetada[serie].iloc[trecho].to_numpy(dtype=float)
        injetada.iloc[trecho, injetada.columns.get_loc(serie)] = transformar(valor)  # type: ignore[operator]
        gabarito.append(
            Injecao(
                tipo=tipo,
                serie=serie,
                inicio=janela.index[inicio].to_pydatetime(),
                fim=janela.index[inicio + duracao_min - 1].to_pydatetime(),
            )
        )
    return injetada, gabarito
