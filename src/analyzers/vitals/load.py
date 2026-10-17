"""Leitura de registros numerics do MIMIC-III Waveform Database (ESTRATEGIA.md, seção 6).

`carregar` lê um registro com `wfdb` e devolve um DataFrame com as colunas `FC`, `FR`, `SpO2`,
`PAS`, um minuto por linha (`DatetimeIndex`). Lacunas curtas (até `LACUNA_MAX_MIN` minutos) são
interpoladas; lacunas maiores ficam como NaN, que os detectores tratam como ausência de dado.

`reamostrar` e `de_sinais` são puras e não dependem de `wfdb`, então os testes as cobrem
sem baixar nada. O `wfdb` só é importado dentro de `carregar`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LACUNA_MAX_MIN = 5

# Nome do sinal no registro -> coluna canônica. Para PAS a ordem é preferência: a pressão
# arterial invasiva (ABP) é mais confiável que a não invasiva (NBP), quando ambas existem.
NOMES_SERIES: dict[str, tuple[str, ...]] = {
    "FC": ("HR",),
    "FR": ("RESP",),
    "SpO2": ("SpO2",),
    "PAS": ("ABP Sys", "ART Sys", "NBP Sys"),
}

ORIGEM_PADRAO = datetime(2150, 1, 1)  # início fictício quando o registro não traz data


def reamostrar(serie: pd.Series, limite_lacuna_min: int = LACUNA_MAX_MIN) -> pd.Series:
    """Média por minuto; lacunas de até `limite_lacuna_min` minutos são interpoladas."""
    por_minuto = serie.resample("1min").mean()
    preenchido = por_minuto.interpolate(limit_area="inside")
    falta = por_minuto.isna()
    grupo = (falta != falta.shift()).cumsum()
    tamanho_da_lacuna = falta.groupby(grupo).transform("sum")
    return preenchido.mask(falta & (tamanho_da_lacuna > limite_lacuna_min))


def de_sinais(brutos: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas com nomes do MIMIC-III (índice temporal) nas colunas canônicas.

    Colunas ausentes no registro viram NaN para toda a janela.
    """
    minutos = brutos.resample("1min").mean().index
    saida: dict[str, pd.Series] = {}
    for canonica, candidatos in NOMES_SERIES.items():
        origem = next((nome for nome in candidatos if nome in brutos.columns), None)
        if origem is None:
            saida[canonica] = pd.Series(float("nan"), index=minutos)
        else:
            saida[canonica] = reamostrar(brutos[origem].astype(float)).reindex(minutos)
    return pd.DataFrame(saida, index=minutos)


def carregar(caminho_base: str | Path) -> pd.DataFrame:
    """Lê um registro numerics (ex.: `data/vitals/mimic3wdb/p000/p00xxxx/3xxxxx_0001`)."""
    import wfdb  # import tardio: o restante do módulo funciona sem wfdb instalado

    registro = wfdb.rdrecord(str(caminho_base))
    inicio = getattr(registro, "base_datetime", None) or ORIGEM_PADRAO
    segundos = np.arange(registro.p_signal.shape[0]) / registro.fs
    instantes = pd.to_datetime(inicio) + pd.to_timedelta(segundos, unit="s")
    brutos = pd.DataFrame(registro.p_signal, columns=registro.sig_name, index=instantes)
    return de_sinais(brutos)
