"""Detectores de Anomalia nos Sinais Vitais (ESTRATEGIA.md, seção 6).

Três técnicas com a mesma interface `detect(janela, paciente_id) -> list[Alerta]`:

    regras_clinicas   limites fixos: SpO2 < 90, FC > 120 ou < 45, PAS > 180 ou < 90, FR > 25
    zscore_movel      |x - média| > 3σ da própria história recente (60 min anteriores)
    isolation_forest  anomalia multivariada sobre médias móveis de 5 min (FC, FR, SpO2, PAS)

`janela` é um DataFrame com índice `DatetimeIndex` (1 amostra por minuto, lacunas como NaN,
como sai de `load.py`) e colunas `FC`, `FR`, `SpO2`, `PAS`. Cada Anomalia vira um Alerta no
Contrato do Alerta (`origem=sinais_vitais`, `tipo=sinal_vital`). Uma Anomalia é um episódio:
amostras consecutivas fora do padrão viram um único Alerta, com `evidencia["janela"]` em
segundos desde o início da janela recebida. Os ids são determinísticos.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core.alerts import Alerta, Origem, Severidade, Tipo

SERIES = ("FC", "FR", "SpO2", "PAS")

# Regras clínicas: (série, operador, limite, severidade). Só os limites da seção 6.
REGRAS_CLINICAS: tuple[tuple[str, str, float, Severidade], ...] = (
    ("SpO2", "<", 90, Severidade.CRITICA),
    ("FC", ">", 120, Severidade.ALTA),
    ("FC", "<", 45, Severidade.ALTA),
    ("PAS", ">", 180, Severidade.ALTA),
    ("PAS", "<", 90, Severidade.ALTA),
    ("FR", ">", 25, Severidade.ALTA),
)

JANELA_ZSCORE = 60  # amostras (minutos) de história
MIN_HISTORIA_ZSCORE = 30  # abaixo disso o z-score não tem base estatística
LIMITE_ZSCORE = 3.0
# Desvio absoluto mínimo (unidade da série). Com história muito estável o σ fica pequeno e
# 3σ vira ruído fisiológico; exigir um desvio clinicamente perceptível evita esses falsos positivos.
DESVIO_MINIMO_ZSCORE: dict[str, float] = {"FC": 5, "FR": 3, "SpO2": 2, "PAS": 10}

JANELA_IF = 5  # minutos para a média móvel que alimenta o Isolation Forest
MIN_AMOSTRAS_IF = 50  # menos que isso e o modelo não aprende o padrão do paciente
CONTAMINACAO_IF = 0.01  # fração esperada de anomalias; o modelo sempre marca ~essa fração

_NAMESPACE = uuid.UUID("0b7e2f4a-9c1d-4e3b-8a6f-2d5c7e9f1a30")

Detector = Callable[[pd.DataFrame, str], list[Alerta]]


def _episodios(mascara: pd.Series) -> list[tuple[int, int]]:
    """Posições (início, fim) de cada trecho contínuo de True, inclusivas."""
    posicoes = np.flatnonzero(mascara.fillna(False).to_numpy())
    if posicoes.size == 0:
        return []
    grupos = np.split(posicoes, np.flatnonzero(np.diff(posicoes) != 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in grupos]


def _alerta(
    tecnica: str,
    paciente_id: str,
    janela: pd.DataFrame,
    inicio: int,
    fim: int,
    series: str,
    severidade: Severidade,
    descricao: str,
    **extra: object,
) -> Alerta:
    t0 = janela.index[inicio]
    t1 = janela.index[fim]
    ini_s = (t0 - janela.index[0]).total_seconds()
    fim_s = (t1 - janela.index[0]).total_seconds()
    valores = janela[series.split(",")].iloc[inicio : fim + 1]
    chave = f"{tecnica}|{series}|{t0.isoformat()}|{t1.isoformat()}"
    return Alerta(
        id=str(uuid.uuid5(_NAMESPACE, f"{paciente_id}|{chave}")),
        paciente_id=paciente_id,
        origem=Origem.SINAIS_VITAIS,
        tipo=Tipo.SINAL_VITAL,
        severidade=severidade,
        descricao=descricao,
        evidencia={
            "tecnica": tecnica,
            "serie": series,
            "janela": [ini_s, fim_s],
            "valores": valores.to_numpy(dtype=float).round(2).tolist(),
            **extra,
        },
        detectado_em=t1.to_pydatetime(),
    )


def regras_clinicas(janela: pd.DataFrame, paciente_id: str) -> list[Alerta]:
    alertas: list[Alerta] = []
    for serie, operador, limite, severidade in REGRAS_CLINICAS:
        valor = janela[serie]
        mascara = valor < limite if operador == "<" else valor > limite
        for inicio, fim in _episodios(mascara):
            descricao = f"{serie} {operador} {limite} por {fim - inicio + 1} min"
            alertas.append(
                _alerta(
                    "regras_clinicas",
                    paciente_id,
                    janela,
                    inicio,
                    fim,
                    serie,
                    severidade,
                    descricao,
                    limite=limite,
                )
            )
    return alertas


def zscore_movel(janela: pd.DataFrame, paciente_id: str) -> list[Alerta]:
    alertas: list[Alerta] = []
    for serie in SERIES:
        valor = janela[serie]
        # shift(1): o ponto avaliado não entra na própria média nem no próprio desvio
        historia = valor.shift(1).rolling(JANELA_ZSCORE, min_periods=MIN_HISTORIA_ZSCORE)
        media = historia.mean()
        desvio = historia.std().replace(0, np.nan)
        z = (valor - media) / desvio
        relevante = (valor - media).abs() >= DESVIO_MINIMO_ZSCORE[serie]
        for inicio, fim in _episodios((z.abs() > LIMITE_ZSCORE) & relevante):
            pico = float(z.iloc[inicio : fim + 1].abs().max())
            alertas.append(
                _alerta(
                    "zscore_movel",
                    paciente_id,
                    janela,
                    inicio,
                    fim,
                    serie,
                    Severidade.MEDIA,
                    f"{serie} fora de {LIMITE_ZSCORE:g}σ da história recente (pico {pico:.1f}σ)",
                    z_pico=round(pico, 2),
                )
            )
    return alertas


def isolation_forest(janela: pd.DataFrame, paciente_id: str) -> list[Alerta]:
    features = janela[list(SERIES)].rolling(JANELA_IF, min_periods=JANELA_IF).mean()
    features = features.dropna()
    if len(features) < MIN_AMOSTRAS_IF:
        return []

    modelo = IsolationForest(n_estimators=200, contamination=CONTAMINACAO_IF, random_state=0)
    rotulos = modelo.fit_predict(features.to_numpy())
    # features começa depois das primeiras JANELA_IF-1 amostras; realinha com `janela`
    posicoes = np.flatnonzero(janela.index.isin(features.index))
    mascara = pd.Series(rotulos == -1, index=posicoes)
    mascara = mascara.reindex(range(len(janela)), fill_value=False)

    alertas: list[Alerta] = []
    for inicio, fim in _episodios(mascara):
        alertas.append(
            _alerta(
                "isolation_forest",
                paciente_id,
                janela,
                inicio,
                fim,
                ",".join(SERIES),
                Severidade.BAIXA,
                "padrão conjunto de FC, FR, SpO2 e PAS fora do comum (Isolation Forest)",
            )
        )
    return alertas


TECNICAS: dict[str, Detector] = {
    "regras_clinicas": regras_clinicas,
    "zscore_movel": zscore_movel,
    "isolation_forest": isolation_forest,
}


def detect(janela: pd.DataFrame, paciente_id: str, tecnica: str) -> list[Alerta]:
    """Roda uma técnica pelo nome (`TECNICAS`) sobre a janela e devolve os Alertas."""
    if tecnica not in TECNICAS:
        raise ValueError(f"técnica '{tecnica}' desconhecida; use uma de {sorted(TECNICAS)}")
    if not isinstance(janela.index, pd.DatetimeIndex):
        raise ValueError("a janela precisa de índice DatetimeIndex (ver load.py)")
    return TECNICAS[tecnica](janela, paciente_id)
