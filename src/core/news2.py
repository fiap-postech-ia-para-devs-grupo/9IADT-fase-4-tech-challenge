"""NEWS2 parcial: escore de alerta precoce a partir dos Sinais Vitais monitorados.

O NEWS2 (Royal College of Physicians, 2017) soma pontos de 0 a 3 por parâmetro. Aqui entram os
quatro que o MIMIC-III Waveform traz: frequência respiratória, SpO2 (escala 1), pressão
sistólica e frequência cardíaca. Temperatura, nível de consciência e uso de O2 suplementar não
existem nos registros e contam 0, então o escore é um **piso** do NEWS2 real.

É apoio à decisão exibido no Painel do Paciente, não um Analisador: não gera Alerta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import pandas as pd

# (limite superior inclusivo, pontos), em ordem crescente; o último cobre o resto.
_FAIXAS: dict[str, tuple[tuple[float, int], ...]] = {
    "FR": ((8, 3), (11, 1), (20, 0), (24, 2), (math.inf, 3)),
    "SpO2": ((91, 3), (93, 2), (95, 1), (math.inf, 0)),
    "PAS": ((90, 3), (100, 2), (110, 1), (219, 0), (math.inf, 3)),
    "FC": ((40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (math.inf, 3)),
}
PARAMETROS = tuple(_FAIXAS)
AUSENTES = ("temperatura", "nível de consciência", "O2 suplementar")


@dataclass(frozen=True)
class Escore:
    total: int
    pontos: dict[str, int]  # por parâmetro; parâmetro sem medida fica de fora
    risco: str  # "baixo", "baixo-médio", "médio", "alto"
    resposta: str  # conduta recomendada pelo NEWS2 para o risco

    @property
    def tem_parametro_vermelho(self) -> bool:
        return any(p == 3 for p in self.pontos.values())


_RESPOSTA = {
    "baixo": "monitorização de rotina (a cada 12 h)",
    "baixo-médio": "avaliação urgente pela enfermagem; considerar médico (a cada 1 h)",
    "médio": "avaliação médica urgente (a cada 1 h)",
    "alto": "avaliação de emergência pela equipe de resposta rápida (contínua)",
}


def pontos(parametro: str, valor: float) -> int | None:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    for limite, ponto in _FAIXAS[parametro]:
        if valor <= limite:
            return ponto
    return None  # pragma: no cover - o último limite é infinito


def calcular(sinais: dict[str, float]) -> Escore:
    por_parametro = {
        nome: p
        for nome in PARAMETROS
        if nome in sinais and (p := pontos(nome, sinais[nome])) is not None
    }
    total = sum(por_parametro.values())
    if total >= 7:
        risco = "alto"
    elif total >= 5:
        risco = "médio"
    elif any(p == 3 for p in por_parametro.values()):
        risco = "baixo-médio"
    else:
        risco = "baixo"
    return Escore(total, por_parametro, risco, _RESPOSTA[risco])


def serie(sinais: pd.DataFrame) -> pd.Series:
    """NEWS2 parcial minuto a minuto; lacunas são preenchidas com o último valor medido."""
    preenchido = sinais[list(PARAMETROS)].ffill()
    return pd.Series(
        [calcular(linha.to_dict()).total for _, linha in preenchido.iterrows()],
        index=sinais.index,
        name="NEWS2",
    )


# --- Alerta por tendência: piora do escore, não só um valor fora da faixa -------------------

JANELA_TENDENCIA_MIN = 60
SUBIDA_MINIMA = 2
_NAMESPACE = "news2-tendencia"


def alertas_de_tendencia(sinais: pd.DataFrame, paciente_id: str) -> list:
    """Alerta quando o NEWS2 parcial sobe `SUBIDA_MINIMA` pontos ou mais em até uma hora e chega
    a risco médio (5+) ou alto (7+). Um Alerta por episódio: só volta a disparar depois que o
    escore cai abaixo de 5.

    É a piora clínica que o NEWS2 quer captar: o mesmo 5 vindo de 1 é mais grave que um 5
    estável. Sai no Contrato como Sinal Vital (técnica `news2_tendencia`).
    """
    import uuid

    from src.core.alerts import Alerta, Origem, Severidade, Tipo

    escores = serie(sinais)
    alertas = []
    em_episodio = False
    for instante, total in zip(pd.DatetimeIndex(escores.index), escores, strict=True):
        if total < 5:
            em_episodio = False
            continue
        janela = escores.loc[instante - pd.Timedelta(minutes=JANELA_TENDENCIA_MIN) : instante]
        subida = int(total - janela.min())
        if em_episodio or subida < SUBIDA_MINIMA:
            continue
        em_episodio = True
        linha = sinais.loc[instante, list(PARAMETROS)].ffill()
        pontos_por = calcular(sinais.loc[:instante, list(PARAMETROS)].ffill().iloc[-1].to_dict())
        piores = [p for p, v in pontos_por.pontos.items() if v >= 2] or list(pontos_por.pontos)
        alto = total >= 7
        momento = cast(datetime, instante.to_pydatetime())  # índice sem NaT
        alertas.append(
            Alerta(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_NAMESPACE}|{paciente_id}|{momento}")),
                paciente_id=paciente_id,
                origem=Origem.SINAIS_VITAIS,
                tipo=Tipo.SINAL_VITAL,
                severidade=Severidade.CRITICA if alto else Severidade.ALTA,
                descricao=(
                    f"Piora clínica: NEWS2 subiu {subida} pontos em até "
                    f"{JANELA_TENDENCIA_MIN} min e chegou a {int(total)} "
                    f"({'risco alto' if alto else 'risco médio'})"
                ),
                evidencia={
                    "tecnica": "news2_tendencia",
                    "serie": ",".join(piores),
                    "news2": int(total),
                    "subida": subida,
                    "valores": [[round(float(v), 1)] for v in linha.fillna(0).tolist()],
                    "resposta": pontos_por.resposta,
                },
                detectado_em=momento,
            )
        )
    return alertas
