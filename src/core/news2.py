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
