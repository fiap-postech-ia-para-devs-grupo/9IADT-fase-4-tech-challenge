"""Fusão: Alertas de modalidades diferentes viram Alertas Compostos (ESTRATEGIA.md, seção 8).

`fundir` recebe os Alertas de um conjunto de pacientes e devolve os Alertas Compostos das regras
da tabela `REGRAS`. Uma regra dispara quando dois Alertas do mesmo paciente, de modalidades
diferentes, caem numa janela de `JANELA_FUSAO` (24 h simuladas). Os ids são determinísticos.

Convenções que dependem de outros Analisadores:
    - Termo Crítico só conta como respiratório se `evidencia["categoria"] == "respiratorio"`;
      Alteração Vocal já é respiratória por definição (CONTEXT.md).
    - Desvio de Execução "recorrente" são pelo menos `MIN_DESVIOS` Alertas do mesmo paciente
      na janela.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta

from src.analyzers.prescriptions import listas
from src.core.alerts import Alerta, Origem, Severidade, Tipo

JANELA_FUSAO = timedelta(hours=24)
MIN_DESVIOS = 2

# Princípios ativos usados pelas regras de medicação (ver listas.py para a curadoria).
OPIOIDES_E_SEDATIVOS = listas.ALTA_VIGILANCIA["opioide"] + listas.ALTA_VIGILANCIA["sedativo"]
ANTI_HIPERTENSIVOS = (
    listas.CLASSES_TERAPEUTICAS["beta_bloqueador"] + listas.CLASSES_TERAPEUTICAS["ieca_bra"]
)
ANTIARRITMICOS = ("amiodarone", "digoxin", "sotalol", "procainamide", "lidocaine")

_NAMESPACE = uuid.UUID("c3a1e0d2-7b4f-4f8e-9d31-5a2b6c8e0f47")

Condicao = Callable[[Alerta], bool]


def _series(alerta: Alerta) -> set[str]:
    return set(alerta.evidencia.get("serie", "").split(","))


def _droga(alerta: Alerta) -> str:
    return str(alerta.evidencia.get("droga", "")).lower()


def _contem(alerta: Alerta, principios: tuple[str, ...]) -> bool:
    return any(principio in _droga(alerta) for principio in principios)


def _sinal_respiratorio(a: Alerta) -> bool:
    return a.origem is Origem.SINAIS_VITAIS and bool(_series(a) & {"SpO2", "FR"})


def _sinal_hemodinamico(a: Alerta) -> bool:
    return a.origem is Origem.SINAIS_VITAIS and bool(_series(a) & {"PAS", "FC"})


def _audio_respiratorio(a: Alerta) -> bool:
    if a.origem is not Origem.AUDIO:
        return False
    if a.tipo is Tipo.ALTERACAO_VOCAL:
        return True
    return a.tipo is Tipo.TERMO_CRITICO and a.evidencia.get("categoria") == "respiratorio"


def _audio_dor(a: Alerta) -> bool:
    return (
        a.origem is Origem.AUDIO
        and a.tipo is Tipo.TERMO_CRITICO
        and a.evidencia.get("categoria") == "dor"
    )


def _queda(a: Alerta) -> bool:
    return a.origem is Origem.VIDEO and a.tipo is Tipo.QUEDA


def _desvio(a: Alerta) -> bool:
    return a.origem is Origem.VIDEO and a.tipo is Tipo.DESVIO_EXECUCAO


def _medicacao_sedativa(a: Alerta) -> bool:
    return (
        a.origem is Origem.PRESCRICAO
        and a.evidencia.get("regra") == "alta_vigilancia"
        and _contem(a, OPIOIDES_E_SEDATIVOS)
    )


def _medicacao_cardiovascular(a: Alerta) -> bool:
    return a.origem is Origem.PRESCRICAO and _contem(a, ANTI_HIPERTENSIVOS + ANTIARRITMICOS)


@dataclass(frozen=True)
class Regra:
    nome: str
    origem_a: Condicao
    origem_b: Condicao
    severidade: Severidade
    descricao: str


REGRAS: tuple[Regra, ...] = (
    Regra(
        "insuficiencia_respiratoria",
        _sinal_respiratorio,
        _audio_respiratorio,
        Severidade.CRITICA,
        "Possível insuficiência respiratória",
    ),
    Regra(
        "queda_com_sedativo",
        _queda,
        _medicacao_sedativa,
        Severidade.ALTA,
        "Queda possivelmente associada a medicação",
    ),
    Regra(
        "efeito_adverso_cardiovascular",
        _sinal_hemodinamico,
        _medicacao_cardiovascular,
        Severidade.ALTA,
        "Possível efeito adverso de medicação",
    ),
)


def _composto(
    regra: str,
    severidade: Severidade,
    descricao: str,
    origens: list[Alerta],
) -> Alerta:
    ids = sorted(a.id for a in origens)
    return Alerta(
        id=str(uuid.uuid5(_NAMESPACE, f"{regra}|{'|'.join(ids)}")),
        paciente_id=origens[0].paciente_id,
        origem=Origem.FUSAO,
        tipo=Tipo.COMPOSTO,
        severidade=severidade,
        descricao=descricao,
        evidencia={"regra": regra, "janela_h": JANELA_FUSAO.total_seconds() / 3600},
        detectado_em=max(a.detectado_em for a in origens),
        alertas_origem=ids,
    )


def _proximos(a: Alerta, b: Alerta) -> bool:
    return abs(a.detectado_em - b.detectado_em) <= JANELA_FUSAO


def fundir(alertas: Iterable[Alerta]) -> list[Alerta]:
    alertas = list(alertas)
    compostos: dict[str, Alerta] = {}

    for regra in REGRAS:
        for a in alertas:
            if not regra.origem_a(a):
                continue
            for b in alertas:
                if b.paciente_id != a.paciente_id or b.origem is a.origem:
                    continue
                if not regra.origem_b(b) or not _proximos(a, b):
                    continue
                composto = _composto(regra.nome, regra.severidade, regra.descricao, [a, b])
                compostos[composto.id] = composto

    compostos.update({c.id: c for c in _desvios_recorrentes_com_dor(alertas)})
    return sorted(compostos.values(), key=lambda c: c.detectado_em)


def _desvios_recorrentes_com_dor(alertas: list[Alerta]) -> list[Alerta]:
    """Desvio de Execução recorrente (≥ MIN_DESVIOS na janela) + Termo Crítico de dor."""
    saida: list[Alerta] = []
    for dor in (a for a in alertas if _audio_dor(a)):
        proximos = [
            d
            for d in alertas
            if _desvio(d) and d.paciente_id == dor.paciente_id and _proximos(d, dor)
        ]
        if len(proximos) >= MIN_DESVIOS:
            origens = [dor, *proximos]
            saida.append(
                _composto(
                    "agravamento_em_reabilitacao",
                    Severidade.MEDIA,
                    "Possível agravamento em reabilitação",
                    origens,
                )
            )
    return saida
