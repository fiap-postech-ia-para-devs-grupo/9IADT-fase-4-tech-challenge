"""Mudança Terapêutica Atípica: 4 regras sobre as prescrições (ESTRATEGIA.md, seção 7).

    salto_de_dose       aumento de dose > 50% da mesma droga em < 24h
    alta_vigilancia     início de droga de alta vigilância (primeira da classe na internação)
    trocas_excessivas   mais de N trocas de princípio ativo na mesma classe em 48h
    duplicidade         duas drogas da mesma classe ativas ao mesmo tempo

`analisar` recebe as prescrições de `load.carregar` e devolve Alertas no Contrato do Alerta
(`origem=prescricao`, `tipo=mudanca_terapeutica`, `evidencia["regra"]` com o nome da regra).
Os ids são determinísticos: reanalisar os mesmos dados gera os mesmos Alertas.
"""

# pyright: reportAttributeAccessIssue=false
# (as stubs do pandas tipam as linhas de `itertuples()` como tuple; o acesso por nome é o idioma)

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.analyzers.prescriptions import listas
from src.core.alerts import Alerta, Origem, Severidade, Tipo

LIMITE_SALTO = 1.5  # nova dose / dose anterior
JANELA_SALTO = timedelta(hours=24)
JANELA_TROCAS = timedelta(hours=48)
MAX_TROCAS = 4  # mais que isso na janela é excessivo

REGRAS = ("salto_de_dose", "alta_vigilancia", "trocas_excessivas", "duplicidade")

_NAMESPACE = uuid.UUID("6f1c4d1e-3a52-4a56-9a52-6d0c9e0d4a11")


def paciente_id(subject_id: int) -> str:
    return f"mimic4-{subject_id}"


def _alerta(
    regra: str,
    subject_id: int,
    hadm_id: int,
    quando: datetime,
    severidade: Severidade,
    descricao: str,
    chave: str,
    **evidencia: Any,
) -> Alerta:
    return Alerta(
        id=str(uuid.uuid5(_NAMESPACE, f"{regra}|{hadm_id}|{chave}|{quando.isoformat()}")),
        paciente_id=paciente_id(subject_id),
        origem=Origem.PRESCRICAO,
        tipo=Tipo.MUDANCA_TERAPEUTICA,
        severidade=severidade,
        descricao=descricao,
        evidencia={
            "regra": regra,
            "hadm_id": hadm_id,
            "inicio": quando.isoformat(),
            "versao_listas": listas.VERSAO,
            **evidencia,
        },
        detectado_em=quando,
    )


def _terapia(prescricoes: pd.DataFrame) -> pd.DataFrame:
    """Só terapia de verdade: sem soluções de manutenção de acesso (flush/lock)."""
    rotina = prescricoes["drug"].map(listas.eh_rotina).astype(bool)
    return prescricoes.loc[~rotina]


def _internacoes(prescricoes: pd.DataFrame) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """(subject_id, hadm_id, prescrições da internação em ordem cronológica)."""
    for chave, grupo in prescricoes.groupby(["subject_id", "hadm_id"]):
        subject_id, hadm_id = chave  # type: ignore[misc]
        yield int(subject_id), int(hadm_id), grupo.sort_values("starttime")


def _texto(valor: Any) -> str | None:
    return None if pd.isna(valor) else str(valor)


def _numero(valor: float) -> str:
    return f"{valor:g}"


def salto_de_dose(prescricoes: pd.DataFrame) -> list[Alerta]:
    alertas: list[Alerta] = []
    terapia = _terapia(prescricoes)
    reposicao = terapia["drug"].map(listas.eh_reposicao).astype(bool)
    com_dose = terapia.loc[~reposicao].dropna(subset=["dose"])
    com_dose = com_dose[com_dose["dose"] > 0].assign(droga_norm=lambda d: d["drug"].str.lower())
    chaves = ["hadm_id", "droga_norm", "route", "dose_unit_rx"]
    for _, grupo in com_dose.groupby(chaves, dropna=False):
        anterior = None
        for linha in grupo.sort_values("starttime").itertuples():
            if (
                anterior is not None
                and linha.dose > anterior.dose * LIMITE_SALTO
                and linha.starttime - anterior.starttime < JANELA_SALTO
            ):
                unidade = _texto(linha.dose_unit_rx) or ""
                alertas.append(
                    _alerta(
                        "salto_de_dose",
                        linha.subject_id,
                        linha.hadm_id,
                        linha.starttime.to_pydatetime(),
                        Severidade.ALTA if listas.alta_vigilancia(linha.drug) else Severidade.MEDIA,
                        f"Aumento de dose > 50% em < 24h: {linha.drug} "
                        f"{_numero(anterior.dose)} → {_numero(linha.dose)} {unidade}".rstrip(),
                        chave=linha.droga_norm,
                        droga=linha.drug,
                        dose_anterior=anterior.dose,
                        dose_nova=linha.dose,
                        unidade=unidade or None,
                        via=_texto(linha.route),
                    )
                )
            anterior = linha
    return alertas


def alta_vigilancia(prescricoes: pd.DataFrame, emar: pd.DataFrame | None = None) -> list[Alerta]:
    """Primeira prescrição de cada classe de alta vigilância por internação."""
    alertas: list[Alerta] = []
    for subject_id, hadm_id, grupo in _internacoes(_terapia(prescricoes)):
        vistas: set[str] = set()
        for linha in grupo.itertuples():
            principio = listas.alta_vigilancia(linha.drug)
            if principio is None or principio.classe in vistas:
                continue
            vistas.add(principio.classe)
            evidencia: dict[str, Any] = {
                "droga": linha.drug,
                "classe": principio.classe,
                "dose": None if pd.isna(linha.dose) else linha.dose,
                "unidade": _texto(linha.dose_unit_rx),
                "via": _texto(linha.route),
            }
            administrada = _primeira_administracao(emar, hadm_id, principio.ativo)
            if administrada is not None:
                evidencia["primeira_administracao"] = administrada.isoformat()
            grave = principio.classe in {"opioide", "anticoagulante"}
            alertas.append(
                _alerta(
                    "alta_vigilancia",
                    subject_id,
                    hadm_id,
                    linha.starttime.to_pydatetime(),
                    Severidade.ALTA if grave else Severidade.MEDIA,
                    f"Início de droga de alta vigilância ({principio.classe}): {linha.drug}",
                    chave=principio.classe,
                    **evidencia,
                )
            )
    return alertas


def trocas_excessivas(prescricoes: pd.DataFrame) -> list[Alerta]:
    """Trocas de princípio ativo dentro da mesma classe; > MAX_TROCAS em 48h dispara.

    Suspensões não entram na conta: o MIMIC só registra o `stoptime` prescrito, não o momento em
    que a droga foi realmente suspensa.
    """
    alertas: list[Alerta] = []
    for subject_id, hadm_id, grupo in _internacoes(_terapia(prescricoes)):
        trocas: list[tuple[datetime, str, str, str]] = []
        ultimo: dict[str, str] = {}
        for linha in grupo.itertuples():
            principio = listas.classe_terapeutica(linha.drug)
            if principio is None:
                continue
            anterior = ultimo.get(principio.classe)
            if anterior is not None and anterior != principio.ativo:
                trocas.append(
                    (linha.starttime.to_pydatetime(), principio.classe, anterior, principio.ativo)
                )
            ultimo[principio.classe] = principio.ativo

        silencio_ate = datetime.min
        inicio = 0
        for i, (quando, *_) in enumerate(trocas):
            while quando - trocas[inicio][0] > JANELA_TROCAS:
                inicio += 1
            na_janela = trocas[inicio : i + 1]
            if len(na_janela) > MAX_TROCAS and quando >= silencio_ate:
                alertas.append(
                    _alerta(
                        "trocas_excessivas",
                        subject_id,
                        hadm_id,
                        quando,
                        Severidade.BAIXA,
                        f"{len(na_janela)} trocas de medicação em 48h",
                        chave="trocas",
                        trocas=[
                            {"em": t.isoformat(), "classe": c, "de": a, "para": b}
                            for t, c, a, b in na_janela
                        ],
                        n_trocas=len(na_janela),
                        droga=na_janela[-1][3],
                    )
                )
                silencio_ate = quando + JANELA_TROCAS  # um Alerta por episódio
    return alertas


def duplicidade(prescricoes: pd.DataFrame) -> list[Alerta]:
    """Duas drogas da mesma classe, de princípios ativos diferentes, ativas ao mesmo tempo."""
    alertas: list[Alerta] = []
    for subject_id, hadm_id, grupo in _internacoes(_terapia(prescricoes)):
        ativas: dict[str, list[tuple[Any, listas.Principio]]] = {}
        ja_alertados: set[tuple[str, frozenset[str]]] = set()
        for linha in grupo.itertuples():
            principio = listas.classe_terapeutica(linha.drug)
            if principio is None:
                continue
            vigentes = [
                (outra, p)
                for outra, p in ativas.get(principio.classe, [])
                if outra.stoptime > linha.starttime
            ]
            for outra, p in vigentes:
                par = frozenset({p.ativo, principio.ativo})
                if p.ativo == principio.ativo or (principio.classe, par) in ja_alertados:
                    continue
                ja_alertados.add((principio.classe, par))
                grave = listas.alta_vigilancia(linha.drug) is not None
                alertas.append(
                    _alerta(
                        "duplicidade",
                        subject_id,
                        hadm_id,
                        linha.starttime.to_pydatetime(),
                        Severidade.ALTA if grave else Severidade.MEDIA,
                        f"Duplicidade terapêutica ({principio.classe}): "
                        f"{outra.drug} e {linha.drug} ativas ao mesmo tempo",
                        chave="|".join(sorted(par)),
                        droga=linha.drug,
                        classe=principio.classe,
                        drogas=[outra.drug, linha.drug],
                        sobreposicao_ate=min(outra.stoptime, linha.stoptime).isoformat(),
                    )
                )
            ativas[principio.classe] = [*vigentes, (linha, principio)]
    return alertas


def analisar(prescricoes: pd.DataFrame, emar: pd.DataFrame | None = None) -> list[Alerta]:
    """As 4 regras sobre todas as internações, em ordem cronológica."""
    alertas = [
        *salto_de_dose(prescricoes),
        *alta_vigilancia(prescricoes, emar),
        *trocas_excessivas(prescricoes),
        *duplicidade(prescricoes),
    ]
    return sorted(alertas, key=lambda a: a.detectado_em)


def contar_por_regra(alertas: list[Alerta]) -> dict[str, int]:
    contagem: dict[str, int] = dict.fromkeys(REGRAS, 0)
    for alerta in alertas:
        contagem[alerta.evidencia["regra"]] += 1
    return contagem


def _primeira_administracao(emar: pd.DataFrame | None, hadm_id: int, ativo: str) -> datetime | None:
    """Primeira dose de fato dada ao paciente (eMAR), para confirmar o início da droga."""
    if emar is None:
        return None
    dadas = emar[
        (emar["hadm_id"] == hadm_id)
        & (emar["event_txt"] == "Administered")
        & emar["medication"].str.lower().str.contains(ativo, regex=False, na=False)
    ]
    return None if dadas.empty else dadas["charttime"].min().to_pydatetime()
