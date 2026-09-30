"""Contrato do Alerta (ESTRATEGIA.md, seção 3).

Todo Analisador grava Alertas exatamente neste formato; a Central de Alertas e o SQLite
(`src/core/db.py`) o consomem.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict


class Origem(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    SINAIS_VITAIS = "sinais_vitais"
    PRESCRICAO = "prescricao"
    FUSAO = "fusao"


class Tipo(StrEnum):
    DESVIO_EXECUCAO = "desvio_execucao"
    QUEDA = "queda"
    TERMO_CRITICO = "termo_critico"
    ALTERACAO_VOCAL = "alteracao_vocal"
    SINAL_VITAL = "sinal_vital"
    MUDANCA_TERAPEUTICA = "mudanca_terapeutica"
    COMPOSTO = "composto"


class Severidade(StrEnum):
    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"


class Status(StrEnum):
    NOVO = "novo"
    RECONHECIDO = "reconhecido"
    RESOLVIDO = "resolvido"


TIPOS_POR_ORIGEM: dict[Origem, frozenset[Tipo]] = {
    Origem.VIDEO: frozenset({Tipo.DESVIO_EXECUCAO, Tipo.QUEDA}),
    Origem.AUDIO: frozenset({Tipo.TERMO_CRITICO, Tipo.ALTERACAO_VOCAL}),
    Origem.SINAIS_VITAIS: frozenset({Tipo.SINAL_VITAL}),
    Origem.PRESCRICAO: frozenset({Tipo.MUDANCA_TERAPEUTICA}),
    Origem.FUSAO: frozenset({Tipo.COMPOSTO}),
}


# Formatos de `evidencia` por origem. Campos extras são permitidos (ex.: `tecnica`,
# `probabilidade`); o único requisito validado é ser um objeto JSON.


class EvidenciaVideo(TypedDict, total=False):
    frame: str  # caminho do frame anotado com o esqueleto
    t: float  # segundos desde o início do clipe
    angulo: float


class EvidenciaAudio(TypedDict, total=False):
    trecho: str  # trecho da transcrição (ou arquivo de áudio)
    inicio: float  # segundos desde o início da Consulta


class EvidenciaSinalVital(TypedDict, total=False):
    serie: str  # "FC", "FR", "SpO2", "PAS"...
    janela: list[float]  # [t0, t1] em segundos simulados
    valores: list[float]


class EvidenciaPrescricao(TypedDict, total=False):
    droga: str
    regra: str  # "salto_de_dose", "alta_vigilancia", "trocas_excessivas", "duplicidade"


class EvidenciaComposto(TypedDict, total=False):
    regra: str  # regra de Fusão que disparou
    janela_h: float


@dataclass(frozen=True)
class Alerta:
    paciente_id: str
    origem: Origem
    tipo: Tipo
    severidade: Severidade
    descricao: str
    evidencia: dict[str, Any]
    detectado_em: datetime
    alertas_origem: list[str] = field(default_factory=list)
    status: Status = Status.NOVO
    reconhecido_por: str | None = None
    reconhecido_em: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "origem", Origem(self.origem))
        object.__setattr__(self, "tipo", Tipo(self.tipo))
        object.__setattr__(self, "severidade", Severidade(self.severidade))
        object.__setattr__(self, "status", Status(self.status))

        if self.tipo not in TIPOS_POR_ORIGEM[self.origem]:
            raise ValueError(f"tipo '{self.tipo}' não pertence à origem '{self.origem}'")
        if self.origem is Origem.FUSAO and len(self.alertas_origem) < 2:
            raise ValueError("Alerta Composto exige ao menos dois alertas_origem")
        if self.origem is not Origem.FUSAO and self.alertas_origem:
            raise ValueError("só Alerta Composto (origem 'fusao') tem alertas_origem")
        if not isinstance(self.evidencia, dict):
            raise ValueError("evidencia precisa ser um objeto JSON (dict)")
        try:
            json.dumps(self.evidencia)
        except TypeError as erro:
            raise ValueError(f"evidencia não é serializável em JSON: {erro}") from erro

    def para_dict(self) -> dict[str, Any]:
        """Dict serializável em JSON: enums como str, datetimes em ISO 8601."""
        return {
            "id": self.id,
            "paciente_id": self.paciente_id,
            "origem": str(self.origem),
            "tipo": str(self.tipo),
            "severidade": str(self.severidade),
            "descricao": self.descricao,
            "evidencia": self.evidencia,
            "detectado_em": self.detectado_em.isoformat(),
            "alertas_origem": list(self.alertas_origem),
            "status": str(self.status),
            "reconhecido_por": self.reconhecido_por,
            "reconhecido_em": self.reconhecido_em.isoformat() if self.reconhecido_em else None,
        }

    @classmethod
    def de_dict(cls, dados: dict[str, Any]) -> Alerta:
        reconhecido_em = dados.get("reconhecido_em")
        return cls(
            id=dados["id"],
            paciente_id=dados["paciente_id"],
            origem=dados["origem"],
            tipo=dados["tipo"],
            severidade=dados["severidade"],
            descricao=dados["descricao"],
            evidencia=dados["evidencia"],
            detectado_em=datetime.fromisoformat(dados["detectado_em"]),
            alertas_origem=list(dados.get("alertas_origem") or []),
            status=dados.get("status", Status.NOVO),
            reconhecido_por=dados.get("reconhecido_por"),
            reconhecido_em=datetime.fromisoformat(reconhecido_em) if reconhecido_em else None,
        )


class TransicaoInvalida(ValueError):
    """Mudança de status fora do fluxo novo → reconhecido → resolvido."""


def reconhecer(alerta: Alerta, por: str, em: datetime) -> Alerta:
    """Reconhecimento: um membro da equipe médica assume o tratamento do Alerta."""
    if alerta.status is not Status.NOVO:
        raise TransicaoInvalida(f"só Alerta novo pode ser reconhecido (está {alerta.status})")
    return replace(alerta, status=Status.RECONHECIDO, reconhecido_por=por, reconhecido_em=em)


def resolver(alerta: Alerta) -> Alerta:
    if alerta.status is not Status.RECONHECIDO:
        raise TransicaoInvalida(f"só Alerta reconhecido pode ser resolvido (está {alerta.status})")
    return replace(alerta, status=Status.RESOLVIDO)
