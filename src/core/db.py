"""Persistência dos Alertas em SQLite.

Uso típico (Analisadores e Streamlit):

    conn = db.conectar()             # cria o schema se preciso
    db.salvar(conn, alerta)
    db.listar(conn, status=Status.NOVO)
    db.reconhecer(conn, alerta.id, por="enf. Ana")
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.core import alerts
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo

_RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = Path(os.environ.get("ALERTAS_DB", _RAIZ / "data" / "alertas.db"))


def _enum_check(coluna: str, enum: type[StrEnum]) -> str:
    valores = ", ".join(f"'{membro.value}'" for membro in enum)
    return f"CHECK ({coluna} IN ({valores}))"


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS alertas (
    id              TEXT PRIMARY KEY,
    paciente_id     TEXT NOT NULL,
    origem          TEXT NOT NULL {_enum_check("origem", Origem)},
    tipo            TEXT NOT NULL {_enum_check("tipo", Tipo)},
    severidade      TEXT NOT NULL {_enum_check("severidade", Severidade)},
    descricao       TEXT NOT NULL,
    evidencia       TEXT NOT NULL,              -- JSON
    detectado_em    TEXT NOT NULL,              -- ISO 8601
    alertas_origem  TEXT NOT NULL DEFAULT '[]', -- JSON list, só em Alerta Composto
    status          TEXT NOT NULL {_enum_check("status", Status)},
    reconhecido_por TEXT,
    reconhecido_em  TEXT                        -- ISO 8601
);
CREATE INDEX IF NOT EXISTS idx_alertas_paciente ON alertas (paciente_id, detectado_em);
"""

_PESO_SEVERIDADE = " ".join(
    f"WHEN '{severidade.value}' THEN {peso}" for peso, severidade in enumerate(Severidade)
)

Filtro = str | Iterable[str] | None

_COLUNAS = tuple(campo.name for campo in fields(Alerta))


def conectar(caminho: str | Path = CAMINHO_PADRAO) -> sqlite3.Connection:
    """Abre (ou cria) o banco de Alertas e garante o schema."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(caminho)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def salvar(conn: sqlite3.Connection, alerta: Alerta) -> None:
    """Grava o Alerta; se o id já existe, sobrescreve."""
    linha = _para_linha(alerta)
    marcadores = ", ".join("?" for _ in _COLUNAS)
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO alertas ({', '.join(_COLUNAS)}) VALUES ({marcadores})",
            [linha[coluna] for coluna in _COLUNAS],
        )


def obter(conn: sqlite3.Connection, alerta_id: str) -> Alerta | None:
    linha = conn.execute("SELECT * FROM alertas WHERE id = ?", (alerta_id,)).fetchone()
    return _de_linha(linha) if linha else None


def listar(
    conn: sqlite3.Connection,
    *,
    origem: Filtro = None,
    status: Filtro = None,
    paciente_id: Filtro = None,
) -> list[Alerta]:
    """Alertas mais severos primeiro e, na mesma severidade, mais recentes primeiro.

    Cada filtro aceita um valor ou uma coleção de valores (ex.: multiselect do Streamlit);
    `None` não filtra.
    """
    condicoes: list[str] = []
    parametros: list[str] = []
    for coluna, filtro in (("origem", origem), ("status", status), ("paciente_id", paciente_id)):
        if filtro is None:
            continue
        valores = [str(filtro)] if isinstance(filtro, str) else [str(v) for v in filtro]
        condicoes.append(f"{coluna} IN ({', '.join('?' for _ in valores)})" if valores else "0")
        parametros.extend(valores)

    where = f"WHERE {' AND '.join(condicoes)}" if condicoes else ""
    linhas = conn.execute(
        f"SELECT * FROM alertas {where} "
        f"ORDER BY CASE severidade {_PESO_SEVERIDADE} END DESC, detectado_em DESC",
        parametros,
    ).fetchall()
    return [_de_linha(linha) for linha in linhas]


def reconhecer(
    conn: sqlite3.Connection, alerta_id: str, por: str, em: datetime | None = None
) -> Alerta:
    """Reconhecimento na Central de Alertas; `em` padrão é agora."""
    reconhecido = alerts.reconhecer(_obter_ou_falhar(conn, alerta_id), por, em or datetime.now())
    salvar(conn, reconhecido)
    return reconhecido


def resolver(conn: sqlite3.Connection, alerta_id: str) -> Alerta:
    resolvido = alerts.resolver(_obter_ou_falhar(conn, alerta_id))
    salvar(conn, resolvido)
    return resolvido


def _obter_ou_falhar(conn: sqlite3.Connection, alerta_id: str) -> Alerta:
    alerta = obter(conn, alerta_id)
    if alerta is None:
        raise KeyError(f"Alerta {alerta_id} não existe")
    return alerta


def _para_linha(alerta: Alerta) -> dict[str, Any]:
    dados = alerta.para_dict()
    dados["evidencia"] = json.dumps(dados["evidencia"], ensure_ascii=False)
    dados["alertas_origem"] = json.dumps(dados["alertas_origem"])
    return dados


def _de_linha(linha: sqlite3.Row) -> Alerta:
    dados = dict(linha)
    dados["evidencia"] = json.loads(dados["evidencia"])
    dados["alertas_origem"] = json.loads(dados["alertas_origem"])
    return Alerta.de_dict(dados)
