"""Catálogo das fontes de monitoramento disponíveis para ligar a uma admissão.

Cada opção é um dado público já baixado: registro de Sinais Vitais do MIMIC-III, internação do
MIMIC-IV Demo com prescrições, Consulta do PriMock57 com resposta do Azure em cache e câmera
do leito (webcam ou um clipe gravado, tocado como se fosse a câmera).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_RAIZ = Path(__file__).resolve().parents[2]
_DADOS = _RAIZ / "data"
WEBCAM = "webcam"
MIN_PRESCRICOES = 20


@dataclass(frozen=True)
class OpcaoPrescricao:
    subject_id: int
    hadm_id: int
    idade: int | None
    sexo: str | None
    inicio: pd.Timestamp  # primeira prescrição: vira a âncora
    n_prescricoes: int

    @property
    def rotulo(self) -> str:
        perfil = f"{self.idade} anos, {self.sexo}" if self.idade else "perfil desconhecido"
        return f"internação {self.hadm_id} · {perfil} · {self.n_prescricoes} prescrições"


def registros_vitais() -> list[str]:
    lista = _DADOS / "vitals" / "selecionados.txt"
    pasta = _DADOS / "vitals" / "mimic3wdb"
    if not lista.exists():
        return []
    registros = [linha.strip() for linha in lista.read_text().splitlines() if linha.strip()]
    return [r for r in registros if (pasta / f"{r}.hea").exists()]


def prescricoes() -> list[OpcaoPrescricao]:
    hosp = _DADOS / "prescriptions" / "mimic4demo" / "hosp"
    if not (hosp / "prescriptions.csv.gz").exists():
        return []
    tabela = pd.read_csv(
        hosp / "prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "starttime", "drug_type"],
        parse_dates=["starttime"],
    )
    tabela = tabela[tabela["drug_type"] == "MAIN"]
    grupos = (
        tabela.groupby(["subject_id", "hadm_id"])
        .agg(inicio=("starttime", "min"), n=("starttime", "size"))
        .reset_index()
    )
    grupos = grupos[grupos["n"] >= MIN_PRESCRICOES]
    pessoas = pd.read_csv(hosp / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    grupos = grupos.merge(pessoas, on="subject_id", how="left")
    sexo = {"M": "masculino", "F": "feminino"}
    return [
        OpcaoPrescricao(
            int(linha["subject_id"]),
            int(linha["hadm_id"]),
            int(linha["anchor_age"]) if pd.notna(linha["anchor_age"]) else None,
            sexo.get(str(linha["gender"])),
            linha["inicio"],
            int(linha["n"]),
        )
        for linha in grupos.sort_values("n", ascending=False).to_dict("records")
    ]


def consultas() -> dict[str, int]:
    """Consultas com resposta do Azure em cache e quantos Termos Críticos cada uma tem."""
    cache = _DADOS / "audio" / "cache_primock57"
    ids = sorted(
        arquivo.name.removesuffix(".ta_health_v2.json")
        for arquivo in cache.glob("*.ta_health_v2.json")
    )
    termos = _RAIZ / "results" / "audio" / "termos_criticos.csv"
    contagem: dict[str, int] = {}
    if termos.exists():
        contagem = pd.read_csv(termos)["consulta"].value_counts().to_dict()
    return {consulta: int(contagem.get(consulta, 0)) for consulta in ids}


def cameras() -> dict[str, str]:
    """id → rótulo: a webcam e os clipes baixados, tocados como câmera do leito."""
    opcoes = {WEBCAM: "Webcam deste computador (ao vivo)"}
    manifesto = _DADOS / "video" / "clipes.json"
    if manifesto.exists():
        for clipe in json.loads(manifesto.read_text(encoding="utf-8"))["clipes"]:
            if (_DADOS / "video" / clipe["arquivo"]).exists():
                opcoes[clipe["id"]] = f"Clipe {clipe['id']} — {clipe['gabarito']}"
    return opcoes


def clipe(clipe_id: str) -> dict | None:
    manifesto = _DADOS / "video" / "clipes.json"
    if not manifesto.exists():
        return None
    clipes = json.loads(manifesto.read_text(encoding="utf-8"))["clipes"]
    return next((c for c in clipes if c["id"] == clipe_id), None)
