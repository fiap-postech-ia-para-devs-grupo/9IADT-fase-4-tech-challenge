"""Ficha do Paciente de Demonstração para as telas: identificação clínica e rótulos legíveis.

Idade, sexo e tipo de admissão vêm do paciente do MIMIC-IV Demo que fornece as prescrições
(docs/adr/0001: o paciente é composto, então a ficha é a dessa fonte). O leito é fictício e
fica em `data/demo_patients.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analyzers.prescriptions import load as load_prescricoes
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo
from src.core.demo import PacienteDemo

ROTULO_ORIGEM = {
    Origem.SINAIS_VITAIS: "Sinais vitais",
    Origem.AUDIO: "Áudio (Consulta)",
    Origem.VIDEO: "Vídeo",
    Origem.PRESCRICAO: "Prescrição",
    Origem.FUSAO: "Fusão",
}
ROTULO_TIPO = {
    Tipo.SINAL_VITAL: "Sinal vital",
    Tipo.TERMO_CRITICO: "Termo Crítico",
    Tipo.ALTERACAO_VOCAL: "Alteração Vocal",
    Tipo.EVENTO_SONORO: "Evento sonoro no leito",
    Tipo.DESVIO_EXECUCAO: "Desvio de Execução",
    Tipo.QUEDA: "Queda",
    Tipo.MUDANCA_TERAPEUTICA: "Mudança Terapêutica",
    Tipo.COMPOSTO: "Alerta Composto",
}
ROTULO_SEVERIDADE = {
    Severidade.CRITICA: "Crítica",
    Severidade.ALTA: "Alta",
    Severidade.MEDIA: "Média",
    Severidade.BAIXA: "Baixa",
}
ROTULO_STATUS = {
    Status.NOVO: "Novo",
    Status.RECONHECIDO: "Reconhecido",
    Status.RESOLVIDO: "Resolvido",
}
_SEXO = {"M": "masculino", "F": "feminino"}
_ADMISSAO = {
    "EW EMER.": "emergência",
    "DIRECT EMER.": "emergência (direta)",
    "URGENT": "urgência",
    "ELECTIVE": "eletiva",
    "OBSERVATION ADMIT": "observação",
    "EU OBSERVATION": "observação",
    "DIRECT OBSERVATION": "observação",
    "SURGICAL SAME DAY ADMISSION": "cirúrgica",
}

# Paciente ainda não alocado: uploads do Laboratório. Os demais ids que não são de
# demonstração são registros do MIMIC-III usados na avaliação dos Sinais Vitais.
AVULSO = "avulso"


@dataclass(frozen=True)
class Ficha:
    paciente_id: str
    nome: str
    leito: str
    idade: int | None
    sexo: str | None
    admissao: str | None

    @property
    def identificacao(self) -> str:
        partes = [self.leito or None, f"{self.idade} anos" if self.idade else None, self.sexo]
        return " · ".join(p for p in partes if p)


def carregar_fichas(
    pacientes: list[PacienteDemo], pasta: Path = load_prescricoes.PASTA_PADRAO
) -> dict[str, Ficha]:
    """Ficha de cada Paciente de Demonstração; sem o MIMIC-IV baixado, só nome e leito."""
    hosp = Path(pasta) / "hosp"
    try:
        pessoas = pd.read_csv(
            hosp / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"]
        )
        admissoes = pd.read_csv(hosp / "admissions.csv.gz", usecols=["hadm_id", "admission_type"])
    except FileNotFoundError:
        pessoas = pd.DataFrame(columns=["subject_id", "gender", "anchor_age"])
        admissoes = pd.DataFrame(columns=["hadm_id", "admission_type"])
    pessoas = pessoas.set_index("subject_id")
    admissoes = admissoes.set_index("hadm_id")

    fichas = {}
    for paciente in pacientes:
        fonte = paciente.prescricoes
        pessoa = pessoas.loc[fonte.subject_id] if fonte.subject_id in pessoas.index else None
        tipo = (
            admissoes.at[fonte.hadm_id, "admission_type"]
            if fonte.hadm_id in admissoes.index
            else None
        )
        fichas[paciente.id] = Ficha(
            paciente_id=paciente.id,
            nome=paciente.nome,
            leito=paciente.leito,
            idade=int(pessoa["anchor_age"]) if pessoa is not None else None,
            sexo=_SEXO.get(str(pessoa["gender"])) if pessoa is not None else None,
            admissao=_ADMISSAO.get(str(tipo), str(tipo).lower()) if tipo is not None else None,
        )
    return fichas


def rotulo_paciente(paciente_id: str, fichas: dict[str, Ficha]) -> str:
    if paciente_id in fichas:
        ficha = fichas[paciente_id]
        return f"{ficha.leito} · {paciente_id}" if ficha.leito else paciente_id
    if paciente_id == AVULSO:
        return "Laboratório (sem leito)"
    return f"Registro {paciente_id} (avaliação)"


def escopo(paciente_id: str, fichas: dict[str, Ficha]) -> str:
    if paciente_id in fichas:
        return "internados"
    if paciente_id == AVULSO:
        return "laboratorio"
    return "avaliacao"


def descricao_curta(alerta: Alerta) -> str:
    return f"{ROTULO_ORIGEM[alerta.origem]} · {ROTULO_TIPO[alerta.tipo]}"
