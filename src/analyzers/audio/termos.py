"""Lista curada de Termos Críticos (ESTRATEGIA.md, seção 5).

Entidades clínicas que a Text Analytics for Health extrai (`SymptomOrSign`, `Diagnosis`,
`MedicationName`...) e que, quando aparecem na fala do paciente, merecem a atenção da equipe
médica. `categoria` só é preenchida para "respiratorio" e "dor": são as duas que a Fusão
(ESTRATEGIA.md, seção 8) cruza com outras modalidades; os demais termos ficam sem categoria
(`evidencia["categoria"]` ausente, ver `EvidenciaAudio` em `src/core/alerts.py`).

    uv run python -c "
    from src.analyzers.audio.termos import localizar
    print(localizar('shortness of breath', 'SymptomOrSign'))
    "
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORIA_RESPIRATORIO = "respiratorio"
CATEGORIA_DOR = "dor"

# termo curado -> categoria (None quando não importa para a Fusão)
TERMOS_CRITICOS: dict[str, str | None] = {
    "dyspnea": CATEGORIA_RESPIRATORIO,
    "shortness of breath": CATEGORIA_RESPIRATORIO,
    "short of breath": CATEGORIA_RESPIRATORIO,
    "breathlessness": CATEGORIA_RESPIRATORIO,
    "difficulty breathing": CATEGORIA_RESPIRATORIO,
    "wheezing": CATEGORIA_RESPIRATORIO,
    "wheeze": CATEGORIA_RESPIRATORIO,
    "chest tightness": CATEGORIA_RESPIRATORIO,
    "chest pain": CATEGORIA_DOR,
    "abdominal pain": CATEGORIA_DOR,
    "headache": CATEGORIA_DOR,
    "knee pain": CATEGORIA_DOR,
    "back pain": CATEGORIA_DOR,
    "fatigue": None,
    "tiredness": None,
    "dizziness": None,
    "nausea": None,
    "vomiting": None,
    "fever": None,
    "palpitations": None,
    "confusion": None,
}

# Categorias de entidade da TA for Health que podem conter um Termo Crítico (as demais —
# MedicationName, Dosage, Age... — não são sintoma/diagnóstico e não entram na busca).
CATEGORIAS_AZURE_RELEVANTES = frozenset({"SymptomOrSign", "Diagnosis", "SymptomSignOrFinding"})


@dataclass(frozen=True)
class TermoCritico:
    termo: str
    categoria: str | None
    entidade_categoria: str
    texto_original: str


def localizar(texto: str, entidade_categoria: str) -> TermoCritico | None:
    """Casa `texto` (texto de uma entidade clínica da categoria Azure `entidade_categoria`,
    ex.: "SymptomOrSign") contra a lista curada.

    Casamento por fronteira de palavra (minúsculas): cobre tanto a entidade exata ("dyspnea")
    quanto formas mais longas que a TA for Health às vezes devolve ("mild dyspnea on exertion"
    contém "dyspnea"). Fronteira de palavra evita falso positivo de substring solta (ex.: a
    entidade "pain" não bate com "chest pain").
    """
    normalizado = texto.strip().lower()
    if not normalizado:
        return None
    for termo, categoria in TERMOS_CRITICOS.items():
        padrao = rf"\b{re.escape(termo)}\b"
        if re.search(padrao, normalizado):
            return TermoCritico(
                termo=termo,
                categoria=categoria,
                entidade_categoria=entidade_categoria,
                texto_original=texto,
            )
    return None
