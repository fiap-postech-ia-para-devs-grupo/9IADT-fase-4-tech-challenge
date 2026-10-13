"""Listas de drogas de alta vigilância e classes terapêuticas (ESTRATEGIA.md, seção 7).

Curadas à mão para o escopo do Tech Challenge e versionadas aqui: qualquer mudança nas listas
muda `VERSAO` e, com ela, a contagem de Alertas por regra em `results/prescriptions/`.

Cada lista mapeia uma classe para os **princípios ativos** (comparação por substring, sem
diferenciar maiúsculas) que a compõem. O princípio ativo, e não o nome da prescrição, é a
unidade de comparação: "Metoprolol Tartrate" e "Metoprolol Succinate XL" são o mesmo princípio.
"""

from __future__ import annotations

from typing import NamedTuple

VERSAO = "1.0.0"

# Início de uma destas drogas é uma Mudança Terapêutica Atípica por si só.
ALTA_VIGILANCIA: dict[str, tuple[str, ...]] = {
    "anticoagulante": (
        "heparin",
        "enoxaparin",
        "warfarin",
        "argatroban",
        "fondaparinux",
        "apixaban",
        "rivaroxaban",
        "dabigatran",
    ),
    "insulina": ("insulin",),
    "opioide": (
        "morphine",
        "hydromorphone",
        "dilaudid",
        "oxycodone",
        "fentanyl",
        "methadone",
        "hydrocodone",
        "tramadol",
        "meperidine",
        "remifentanil",
    ),
    "sedativo": (
        "lorazepam",
        "midazolam",
        "diazepam",
        "alprazolam",
        "clonazepam",
        "propofol",
        "dexmedetomidine",
        "zolpidem",
    ),
}

# Duas drogas ativas ao mesmo tempo na mesma classe, com princípios ativos distintos, é
# Duplicidade Terapêutica. Ficam de fora combinações que são esquema padrão:
# heparina + varfarina (ponte de anticoagulação), insulina basal + rápida e AAS + clopidogrel.
CLASSES_TERAPEUTICAS: dict[str, tuple[str, ...]] = {
    "opioide": ALTA_VIGILANCIA["opioide"],
    "benzodiazepina": ("lorazepam", "midazolam", "diazepam", "alprazolam", "clonazepam"),
    "anti_inflamatorio_nao_esteroide": ("ibuprofen", "ketorolac", "naproxen", "celecoxib"),
    "inibidor_bomba_protons": ("pantoprazole", "omeprazole", "esomeprazole", "lansoprazole"),
    "antagonista_h2": ("famotidine", "ranitidine"),
    "beta_bloqueador": ("metoprolol", "atenolol", "carvedilol", "labetalol", "propranolol"),
    "ieca_bra": ("lisinopril", "enalapril", "captopril", "losartan", "valsartan"),
    "estatina": ("atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"),
    "diuretico_de_alca": ("furosemide", "torsemide", "bumetanide"),
    "antiemetico": ("ondansetron", "metoclopramide", "prochlorperazine", "promethazine"),
}

# Reposição de eletrólitos e glicose é titulada pelo resultado de exames, então doses que sobem e
# descem são rotina: ficam fora da regra de salto de dose.
REPOSICAO = (
    "magnesium",
    "calcium",
    "potassium",
    "phosph",
    "sodium chloride",
    "sodium bicarbonate",
    "dextrose",
)

# Soluções de manutenção de acesso, não terapia: "Heparin Flush" não é anticoagulação.
ROTINA = ("flush", "lock")


class Principio(NamedTuple):
    classe: str
    ativo: str


def eh_rotina(droga: str) -> bool:
    return any(termo in droga.lower() for termo in ROTINA)


def eh_reposicao(droga: str) -> bool:
    return any(termo in droga.lower() for termo in REPOSICAO)


def identificar(droga: str, tabela: dict[str, tuple[str, ...]]) -> Principio | None:
    """Classe e princípio ativo de `droga` na `tabela`, ou None se não constar."""
    nome = droga.lower()
    if eh_rotina(nome):
        return None
    for classe, ativos in tabela.items():
        for ativo in ativos:
            if ativo in nome:
                return Principio(classe, ativo)
    return None


def alta_vigilancia(droga: str) -> Principio | None:
    return identificar(droga, ALTA_VIGILANCIA)


def classe_terapeutica(droga: str) -> Principio | None:
    return identificar(droga, CLASSES_TERAPEUTICAS)
