"""Roda o Analisador de Prescrições nos 100 pacientes do MIMIC-IV Demo.

Gera em `results/prescriptions/`: contagem de Alertas por regra (CSV + Markdown) e a linha do
tempo de prescrições dos pacientes com mais regras distintas disparadas (HTML). Com `--gravar`,
grava os Alertas novos no SQLite (sem sobrescrever os que já foram reconhecidos).

    uv run python -m scripts.analisar_prescricoes
    uv run python -m scripts.analisar_prescricoes --gravar --linhas-do-tempo 2
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.analyzers.prescriptions import listas, load, rules, timeline
from src.core import db
from src.core.alerts import Alerta

RESULTADOS = Path(__file__).resolve().parent.parent / "results" / "prescriptions"


def contagem_por_regra(alertas: list[Alerta], n_pacientes: int) -> pd.DataFrame:
    contagem = rules.contar_por_regra(alertas)
    pacientes_por_regra = {
        regra: len({a.paciente_id for a in alertas if a.evidencia["regra"] == regra})
        for regra in rules.REGRAS
    }
    return pd.DataFrame(
        {
            "regra": list(rules.REGRAS),
            "alertas": [contagem[r] for r in rules.REGRAS],
            "pacientes": [pacientes_por_regra[r] for r in rules.REGRAS],
            "pct_pacientes": [
                round(100 * pacientes_por_regra[r] / n_pacientes) for r in rules.REGRAS
            ],
        }
    )


def _markdown(tabela: pd.DataFrame) -> str:
    linhas = [
        "| " + " | ".join(tabela.columns) + " |",
        "| " + " | ".join("---" for _ in tabela.columns) + " |",
    ]
    linhas += [
        "| " + " | ".join(str(v) for v in linha) + " |" for linha in tabela.itertuples(index=False)
    ]
    return "\n".join(linhas)


def pacientes_para_linha_do_tempo(alertas: list[Alerta], quantos: int) -> list[tuple[int, int]]:
    """(subject_id, hadm_id) das internações com mais regras distintas; empate: mais Alertas."""
    regras: dict[tuple[int, int], set[str]] = {}
    total: Counter[tuple[int, int]] = Counter()
    for alerta in alertas:
        chave = (int(alerta.paciente_id.split("-")[1]), alerta.evidencia["hadm_id"])
        regras.setdefault(chave, set()).add(alerta.evidencia["regra"])
        total[chave] += 1
    ordenadas = sorted(regras, key=lambda c: (-len(regras[c]), -total[c], c))
    return ordenadas[:quantos]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--gravar", action="store_true", help="grava os Alertas novos no SQLite")
    parser.add_argument("--linhas-do-tempo", type=int, default=2, metavar="N")
    args = parser.parse_args(argv)

    dados = load.carregar()
    alertas = rules.analisar(dados.prescricoes, dados.emar)
    n_pacientes = len(dados.prescricoes["subject_id"].unique())

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela = contagem_por_regra(alertas, n_pacientes)
    tabela.to_csv(RESULTADOS / "contagem_por_regra.csv", index=False)
    (RESULTADOS / "contagem_por_regra.md").write_text(
        f"Listas v{listas.VERSAO} · {n_pacientes} pacientes · {len(alertas)} Alertas\n\n"
        + _markdown(tabela)
        + "\n",
        encoding="utf-8",
    )
    print(tabela.to_string(index=False))

    for subject_id, hadm_id in pacientes_para_linha_do_tempo(alertas, args.linhas_do_tempo):
        figura = timeline.figura(dados.prescricoes, alertas, hadm_id)
        destino = RESULTADOS / f"linha_do_tempo_{subject_id}_{hadm_id}.html"
        figura.write_html(destino, include_plotlyjs="cdn")
        print(f"linha do tempo: {destino.relative_to(RESULTADOS.parent.parent)}")

    if args.gravar:
        with db.conectar() as conn:
            novos = [a for a in alertas if db.obter(conn, a.id) is None]
            for alerta in novos:
                db.salvar(conn, alerta)
        print(f"{len(novos)} Alertas novos gravados em {db.CAMINHO_PADRAO}")


if __name__ == "__main__":
    main()
