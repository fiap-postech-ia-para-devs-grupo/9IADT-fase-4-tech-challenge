"""Avalia os detectores de Sinais Vitais em registros reais do MIMIC-III com anomalias injetadas.

    uv run python scripts/avaliar_vitals.py

Lê os registros de `data/vitals/mimic3wdb/selecionados.txt` (um caminho por linha, relativo a
`data/vitals/mimic3wdb`), injeta uma Anomalia de cada tipo por registro em trechos sem Anomalias
naturais e grava `results/vitals/avaliacao.csv` e `results/vitals/naturais.csv`.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from src.analyzers.vitals import detectors, evaluate, inject, load

RAIZ = pathlib.Path("data/vitals/mimic3wdb")
SAIDA = pathlib.Path("results/vitals")
TIPOS = list(inject.TIPOS)
DURACAO_MIN = 20
SEED = 0


def naturais(janela: pd.DataFrame, paciente: str) -> tuple[pd.Series, dict[str, int]]:
    """Máscara por minuto das Anomalias que as técnicas já acham no registro sem injeção."""
    mascara = pd.Series(False, index=janela.index)
    contagem = {}
    for tecnica in ("regras_clinicas", "zscore_movel"):
        alertas = detectors.detect(janela, paciente, tecnica)
        contagem[tecnica] = len(alertas)
        for a in alertas:
            ini, fim = a.evidencia["janela"]
            mascara.iloc[int(ini // 60) : int(fim // 60) + 1] = True
    return mascara, contagem


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    registros = [
        linha.strip()
        for linha in (RAIZ.parent / "selecionados.txt").read_text().splitlines()
        if linha.strip()
    ]
    linhas_avaliacao, linhas_naturais = [], []

    for rel in registros:
        paciente = rel.rsplit("/", 1)[-1]
        janela = load.carregar(RAIZ / rel)
        mascara, contagem = naturais(janela, paciente)
        linhas_naturais.append({"registro": paciente, "minutos": len(janela), **contagem})

        planos = inject.planejar(
            janela, TIPOS, por_tipo=1, duracao_min=DURACAO_MIN, excluir=mascara, seed=SEED
        )
        if not planos:
            continue
        injetada, gabarito = inject.aplicar(janela, planos, DURACAO_MIN)
        por_tecnica = {t: detectors.detect(injetada, paciente, t) for t in detectors.TECNICAS}
        tabela = evaluate.avaliar(por_tecnica, gabarito, janela.index[0].to_pydatetime())
        tabela.insert(0, "registro", paciente)
        linhas_avaliacao.append(tabela)

    avaliacao = pd.concat(linhas_avaliacao, ignore_index=True)
    avaliacao.to_csv(SAIDA / "avaliacao_por_registro.csv", index=False)

    resumo = (
        avaliacao.groupby("tecnica")
        .agg(
            injecoes=("injecoes", "sum"),
            detectadas=("detectadas", "sum"),
            alertas=("alertas", "sum"),
            atraso_mediano_min=("atraso_mediano_min", "median"),
        )
        .reset_index()
    )
    resumo["recall"] = resumo["detectadas"] / resumo["injecoes"]
    resumo["precisao"] = (
        avaliacao.assign(uteis=avaliacao["precisao"] * avaliacao["alertas"])
        .groupby("tecnica")["uteis"]
        .sum()
        .reindex(resumo["tecnica"])
        .to_numpy()
        / resumo["alertas"]
    )
    resumo.to_csv(SAIDA / "avaliacao.csv", index=False)
    pd.DataFrame(linhas_naturais).to_csv(SAIDA / "naturais.csv", index=False)
    print(resumo.round(3).to_string(index=False))
    print(pd.DataFrame(linhas_naturais).to_string(index=False))


if __name__ == "__main__":
    main()
