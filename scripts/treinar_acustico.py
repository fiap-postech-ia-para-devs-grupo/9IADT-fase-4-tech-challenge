"""Treina e avalia o classificador acústico no subset do Coswara (ESTRATEGIA.md, seção 6).

Lê `data/audio/coswara/combined_data.csv` (rótulos) e `data/audio/coswara/Extracted_data/`
(áudios, baixados por `scripts/download_data.py coswara`), balanceia as classes, separa
treino/teste por pessoa, treina Regressão Logística e Random Forest, grava as métricas em
`results/audio/` e persiste o modelo escolhido em `data/audio/modelo_acustico.joblib` (usado
por `src/analyzers/audio/lab.py` na inferência da Tela 3).

Também roda a inferência sobre a fala do paciente das Consultas do PriMock57 já baixadas
(`scripts/download_data.py primock57`) e grava `results/audio/alteracao_vocal_primock57.csv` +
os Alertas `alteracao_vocal` (Contrato do Alerta) de quem passa do limiar; não depende do
Azure, só do classificador local.

    uv run python -m scripts.treinar_acustico
    uv run python -m scripts.treinar_acustico --gravar   # grava os Alertas novos no SQLite
"""

# pyright: reportArgumentType=false
# (stub do pandas: indexação por máscara booleana às vezes infere Series em vez de DataFrame)

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from src.analyzers.audio import acoustic
from src.core import db

RAIZ_COSWARA = pathlib.Path("data/audio/coswara")
RAIZ_PRIMOCK57 = pathlib.Path("data/audio/primock57")
SAIDA = pathlib.Path("results/audio")
MODELO = pathlib.Path("data/audio/modelo_acustico.joblib")
POR_CLASSE = 300  # alvo por classe (ESTRATEGIA.md, seção 5); usa o que houver, se for menos
SEED = 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--gravar", action="store_true", help="grava os Alertas novos no SQLite")
    args = parser.parse_args(argv)

    SAIDA.mkdir(parents=True, exist_ok=True)

    metadados = acoustic.carregar_metadados(RAIZ_COSWARA / "combined_data.csv")
    disponiveis = acoustic.localizar_arquivos(RAIZ_COSWARA / "Extracted_data")
    metadados_com_audio = metadados[metadados["id"].isin(list(disponiveis))]
    balanceado = acoustic.balancear_por_pessoa(
        metadados_com_audio, por_classe=POR_CLASSE, seed=SEED
    )

    print(f"Pessoas rotuladas com áudio extraído: {len(metadados_com_audio)}")
    print(f"Pessoas após balancear (até {POR_CLASSE}/classe): {len(balanceado)}")
    print(balanceado["rotulo"].value_counts().rename({0: "saudável", 1: "sintomático"}))

    dataset = acoustic.montar_dataset(balanceado, RAIZ_COSWARA / "Extracted_data")
    print(f"Linhas no dataset (1 por gravação): {len(dataset)}")

    treino, teste = acoustic.dividir_por_pessoa(dataset, teste=0.25, seed=SEED)
    print(f"Treino: {treino['id'].nunique()} pessoas / {len(treino)} gravações")
    print(f"Teste:  {teste['id'].nunique()} pessoas / {len(teste)} gravações")

    resultados, contexto = acoustic.treinar_avaliar(treino, teste, seed=SEED)

    linhas = []
    for nome, resultado in resultados.items():
        (tn, fp), (fn, tp) = resultado.matriz_confusao
        linhas.append(
            {
                "modelo": nome,
                "auc": round(resultado.auc, 3),
                "f1": round(resultado.f1, 3),
                "limiar": round(resultado.limiar, 3),
                "verdadeiro_negativo": tn,
                "falso_positivo": fp,
                "falso_negativo": fn,
                "verdadeiro_positivo": tp,
            }
        )
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(SAIDA / "acoustic_metricas.csv", index=False)
    print()
    print(tabela.to_string(index=False))

    melhor = max(resultados, key=lambda nome: resultados[nome].auc)
    acoustic.salvar_modelo(melhor, resultados, contexto, MODELO)
    print(f"\nModelo escolhido (maior AUC): {melhor} -> {MODELO}")

    resumo = {
        "por_classe_alvo": POR_CLASSE,
        "pessoas_treino": int(treino["id"].nunique()),
        "pessoas_teste": int(teste["id"].nunique()),
        "colunas_features": contexto["colunas"],
        "modelo_escolhido": melhor,
        "limiar_escolhido": resultados[melhor].limiar,
    }
    (SAIDA / "acoustic_resumo.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")

    modelo_salvo = acoustic.carregar_modelo(MODELO)
    linhas_inferencia = []
    alertas = []
    for arquivo in sorted((RAIZ_PRIMOCK57 / "audio").glob("*_patient.wav")):
        consulta_id = arquivo.stem.removesuffix("_patient")
        probabilidade = acoustic.inferir(arquivo, modelo_salvo)
        linhas_inferencia.append(
            {
                "consulta": consulta_id,
                "probabilidade": round(probabilidade, 3),
                "alteracao_vocal": probabilidade >= modelo_salvo["limiar"],
            }
        )
        alerta = acoustic.alerta_alteracao_vocal(
            "avulso", consulta_id, probabilidade, modelo_salvo["limiar"]
        )
        if alerta is not None:
            alertas.append(alerta)

    if linhas_inferencia:
        tabela_inferencia = pd.DataFrame(linhas_inferencia)
        tabela_inferencia.to_csv(SAIDA / "alteracao_vocal_primock57.csv", index=False)
        n_total = len(tabela_inferencia)
        print(f"\nInferência no PriMock57: {len(alertas)}/{n_total} com Alteração Vocal")

    if args.gravar:
        with db.conectar() as conn:
            novos = [a for a in alertas if db.obter(conn, a.id) is None]
            for alerta in novos:
                db.salvar(conn, alerta)
        print(f"{len(novos)} Alertas novos gravados em {db.CAMINHO_PADRAO}")


if __name__ == "__main__":
    main()
