"""Roda a camada semântica do Analisador de Áudio nas Consultas do PriMock57 (ESTRATEGIA.md,
seções 5 e 15): Azure STT (cacheado), WER contra a transcrição manual, Termos Críticos (TA
for Health) e Sentiment por fala do paciente.

Requer `.env` preenchido (ver `docs/azure-setup.md`) e os dados de
`uv run python scripts/download_data.py primock57`; sem chaves, roda só do cache versionado (o
WER exige o PriMock57 baixado). Cada Consulta é transcrita/analisada uma
única vez: reexecutar lê o cache em `data/audio/cache_primock57/` em vez de gastar cota.

    uv run python -m scripts.processar_audio_semantico
    uv run python -m scripts.processar_audio_semantico --gravar
"""

from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd
from dotenv import load_dotenv

from src.analyzers.audio.azure import AzureConfig, ConfigAzureIncompleta, carregar_config
from src.analyzers.audio.semantic import processar_consulta
from src.core import db

RAIZ_PRIMOCK57 = pathlib.Path("data/audio/primock57")
CACHE_DIR = pathlib.Path("data/audio/cache_primock57")
SAIDA = pathlib.Path("results/audio")
LISTA_CONSULTAS = pathlib.Path("data/audio/primock57_selecionadas.txt")


def _ler_lista() -> list[str]:
    return [
        linha.strip()
        for linha in LISTA_CONSULTAS.read_text().splitlines()
        if linha.strip() and not linha.startswith("#")
    ]


def _em_cache(consulta_id: str) -> bool:
    arquivos = (
        f"{consulta_id}_patient_16k.stt.json",
        f"{consulta_id}.ta_health_v2.json",
        f"{consulta_id}.sentiment.json",
    )
    return all((CACHE_DIR / arquivo).exists() for arquivo in arquivos)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--gravar", action="store_true", help="grava os Alertas novos no SQLite")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        config = carregar_config(os.environ)
    except ConfigAzureIncompleta as erro:
        sem_cache = [c for c in _ler_lista() if not _em_cache(c)]
        if sem_cache:
            print(f"{erro}\nConsultas sem cache: {', '.join(sem_cache)}")
            return 2
        # Avaliador sem conta Azure: tudo vem do cache versionado, nenhuma chamada é feita.
        print(f"Sem chaves do Azure: usando só o cache versionado em {CACHE_DIR}")
        config = AzureConfig("", "", "", "")

    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas_wer: list[dict[str, object]] = []
    linhas_termos: list[dict[str, object]] = []
    todos_alertas = []

    for consulta_id in _ler_lista():
        resultado = processar_consulta(
            consulta_id, RAIZ_PRIMOCK57, config, CACHE_DIR, paciente_id="avulso"
        )
        linhas_wer.append(
            {
                "consulta": consulta_id,
                "wer": resultado.wer,
                "wer_sem_hesitacoes": resultado.wer_sem_hesitacoes,
            }
        )
        for alerta in resultado.alertas_termo_critico:
            linhas_termos.append(
                {
                    "consulta": consulta_id,
                    "termo": alerta.evidencia["termo"],
                    "categoria": alerta.evidencia.get("categoria"),
                    "inicio_s": alerta.evidencia["inicio"],
                    "trecho": alerta.evidencia["trecho"],
                }
            )
        todos_alertas.extend(resultado.alertas_termo_critico)
        n_termos = len(resultado.alertas_termo_critico)
        print(f"{consulta_id}: WER={resultado.wer}, {n_termos} Termo(s) Crítico(s)")

    tabela_wer = pd.DataFrame(linhas_wer)
    pd.DataFrame(linhas_termos).to_csv(SAIDA / "termos_criticos.csv", index=False)

    n_wer = int(tabela_wer["wer"].notna().to_numpy().sum())
    if n_wer:
        tabela_wer.to_csv(SAIDA / "wer.csv", index=False)
        print(
            f"\nWER médio ({n_wer} Consultas): {tabela_wer['wer'].mean():.3f} "
            f"(sem hesitações: {tabela_wer['wer_sem_hesitacoes'].mean():.3f})"
        )
    else:
        # Sem as transcrições manuais (PriMock57 não baixado) não há WER: mantém o wer.csv.
        print("\nWER não recalculado: rode `scripts/download_data.py primock57` para medir.")
    print(f"Termos Críticos no total: {len(linhas_termos)}")

    if args.gravar:
        with db.conectar() as conn:
            novos = [a for a in todos_alertas if db.obter(conn, a.id) is None]
            for alerta in novos:
                db.salvar(conn, alerta)
        print(f"{len(novos)} Alertas novos gravados em {db.CAMINHO_PADRAO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
