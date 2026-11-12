"""Roda o Analisador de Vídeo em todos os clipes de `data/video/clipes.json` (ESTRATEGIA.md,
seção 4): Relatório de Sessão de cada clipe em `results/video/<clipe>/`, Alertas no SQLite e a
avaliação esperados × detectados em `results/video/avaliacao.csv`.

    uv run python scripts/download_data.py video
    uv run python -m scripts.processar_video

O clipe ligado a um Paciente de Demonstração (`video.clipe` em `data/demo_patients.json`) gera
Alertas para esse paciente, no horário da sessão na internação simulada; os outros saem como
`avulso`. Reexecutar não duplica Alertas nem desfaz Reconhecimentos.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analyzers.video import pose
from src.analyzers.video.lab import SAIDA_PADRAO, analyze_video
from src.core import db, demo
from src.core.alerts import Status

RAIZ = Path(__file__).resolve().parents[1]
PASTA_VIDEO = RAIZ / "data" / "video"
MANIFESTO = PASTA_VIDEO / "clipes.json"


def carregar_clipes(caminho: Path = MANIFESTO) -> list[dict]:
    return json.loads(caminho.read_text(encoding="utf-8"))["clipes"]


def sessoes_dos_pacientes() -> dict[str, tuple[str, datetime]]:
    """clipe → (paciente, início da sessão na internação simulada)."""
    return {
        paciente.video.clipe: (paciente.id, paciente.em(paciente.video.inicio_min))
        for paciente in demo.carregar_pacientes()
        if paciente.video.clipe
    }


def avaliar(clipe: dict, resultado) -> dict:
    esperado = clipe["esperado"]
    seq = resultado.sequencia
    return {
        "clipe": clipe["id"],
        "cenario": clipe["cenario"],
        "exercicio": clipe["exercicio"] or "",
        "vista": clipe["vista"],
        "repeticoes": len(resultado.repeticoes),
        "desvios_esperados": esperado["desvios"],
        "desvios_detectados": len(resultado.desvios),
        "quedas_esperadas": esperado["quedas"],
        "quedas_detectadas": len(resultado.quedas),
        "duracao_s": round(seq.duracao, 1),
        "cpu_s": round(seq.segundos_processamento, 1),
        "cpu_por_s_video": round(resultado.fator_tempo_real, 2),
        "cobertura_pose": round(resultado.cobertura, 2),
        "gabarito": clipe["gabarito"],
    }


def main() -> int:
    clipes = carregar_clipes()
    faltando = [c["arquivo"] for c in clipes if not (PASTA_VIDEO / c["arquivo"]).exists()]
    if faltando:
        print(f"Clipes ausentes ({len(faltando)}): uv run python scripts/download_data.py video")
        return 1

    sessoes = sessoes_dos_pacientes()
    modelo = pose.carregar_modelo()
    linhas = []
    novos = 0
    with db.conectar() as conn:
        for clipe in clipes:
            paciente_id, inicio = sessoes.get(clipe["id"], ("avulso", None))
            resultado = analyze_video(
                PASTA_VIDEO / clipe["arquivo"],
                exercicio=clipe["exercicio"],
                vista=clipe["vista"],
                paciente_id=paciente_id,
                clipe_id=clipe["id"],
                inicio=inicio,
                recorte=tuple(clipe["recorte"]) if clipe["recorte"] else None,
                modelo=modelo,
            )
            for alerta in resultado.alertas:
                gravado = db.obter(conn, alerta.id)
                if gravado is None:
                    novos += 1
                # Alerta ainda não tratado é atualizado (Evidência nova); tratado fica como está.
                if gravado is None or gravado.status is Status.NOVO:
                    db.salvar(conn, alerta)
            linha = avaliar(clipe, resultado)
            linhas.append(linha)
            print(
                f"{clipe['id']:20} desvios {linha['desvios_detectados']}/"
                f"{linha['desvios_esperados']}  quedas {linha['quedas_detectadas']}/"
                f"{linha['quedas_esperadas']}  {linha['cpu_s']:5.1f} s CPU  -> {paciente_id}"
            )

    tabela = pd.DataFrame(linhas)
    SAIDA_PADRAO.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA_PADRAO / "avaliacao.csv", index=False)
    print(f"\n{novos} Alerta(s) novo(s) no banco; avaliação em {SAIDA_PADRAO / 'avaliacao.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
