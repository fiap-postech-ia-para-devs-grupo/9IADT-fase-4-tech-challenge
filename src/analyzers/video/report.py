"""Relatório de Sessão: um documento por vídeo analisado (ESTRATEGIA.md, seção 4; CONTEXT.md).

Lista os Desvios de Execução (e Quedas) com a Evidência: o frame anotado com o esqueleto e o
gráfico do ângulo ao longo do tempo, com as repetições e os limiares marcados.

    caminho = report.gerar(resultado, pasta)  # pasta/relatorio.md + pasta/angulos.png
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # sem janela: roda em servidor, teste e Streamlit
import matplotlib.pyplot as plt  # noqa: E402

from src.analyzers.video.exercise_rules import (  # noqa: E402
    DESCRICAO_REGRA,
    joelho_mais_flexionado,
)

if TYPE_CHECKING:
    from src.analyzers.video.lab import ResultadoVideo

NOME_RELATORIO = "relatorio.md"
NOME_GRAFICO = "angulos.png"


def grafico_angulos(resultado: ResultadoVideo, destino: Path) -> Path:
    serie = resultado.serie
    figura, eixo = plt.subplots(figsize=(10, 3.6), dpi=110)
    if not serie.empty:
        eixo.plot(serie.index, joelho_mais_flexionado(serie), label="joelho (mais flexionado)")
        eixo.plot(serie.index, serie["tronco"], label="inclinação do tronco", alpha=0.7)
        if "valgo" in serie:
            eixo.plot(serie.index, serie["valgo"], label="valgo do joelho de apoio (FPPA)")
    exercicio = resultado.exercicio
    if exercicio is not None and exercicio.valgo_max is not None and "valgo" in serie:
        eixo.axhline(exercicio.valgo_max, color="tab:red", ls="--", lw=1, label="valgo máx.")
    if exercicio is not None and exercicio.joelho_alvo is not None:
        eixo.axhline(
            exercicio.joelho_alvo, color="tab:green", ls="--", lw=1, label="alvo do joelho"
        )
    if exercicio is not None and exercicio.tronco_max is not None:
        eixo.axhline(exercicio.tronco_max, color="tab:orange", ls=":", lw=1, label="tronco máx.")
    for rep in resultado.repeticoes:
        eixo.axvspan(rep.inicio, rep.fim, color="tab:blue", alpha=0.08)
    for desvio in resultado.desvios:
        eixo.axvline(desvio.t, color="tab:red", lw=1.2)
    for queda in resultado.quedas:
        eixo.axvline(queda.t, color="black", lw=2)
    eixo.set_xlabel("tempo do clipe (s)")
    eixo.set_ylabel("graus")
    eixo.set_ylim(-20 if "valgo" in serie else 0, 185)
    eixo.legend(loc="lower right", fontsize=8)
    eixo.set_title("Ângulos ao longo da sessão (vermelho: Desvio; preto: Queda)", fontsize=10)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino


def _tabela(linhas: list[list[str]], cabecalho: list[str]) -> str:
    texto = ["| " + " | ".join(cabecalho) + " |", "|" + " --- |" * len(cabecalho)]
    texto += ["| " + " | ".join(linha) + " |" for linha in linhas]
    return "\n".join(texto)


def markdown(resultado: ResultadoVideo, pasta: Path) -> str:
    """O Relatório de Sessão em Markdown; imagens com caminho relativo a `pasta`."""
    seq = resultado.sequencia
    exercicio = resultado.exercicio
    partes = [
        f"# Relatório de Sessão — {resultado.clipe_id}",
        "",
        f"- **Arquivo:** `{resultado.arquivo}`",
        f"- **Paciente:** {resultado.paciente_id}",
        f"- **Cenário:** {exercicio.rotulo if exercicio else 'quarto (monitoramento de Queda)'}",
        f"- **Vista da câmera:** {resultado.vista}",
        f"- **Duração:** {seq.duracao:.1f} s ({seq.total_frames} frames a {seq.fps:.0f} fps; "
        f"pose em 1 a cada {seq.passo})",
        f"- **Processamento em CPU:** {seq.segundos_processamento:.1f} s "
        f"({resultado.fator_tempo_real:.2f}× o tempo do clipe)",
        f"- **Frames com pessoa detectada:** {resultado.cobertura:.0%}",
        "",
    ]
    if exercicio is not None:
        partes += [
            f"## Repetições ({len(resultado.repeticoes)})",
            "",
        ]
        if resultado.repeticoes:
            partes.append(
                _tabela(
                    [
                        [str(r.numero), f"{r.inicio:.1f}–{r.fim:.1f} s", f"{r.joelho_fundo:.0f}°"]
                        for r in resultado.repeticoes
                    ],
                    ["#", "trecho", "joelho no fundo"],
                )
            )
        else:
            partes.append(
                "Sem repetições segmentadas pelo joelho (exercício avaliado por episódio)."
            )
        partes += ["", f"## Desvios de Execução ({len(resultado.desvios)})", ""]
        if resultado.desvios:
            partes.append(
                _tabela(
                    [
                        [
                            DESCRICAO_REGRA[d.regra],
                            str(d.repeticao or "—"),
                            f"{d.t:.1f} s",
                            f"{d.valor:.0f}°",
                            f"≤ {d.limite:.0f}°",
                        ]
                        for d in resultado.desvios
                    ],
                    ["desvio", "rep./episódio", "instante", "medido", "esperado"],
                )
            )
        else:
            partes.append("Nenhum Desvio de Execução: execução dentro da faixa esperada.")
        partes.append("")

    partes += [f"## Quedas ({len(resultado.quedas)})", ""]
    if resultado.quedas:
        partes.append(
            _tabela(
                [
                    [
                        f"{q.t:.1f} s",
                        f"{q.duracao_transicao:.1f} s",
                        f"{q.razao_caixa:.2f}",
                        f"{q.altura_relativa:.0%}",
                    ]
                    for q in resultado.quedas
                ],
                ["no chão em", "transição", "largura/altura", "altura vs. em pé"],
            )
        )
    else:
        partes.append("Nenhuma Queda detectada.")
    partes.append("")

    if resultado.evidencias:
        partes += ["## Evidências", ""]
        for evidencia in resultado.evidencias:
            relativo = Path(evidencia.caminho).relative_to(pasta).as_posix()
            partes += [f"**{evidencia.titulo}**", "", f"![{evidencia.titulo}]({relativo})", ""]

    partes += [
        "## Ângulos ao longo do tempo",
        "",
        f"![ângulos]({NOME_GRAFICO})",
        "",
        "_Gerado automaticamente pelo Analisador de Vídeo (YOLOv8n-pose). Limiares em "
        "`src/analyzers/video/exercise_rules.py` e `fall.py`._",
        "",
    ]
    return "\n".join(partes)


def gerar(resultado: ResultadoVideo, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    grafico_angulos(resultado, pasta / NOME_GRAFICO)
    destino = pasta / NOME_RELATORIO
    destino.write_text(markdown(resultado, pasta), encoding="utf-8")
    return destino
