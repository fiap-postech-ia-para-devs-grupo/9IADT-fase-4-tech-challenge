"""Baixa os datasets públicos para `data/` (que não vai para o git).

Cada dono pluga o download do seu dataset registrando uma função com `@dataset`:

    @dataset("mimic3wdb", dono="Antonio", destino="vitals/mimic3wdb")
    def baixar_mimic3wdb(destino: Path) -> None:
        wfdb.dl_database("mimic3wdb", destino, records=[...])

A função recebe a pasta de destino já criada e deve ser idempotente (pular o que já existe).

    uv run python scripts/download_data.py                      # todos
    uv run python scripts/download_data.py mimic3wdb coswara    # só alguns
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

RAIZ_DADOS = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Dataset:
    nome: str
    dono: str
    destino: str
    baixar: Callable[[Path], None]


DATASETS: dict[str, Dataset] = {}


def dataset(nome: str, *, dono: str, destino: str):
    def registrar(baixar: Callable[[Path], None]) -> Callable[[Path], None]:
        DATASETS[nome] = Dataset(nome, dono, destino, baixar)
        return baixar

    return registrar


# --- Datasets (ESTRATEGIA.md, seções 4–7). Cada dono troca o `raise` pelo download real. ---


@dataset("video", dono="Marcelo", destino="video")
def baixar_video(destino: Path) -> None:
    """Clipes Pexels/Pixabay; licença/URL de cada um em data/video/SOURCES.md."""
    raise NotImplementedError


@dataset("primock57", dono="V. Geizler", destino="audio/primock57")
def baixar_primock57(destino: Path) -> None:
    """Consultas simuladas + transcrição manual (Git LFS)."""
    raise NotImplementedError


@dataset("coswara", dono="V. Geizler", destino="audio/coswara")
def baixar_coswara(destino: Path) -> None:
    """Subset counting-normal + vowel-a, breathing difficulty/fatigue × saudáveis."""
    raise NotImplementedError


@dataset("mimic3wdb", dono="Antonio", destino="vitals/mimic3wdb")
def baixar_mimic3wdb(destino: Path) -> None:
    """~10–20 registros numerics (FC, FR, SpO2, PA) via wfdb."""
    raise NotImplementedError


@dataset("mimic4demo", dono="V. Blasque", destino="prescriptions/mimic4demo")
def baixar_mimic4demo(destino: Path) -> None:
    """MIMIC-IV Clinical Database Demo: hosp/prescriptions e hosp/emar."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "nomes",
        nargs="*",
        metavar="dataset",
        help=f"um ou mais de: {', '.join(DATASETS)} (padrão: todos)",
    )
    nomes = parser.parse_args(argv).nomes or list(DATASETS)
    if desconhecidos := sorted(set(nomes) - set(DATASETS)):
        parser.error(f"dataset desconhecido: {', '.join(desconhecidos)}")

    falhas = 0
    for nome in nomes:
        ds = DATASETS[nome]
        destino = RAIZ_DADOS / ds.destino
        destino.mkdir(parents=True, exist_ok=True)
        try:
            ds.baixar(destino)
        except NotImplementedError:
            print(f"[pendente] {nome}: download ainda não implementado ({ds.dono})")
        except Exception as erro:  # um dataset quebrado não impede os outros
            falhas += 1
            print(f"[erro]     {nome}: {erro}", file=sys.stderr)
        else:
            print(f"[ok]       {nome} → {destino.relative_to(RAIZ_DADOS.parent)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
