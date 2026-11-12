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
import itertools
import shutil
import string
import sys
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
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


# Wikimedia pede um User-Agent que identifique o projeto; sem ele, devolve 403/429.
USER_AGENT = (
    "FIAP-TechChallenge-Fase4/0.1 "
    "(https://github.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-4-tech-challenge)"
)


@dataset("video", dono="Marcelo", destino="video")
def baixar_video(destino: Path) -> None:
    """Clipes de `data/video/clipes.json` (Wikimedia Commons e UR Fall Detection Dataset).

    Licença, autor e página de cada um em `data/video/SOURCES.md`. Os pesos do YOLOv8n-pose
    são baixados pelo Ultralytics na primeira análise.
    """
    import json

    manifesto = json.loads((RAIZ_DADOS / "video" / "clipes.json").read_text(encoding="utf-8"))
    for clipe in manifesto["clipes"]:
        alvo = destino / clipe["arquivo"]
        if alvo.exists():
            continue
        parcial = alvo.with_suffix(alvo.suffix + ".part")
        pedido = urllib.request.Request(clipe["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(pedido, timeout=120) as resposta, open(parcial, "wb") as saida:
            shutil.copyfileobj(resposta, saida)
        parcial.replace(alvo)


SONS = {
    "alarme_monitor.mp3": "https://upload.wikimedia.org/wikipedia/commons/e/e1/Heart_Monitor_Beep--freesound.org.mp3",
    "grito.ogg": "https://upload.wikimedia.org/wikipedia/commons/d/d9/Wilhelm_Scream.ogg",
    "gemido_choro.ogg": "https://upload.wikimedia.org/wikipedia/commons/e/e1/Male_crying_and_weeping.ogg",
}


@dataset("sons", dono="Marcelo", destino="audio/sons")
def baixar_sons(destino: Path) -> None:
    """Sons do leito para o monitoramento de áudio; licenças em data/audio/sons/SOURCES.md."""
    for arquivo, url in SONS.items():
        alvo = destino / arquivo
        if alvo.exists():
            continue
        pedido = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(pedido, timeout=120) as resposta, open(alvo, "wb") as saida:
            shutil.copyfileobj(resposta, saida)


PRIMOCK57_AUDIO_URL = "https://media.githubusercontent.com/media/babylonhealth/primock57/main/audio"
PRIMOCK57_TRANSCRIPT_URL = "https://raw.githubusercontent.com/babylonhealth/primock57/main/transcripts"


@dataset("primock57", dono="V. Geizler", destino="audio/primock57")
def baixar_primock57(destino: Path) -> None:
    """Canal do paciente das Consultas de `data/audio/primock57_selecionadas.txt`.

    Áudio (LFS, servido sem precisar do cliente git-lfs via media.githubusercontent.com) +
    transcrição manual (.TextGrid, texto puro). O canal do médico não é baixado: os
    Analisadores de Áudio (ESTRATEGIA.md, seção 5) só processam a fala do paciente.
    """
    ids = _ler_lista(RAIZ_DADOS / "audio" / "primock57_selecionadas.txt")
    pasta_audio = destino / "audio"
    pasta_transcricoes = destino / "transcripts"
    pasta_audio.mkdir(parents=True, exist_ok=True)
    pasta_transcricoes.mkdir(parents=True, exist_ok=True)

    for consulta_id in ids:
        audio = pasta_audio / f"{consulta_id}_patient.wav"
        if not audio.exists():
            urllib.request.urlretrieve(f"{PRIMOCK57_AUDIO_URL}/{consulta_id}_patient.wav", audio)
        transcricao = pasta_transcricoes / f"{consulta_id}_patient.TextGrid"
        if not transcricao.exists():
            urllib.request.urlretrieve(
                f"{PRIMOCK57_TRANSCRIPT_URL}/{consulta_id}_patient.TextGrid", transcricao
            )


COSWARA_RAW_URL = "https://raw.githubusercontent.com/iiscleap/Coswara-Data/master"
# Dentro de cada pessoa só interessam estas duas gravações (ESTRATEGIA.md, seção 5).
COSWARA_ARQUIVOS_UTEIS = ("counting-normal.wav", "vowel-a.wav")


def _sufixos_split() -> Iterator[str]:
    """aa, ab, ..., az, ba, ... — mesma convenção do `split -a 2` usado para gerar as partes."""
    letras = string.ascii_lowercase
    return (a + b for a, b in itertools.product(letras, letras))


def _existe_na_url(url: str) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30)
    except urllib.error.HTTPError as erro:
        if erro.code == 404:
            return False
        raise
    else:
        return True


def _baixar_partes_pasta(pasta: str, destino: Path) -> list[Path]:
    """Baixa `<pasta>.tar.gz.aa`, `.ab`, ... até a primeira que não existir; pula as já baixadas."""
    destino.mkdir(parents=True, exist_ok=True)
    partes: list[Path] = []
    for sufixo in _sufixos_split():
        nome = f"{pasta}.tar.gz.{sufixo}"
        arquivo = destino / nome
        if arquivo.exists():
            partes.append(arquivo)
            continue
        url = f"{COSWARA_RAW_URL}/{pasta}/{nome}"
        if not _existe_na_url(url):
            break
        parcial = arquivo.with_suffix(arquivo.suffix + ".part")
        urllib.request.urlretrieve(url, parcial)
        parcial.replace(arquivo)
        partes.append(arquivo)
    if not partes:
        raise RuntimeError(f"nenhuma parte .tar.gz encontrada para a pasta {pasta}")
    return partes


def _extrair_arquivos_uteis(partes: list[Path], pasta: str, extraido: Path) -> None:
    """Concatena as partes (são um .tar.gz só, dividido em bytes) num arquivo temporário e
    extrai dele só os 2 áudios úteis de cada pessoa; os outros 7 áudios não tocam o disco."""
    concatenado = partes[0].parent / f"{pasta}.tar.gz"
    with open(concatenado, "wb") as saida:
        for parte in sorted(partes, key=lambda p: p.suffix):
            with open(parte, "rb") as entrada:
                shutil.copyfileobj(entrada, saida)

    try:
        with tarfile.open(concatenado, mode="r:gz") as tar:
            for membro in tar:
                if membro.isfile() and membro.name.endswith(COSWARA_ARQUIVOS_UTEIS):
                    relativo = Path(membro.name).relative_to(pasta)
                    alvo = extraido / pasta / relativo
                    alvo.parent.mkdir(parents=True, exist_ok=True)
                    origem = tar.extractfile(membro)
                    if origem is None:
                        continue
                    with origem, open(alvo, "wb") as arquivo:
                        shutil.copyfileobj(origem, arquivo)
    finally:
        concatenado.unlink(missing_ok=True)


@dataset("coswara", dono="V. Geizler", destino="audio/coswara")
def baixar_coswara(destino: Path) -> None:
    """Pastas de `data/audio/coswara_pastas_selecionadas.txt` (rationale no próprio arquivo):
    `combined_data.csv` (rótulos `bd`/`ftg`/`covid_status`) + `counting-normal.wav` e
    `vowel-a.wav` de cada pessoa, em `destino/Extracted_data/<pasta>/<id>/`.

    Cada pasta é um único .tar.gz dividido em partes de ~100 MB (convenção do próprio
    Coswara-Data); as partes brutas não ficam em disco depois de extraídas. `acoustic.py`
    decide o subset balanceado por classe a partir do `combined_data.csv`.
    """
    combinado = destino / "combined_data.csv"
    if not combinado.exists():
        urllib.request.urlretrieve(f"{COSWARA_RAW_URL}/combined_data.csv", combinado)

    pastas = _ler_lista(RAIZ_DADOS / "audio" / "coswara_pastas_selecionadas.txt")
    extraido = destino / "Extracted_data"
    extraido.mkdir(parents=True, exist_ok=True)
    brutos = destino / "_tmp_partes"

    for pasta in pastas:
        if (extraido / pasta).exists():
            continue
        partes = _baixar_partes_pasta(pasta, brutos / pasta)
        _extrair_arquivos_uteis(partes, pasta, extraido)
        shutil.rmtree(brutos / pasta)
    if brutos.exists() and not any(brutos.iterdir()):
        brutos.rmdir()


def _ler_lista(caminho: Path) -> list[str]:
    return [
        linha.strip()
        for linha in caminho.read_text().splitlines()
        if linha.strip() and not linha.startswith("#")
    ]


MIMIC3WDB_URL = "https://physionet.org/files/mimic3wdb/1.0/"


@dataset("mimic3wdb", dono="Antonio", destino="vitals/mimic3wdb")
def baixar_mimic3wdb(destino: Path) -> None:
    """Registros numerics listados em `data/vitals/selecionados.txt` (um caminho por linha).

    Baixa o .hea e o .dat de cada um para `destino/<pasta>/`; arquivos já existentes são pulados.
    A lista é escolhida por `scripts/selecionar_registros_vitals.py`.
    """
    import urllib.request

    lista = RAIZ_DADOS / "vitals" / "selecionados.txt"
    registros = [linha.strip() for linha in lista.read_text().splitlines() if linha.strip()]
    for registro in registros:
        for extensao in ("hea", "dat"):
            arquivo = destino / f"{registro}.{extensao}"
            if arquivo.exists():
                continue
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{MIMIC3WDB_URL}{registro}.{extensao}", arquivo)


MIMIC4_DEMO_URL = "https://physionet.org/files/mimic-iv-demo/2.2"
# prescriptions e emar são o insumo do Analisador; patients e admissions dão idade/sexo e o
# intervalo de cada internação (usado quando uma prescrição não tem stoptime).
MIMIC4_DEMO_ARQUIVOS = (
    "LICENSE.txt",
    "hosp/prescriptions.csv.gz",
    "hosp/emar.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
)


@dataset("mimic4demo", dono="V. Blasque", destino="prescriptions/mimic4demo")
def baixar_mimic4demo(destino: Path) -> None:
    """MIMIC-IV Clinical Database Demo (ODbL, acesso aberto): hosp/prescriptions e hosp/emar."""
    for arquivo in MIMIC4_DEMO_ARQUIVOS:
        alvo = destino / arquivo
        if alvo.exists():
            continue
        alvo.parent.mkdir(parents=True, exist_ok=True)
        parcial = alvo.with_suffix(alvo.suffix + ".part")
        urllib.request.urlretrieve(f"{MIMIC4_DEMO_URL}/{arquivo}", parcial)
        parcial.replace(alvo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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
            print(f"[ok]       {nome} -> {destino.relative_to(RAIZ_DADOS.parent)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
