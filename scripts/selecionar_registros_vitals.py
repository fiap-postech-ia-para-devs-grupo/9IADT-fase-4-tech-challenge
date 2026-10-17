import pathlib
import urllib.request

from src.analyzers.vitals import load

URL = "https://physionet.org/files/mimic3wdb/1.0/"
RAIZ = pathlib.Path("data/vitals/mimic3wdb")
bons = []
for p in pathlib.Path("data/vitals/candidatos2.txt").read_text().split():
    for e in ("hea", "dat"):
        destino = RAIZ / f"{p}.{e}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists():
            urllib.request.urlretrieve(URL + f"{p}.{e}", destino)
    try:
        d = load.carregar(RAIZ / p)
    except Exception as erro:
        print(p, "ERRO", erro, flush=True)
        continue
    lacuna = d[["FC", "FR", "SpO2", "PAS"]].isna().mean().max()
    print(p, len(d), round(lacuna, 3), flush=True)
    if lacuna <= 0.5:
        bons.append(p)
pathlib.Path("data/vitals/selecionados.txt").write_text("\n".join(bons) + "\n")
print("selecionados", len(bons), flush=True)
