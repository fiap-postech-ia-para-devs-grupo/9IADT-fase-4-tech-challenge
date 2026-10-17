import pathlib

from src.analyzers.vitals import load

RAIZ = pathlib.Path("data/vitals/mimic3wdb")
JANELA = 40
SERIES = ["FC", "FR", "SpO2", "PAS"]


def janelas_limpas(d):
    ok = d[SERIES].notna().all(axis=1).to_numpy()
    count, i = 0, 0
    while i <= len(ok) - JANELA:
        if ok[i : i + JANELA].all():
            count += 1
            i += JANELA
        else:
            i += 1
    return count


selecionados = []
for hea in sorted(RAIZ.rglob("*n.hea")):
    caminho = hea.with_suffix("")
    rel = caminho.relative_to(RAIZ).as_posix()
    try:
        d = load.carregar(caminho)
    except Exception as erro:
        print(rel, "ERRO", erro, flush=True)
        continue
    lacuna = d[SERIES].isna().mean().max()
    janelas = janelas_limpas(d)
    print(rel, len(d), round(lacuna, 3), janelas, flush=True)
    if lacuna <= 0.5 and janelas >= 3:
        selecionados.append(rel)
pathlib.Path("data/vitals/selecionados_v2.txt").write_text("\n".join(selecionados) + "\n")
print("selecionados_v2", len(selecionados), flush=True)
