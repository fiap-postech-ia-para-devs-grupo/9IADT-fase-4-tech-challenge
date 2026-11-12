"""Objetos e áreas críticas no quadro com YOLOv8n (detecção COCO), para contextualizar a Queda.

O PDF pede "YOLOv8 para detecção de objetos e áreas críticas". No quarto, as áreas críticas são
os pontos de transferência onde o paciente se levanta ou senta: cama, cadeira, sofá e vaso
sanitário. A detecção roda só no frame de Evidência da Queda (um frame por Queda, custo
desprezível em CPU) e diz perto de quais desses objetos o paciente caiu.

    objetos = detectar(frame)
    perto = proximos(objetos, caixa_da_pessoa)   # ["cadeira"]
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_RAIZ = Path(__file__).resolve().parents[3]
MODELO_PADRAO = _RAIZ / "data" / "video" / "yolov8n.pt"

# Classes COCO das áreas críticas de transferência, com o nome em português.
AREAS_CRITICAS = {
    "bed": "cama",
    "chair": "cadeira",
    "couch": "sofá",
    "toilet": "vaso sanitário",
}
CONFIANCA_MINIMA = 0.35
# Distância máxima entre a caixa da pessoa e a do objeto, em larguras da pessoa.
PROXIMIDADE = 0.5


@dataclass(frozen=True)
class Objeto:
    nome: str  # em português
    confianca: float
    caixa: tuple[float, float, float, float]


def carregar_modelo(caminho: Path = MODELO_PADRAO) -> Any:
    from ultralytics import YOLO

    caminho.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(caminho))


def detectar(frame: np.ndarray, modelo: Any = None) -> list[Objeto]:
    modelo = modelo or carregar_modelo()
    resultado = modelo(frame, verbose=False, device="cpu")[0]
    objetos = []
    for classe, confianca, caixa in zip(
        resultado.boxes.cls.cpu().numpy(),
        resultado.boxes.conf.cpu().numpy(),
        resultado.boxes.xyxy.cpu().numpy(),
        strict=True,
    ):
        nome = AREAS_CRITICAS.get(resultado.names[int(classe)])
        if nome and confianca >= CONFIANCA_MINIMA:
            objetos.append(Objeto(nome, float(confianca), tuple(float(v) for v in caixa)))
    return objetos


def _distancia(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Distância entre duas caixas (0 se se sobrepõem)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return float(np.hypot(dx, dy))


def proximos(objetos: list[Objeto], pessoa: tuple[float, float, float, float]) -> list[str]:
    """Nomes (sem repetição) dos objetos de área crítica junto à pessoa."""
    limite = PROXIMIDADE * (pessoa[2] - pessoa[0])
    perto = [o for o in objetos if _distancia(o.caixa, pessoa) <= limite]
    return list(dict.fromkeys(o.nome for o in sorted(perto, key=lambda o: -o.confianca)))


def anotar(frame: np.ndarray, objetos: list[Objeto]) -> np.ndarray:
    saida = frame.copy()
    espessura = max(1, round(max(saida.shape[:2]) / 400))
    for objeto in objetos:
        x0, y0, x1, y1 = (round(v) for v in objeto.caixa)
        cv2.rectangle(saida, (x0, y0), (x1, y1), (0, 165, 255), espessura, cv2.LINE_AA)
        cv2.putText(
            saida,
            unicodedata.normalize("NFKD", f"{objeto.nome} {objeto.confianca:.0%}")
            .encode("ascii", "ignore")
            .decode(),
            (x0 + 3, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * espessura + 0.1,
            (0, 165, 255),
            espessura,
            cv2.LINE_AA,
        )
    return saida
