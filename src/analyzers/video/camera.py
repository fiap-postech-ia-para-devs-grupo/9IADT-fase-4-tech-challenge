"""Captura ao vivo da webcam com o esqueleto em tempo real (Laboratório de Análise).

A câmera é aberta pelo OpenCV no computador onde o Streamlit roda: serve para a demonstração
local (app e câmera na mesma máquina), não para um navegador remoto. Durante a captura, cada
quadro com pose vai para `ao_vivo` já anotado (esqueleto + ângulos). Ao fim, o clipe gravado
passa pelo mesmo `analyze_video` de um upload.

    caminho = capturar(Path("sessao.avi"), segundos=10, ao_vivo=lambda quadro, t: ...)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.analyzers.video import angles
from src.analyzers.video import pose as p

FPS_GRAVACAO = 15.0
# Pose a cada N quadros ao vivo: em CPU, o YOLOv8n-pose não acompanha 30 fps.
PASSO_AO_VIVO = 2


class CameraIndisponivel(RuntimeError):
    pass


def _abrir(indice: int) -> cv2.VideoCapture:
    # No Windows, o DirectShow abre a webcam em menos de 1 s; o backend padrão demora ~10 s.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    captura = cv2.VideoCapture(indice, backend)
    if not captura.isOpened():
        raise CameraIndisponivel(
            f"Câmera {indice} indisponível: confira se existe, se outro app a está usando e "
            "se o Streamlit roda no mesmo computador da câmera."
        )
    return captura


def legenda(keypoints: np.ndarray | None) -> str:
    medidas = angles.angulos(keypoints)
    partes = []
    joelhos = [v for v in (medidas["joelho_esq"], medidas["joelho_dir"]) if v == v]
    if joelhos:
        partes.append(f"joelho {min(joelhos):.0f}")
    if medidas["tronco"] == medidas["tronco"]:
        partes.append(f"tronco {medidas['tronco']:.0f}")
    return "  ".join(partes) or "sem pessoa no quadro"


def capturar(
    destino: Path,
    *,
    segundos: float = 10.0,
    indice: int = 0,
    modelo: Any = None,
    ao_vivo: Callable[[np.ndarray, float], None] | None = None,
) -> Path:
    """Grava `segundos` da webcam em `destino` (AVI/MJPG) e mostra a pose ao vivo."""
    modelo = modelo or p.carregar_modelo()
    captura = _abrir(indice)
    escritor = None
    ultima = p.Pose(0, 0.0, None, None)
    inicio = time.perf_counter()
    quadro_n = 0
    try:
        while (decorrido := time.perf_counter() - inicio) < segundos:
            ok, quadro = captura.read()
            if not ok:
                raise CameraIndisponivel(f"A câmera {indice} parou de enviar imagem.")
            if escritor is None:
                altura, largura = quadro.shape[:2]
                destino.parent.mkdir(parents=True, exist_ok=True)
                escritor = cv2.VideoWriter(
                    str(destino), cv2.VideoWriter.fourcc(*"MJPG"), FPS_GRAVACAO, (largura, altura)
                )
            escritor.write(quadro)
            if quadro_n % PASSO_AO_VIVO == 0:
                resultado = modelo(quadro, verbose=False, device="cpu")[0]
                keypoints, caixa = p.pessoa_principal(resultado)
                ultima = p.Pose(quadro_n, decorrido, keypoints, caixa)
            if ao_vivo is not None:
                anotado = p.anotar(quadro, ultima)
                texto = f"{segundos - decorrido:4.1f}s  {legenda(ultima.keypoints)}"
                cv2.putText(
                    anotado, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                ao_vivo(anotado, decorrido)
            quadro_n += 1
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
    duracao = time.perf_counter() - inicio
    if quadro_n and duracao > 0:
        _corrigir_fps(destino, quadro_n / duracao)
    return destino


def _corrigir_fps(caminho: Path, fps: float) -> None:
    """A pose ao vivo atrasa a leitura da câmera, então o clipe sai com menos quadros por
    segundo que `FPS_GRAVACAO`. Regrava com o fps medido para os tempos (t da Evidência,
    duração da transição de Queda) baterem com o relógio."""
    if abs(fps - FPS_GRAVACAO) < 0.5:
        return
    temporario = caminho.with_name(caminho.stem + "_fps" + caminho.suffix)
    escritor = None
    for _, quadro in p.frames(caminho):
        if escritor is None:
            altura, largura = quadro.shape[:2]
            escritor = cv2.VideoWriter(
                str(temporario), cv2.VideoWriter.fourcc(*"MJPG"), fps, (largura, altura)
            )
        escritor.write(quadro)
    if escritor is not None:
        escritor.release()
        temporario.replace(caminho)
