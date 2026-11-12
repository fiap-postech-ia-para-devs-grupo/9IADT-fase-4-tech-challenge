"""`analyze_video`: o Analisador de Vídeo de ponta a ponta, chamável pela Tela 3 — Laboratório
de Análise — e pelo processamento em lote (ESTRATEGIA.md, seções 4 e 9).

Entrada: um arquivo de vídeo que o OpenCV leia (MP4, WebM, AVI, OGV...), o exercício da Sessão
de Fisioterapia (`None` para vídeo de quarto, só Queda) e a vista da câmera.
Saída: `ResultadoVideo` com a série de ângulos, as repetições, os Desvios de Execução, as
Quedas, os frames de Evidência anotados, o Relatório de Sessão em Markdown e os Alertas já no
Contrato, prontos para `db.salvar`.

    from src.analyzers.video.lab import analyze_video
    resultado = analyze_video(Path("sessao.mp4"), exercicio="agachamento", vista="frontal")
    resultado.relatorio          # results/video/<clipe>/relatorio.md
    for alerta in resultado.alertas:
        db.salvar(conn, alerta)

A Queda é procurada em todo vídeo, inclusive nas sessões: um exercício que não gera Queda é o
teste de falso positivo do detector.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.analyzers.video import angles, fall, objetos, report
from src.analyzers.video import pose as p
from src.analyzers.video.exercise_rules import (
    DESCRICAO_REGRA,
    EXERCICIOS,
    Desvio,
    Exercicio,
    Regra,
    Repeticao,
    Vista,
    avaliar,
    serie_valgo,
)
from src.analyzers.video.fall import Queda
from src.analyzers.video.pose import Recorte, SequenciaPose
from src.core.alerts import Alerta, Origem, Severidade, Tipo

_RAIZ = Path(__file__).resolve().parents[3]
SAIDA_PADRAO = _RAIZ / "results" / "video"
_NAMESPACE = uuid.UUID("0b6f3c52-8e1d-4a77-b2c9-4d5e6f7a8b90")

# Queda de paciente internado é tratada na hora; Desvio de Execução vai para o fisioterapeuta.
SEVERIDADE_QUEDA = Severidade.ALTA
SEVERIDADE_DESVIO = Severidade.MEDIA

# Articulações destacadas no frame de Evidência de cada regra.
_DESTAQUE = {
    Regra.AMPLITUDE_INSUFICIENTE: (p.JOELHO_ESQ, p.JOELHO_DIR),
    Regra.COMPENSACAO_TRONCO: (p.OMBRO_ESQ, p.OMBRO_DIR, p.QUADRIL_ESQ, p.QUADRIL_DIR),
    Regra.VALGO_DINAMICO: (p.JOELHO_ESQ, p.JOELHO_DIR),
    Regra.ASSIMETRIA: (p.JOELHO_ESQ, p.JOELHO_DIR),
}


@dataclass(frozen=True)
class EvidenciaFrame:
    titulo: str
    caminho: str  # absoluto
    t: float


@dataclass(frozen=True)
class ResultadoVideo:
    clipe_id: str
    arquivo: str
    paciente_id: str
    exercicio: Exercicio | None
    vista: str
    sequencia: SequenciaPose
    serie: pd.DataFrame
    repeticoes: list[Repeticao]
    desvios: list[Desvio]
    quedas: list[Queda]
    evidencias: list[EvidenciaFrame]
    relatorio: Path
    alertas: list[Alerta]
    video: Path | None = None  # clipe com o esqueleto sobreposto (WebM), se pedido

    @property
    def fator_tempo_real(self) -> float:
        """Segundos de CPU por segundo de vídeo: abaixo de 1, dá para analisar ao vivo."""
        duracao = self.sequencia.duracao
        return self.sequencia.segundos_processamento / duracao if duracao else 0.0

    @property
    def cobertura(self) -> float:
        poses = self.sequencia.poses
        return sum(q.keypoints is not None for q in poses) / len(poses) if poses else 0.0


def identificador(arquivo: Path) -> str:
    """Nome + hash do conteúdo: dois uploads "video.mp4" diferentes não se misturam."""
    resumo = hashlib.sha256(arquivo.read_bytes()).hexdigest()[:10]
    return f"{arquivo.stem}-{resumo}"


def _relativo(caminho: Path) -> str:
    """Caminho da Evidência como a Central de Alertas espera: relativo à raiz, se possível."""
    try:
        return caminho.resolve().relative_to(_RAIZ).as_posix()
    except ValueError:
        return str(caminho)


def _ascii(texto: str) -> str:
    """A fonte Hershey do OpenCV não tem acentos: "dinâmico" sairia "din??mico"."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def _escrever(frame, texto: str, linha: int = 1, cor=(0, 0, 255)) -> None:
    escala = max(1.0, frame.shape[1] / 800)
    cv2.putText(
        frame,
        _ascii(texto),
        (10, round(30 * escala * linha)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * escala,
        cor,
        max(2, round(2 * escala)),
        cv2.LINE_AA,
    )


def _salvar_frame(
    arquivo: Path,
    sequencia: SequenciaPose,
    indice: int,
    destino: Path,
    rotulo: str,
    destaque: tuple[int, ...] = (),
    objetos_no_quadro: list[objetos.Objeto] | None = None,
) -> Path:
    pose = next(q for q in sequencia.poses if q.frame == indice)
    frame = p.ler_frame(arquivo, indice, sequencia.recorte)
    if objetos_no_quadro:
        frame = objetos.anotar(frame, objetos_no_quadro)
    frame = p.anotar(frame, pose, destaque=destaque)
    _escrever(frame, rotulo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), frame)
    return destino


def gerar_video_anotado(arquivo: Path, sequencia: SequenciaPose, destino: Path) -> Path | None:
    """Clipe com o esqueleto e os ângulos sobrepostos, só com os frames analisados (WebM/VP8,
    que o navegador toca). Devolve None se o OpenCV não tiver o codec."""
    poses = {pose.frame: pose for pose in sequencia.poses}
    fps = max(sequencia.fps / sequencia.passo, 1.0)
    escritor = None
    try:
        for indice, frame in p.frames(arquivo, sequencia.recorte):
            pose = poses.get(indice)
            if pose is None:
                continue
            quadro = p.anotar(frame, pose)
            medidas = angles.angulos(pose.keypoints)
            joelhos = [v for v in (medidas["joelho_esq"], medidas["joelho_dir"]) if v == v]
            linha = f"t {pose.t:5.1f}s"
            if joelhos:  # v == v descarta NaN (articulação oculta)
                linha += f"  joelho {min(joelhos):.0f}"
            if medidas["tronco"] == medidas["tronco"]:
                linha += f"  tronco {medidas['tronco']:.0f}"
            _escrever(quadro, linha, cor=(255, 255, 255))
            if escritor is None:
                altura, largura = quadro.shape[:2]
                destino.parent.mkdir(parents=True, exist_ok=True)
                escritor = cv2.VideoWriter(
                    str(destino), cv2.VideoWriter.fourcc(*"VP80"), fps, (largura, altura)
                )
                if not escritor.isOpened():
                    return None
            escritor.write(quadro)
    finally:
        if escritor is not None:
            escritor.release()
    return destino if escritor is not None and destino.exists() else None


def _id_alerta(paciente_id: str, clipe_id: str, chave: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{paciente_id}|{clipe_id}|{chave}"))


def alerta_desvio(
    desvio: Desvio,
    exercicio: Exercicio,
    paciente_id: str,
    clipe_id: str,
    inicio: datetime,
    frame: str,
) -> Alerta:
    evidencia: dict[str, Any] = {
        "frame": frame,
        "t": round(desvio.t, 2),
        "angulo": round(desvio.valor, 1),
        "limite": desvio.limite,
        "regra": str(desvio.regra),
        "exercicio": exercicio.nome,
        "clipe": clipe_id,
    }
    if desvio.repeticao is not None:
        evidencia["repeticao"] = desvio.repeticao
    if desvio.lado is not None:
        evidencia["lado"] = desvio.lado
    return Alerta(
        id=_id_alerta(paciente_id, clipe_id, f"{desvio.regra}|{desvio.t:.2f}"),
        paciente_id=paciente_id,
        origem=Origem.VIDEO,
        tipo=Tipo.DESVIO_EXECUCAO,
        severidade=SEVERIDADE_DESVIO,
        descricao=f"Desvio de Execução em {exercicio.rotulo.lower()}: {desvio.descricao}",
        evidencia=evidencia,
        detectado_em=inicio + timedelta(seconds=desvio.t),
    )


def alerta_queda(
    queda: Queda,
    paciente_id: str,
    clipe_id: str,
    inicio: datetime,
    frame: str,
    perto_de: list[str] | None = None,
) -> Alerta:
    evidencia: dict[str, Any] = {
        "frame": frame,
        "t": round(queda.t, 2),
        "transicao_s": round(queda.duracao_transicao, 2),
        "razao_caixa": round(queda.razao_caixa, 2),
        "altura_relativa": round(queda.altura_relativa, 2),
        "clipe": clipe_id,
    }
    descricao = queda.descricao
    if perto_de:
        evidencia["objetos_proximos"] = perto_de
        descricao += f", junto a {' e '.join(perto_de)}"
    return Alerta(
        id=_id_alerta(paciente_id, clipe_id, f"queda|{queda.t:.2f}"),
        paciente_id=paciente_id,
        origem=Origem.VIDEO,
        tipo=Tipo.QUEDA,
        severidade=SEVERIDADE_QUEDA,
        descricao=descricao,
        evidencia=evidencia,
        detectado_em=inicio + timedelta(seconds=queda.t),
    )


def analyze_video(
    arquivo: Path,
    *,
    exercicio: str | None = None,
    vista: str = Vista.LATERAL,
    paciente_id: str = "avulso",
    clipe_id: str | None = None,
    inicio: datetime | None = None,
    recorte: Recorte | None = None,
    passo: int = p.PASSO_PADRAO,
    saida: Path = SAIDA_PADRAO,
    modelo: Any = None,
    parametros: dict[str, float] | None = None,
    video_anotado: bool = False,
    detectar_objetos: bool = True,
    modelo_objetos: Any = None,
) -> ResultadoVideo:
    """Pose → ângulos → Desvios e Quedas → frames de Evidência → Relatório de Sessão → Alertas.

    `inicio` é o instante do primeiro frame (no Paciente de Demonstração, o horário da sessão
    na internação simulada); cada Alerta sai em `inicio + t`. Padrão: agora.
    `recorte` (x0, y0, x1, y1) analisa só parte do quadro, ex.: a metade RGB do UR Fall.
    `parametros` sobrescreve a faixa esperada do exercício (`joelho_alvo`, `tronco_max`,
    `valgo_max`, `assimetria_max`), como o fisioterapeuta prescreve para o paciente.
    `video_anotado` grava `anotado.webm` com o esqueleto; `detectar_objetos` roda o YOLOv8 de
    objetos no frame de cada Queda para dizer perto de qual área crítica ela aconteceu.
    """
    if exercicio is not None and exercicio not in EXERCICIOS:
        raise ValueError(f"exercício desconhecido: {exercicio} (opções: {', '.join(EXERCICIOS)})")
    vista = Vista(vista)
    clipe_id = clipe_id or identificador(arquivo)
    inicio = inicio or datetime.now().replace(microsecond=0)
    pasta = saida / clipe_id
    for antigo in [*pasta.glob("desvio_*.jpg"), *pasta.glob("queda_*.jpg")]:
        antigo.unlink()  # reprocessar com outro limiar não deixa Evidência órfã

    sequencia = p.estimar(arquivo, passo=passo, recorte=recorte, modelo=modelo)
    serie = angles.serie(sequencia)
    definicao = EXERCICIOS[exercicio] if exercicio else None
    if definicao is not None and parametros:
        definicao = replace(definicao, **parametros)
    repeticoes, desvios = avaliar(sequencia, serie, definicao, vista) if definicao else ([], [])
    if definicao is not None and definicao.valgo_max is not None and vista is Vista.FRONTAL:
        serie = serie.assign(valgo=serie_valgo(sequencia))
    quedas = fall.detectar(sequencia)

    evidencias: list[EvidenciaFrame] = []
    alertas: list[Alerta] = []
    for numero, desvio in enumerate(desvios, start=1):
        assert definicao is not None
        titulo = f"Desvio {numero}: {DESCRICAO_REGRA[desvio.regra]}"
        caminho = _salvar_frame(
            arquivo,
            sequencia,
            desvio.frame,
            pasta / f"desvio_{numero:02d}.jpg",
            f"{titulo} ({desvio.valor:.0f} graus)",
            _DESTAQUE[desvio.regra],
        )
        evidencias.append(EvidenciaFrame(titulo, str(caminho), desvio.t))
        alertas.append(
            alerta_desvio(desvio, definicao, paciente_id, clipe_id, inicio, _relativo(caminho))
        )
    for numero, queda in enumerate(quedas, start=1):
        titulo = f"Queda {numero}"
        no_quadro: list[objetos.Objeto] = []
        perto: list[str] = []
        pose_queda = next(q for q in sequencia.poses if q.frame == queda.frame)
        if detectar_objetos and pose_queda.caixa is not None:
            frame = p.ler_frame(arquivo, queda.frame, sequencia.recorte)
            no_quadro = objetos.detectar(frame, modelo_objetos)
            perto = objetos.proximos(no_quadro, pose_queda.caixa)
        caminho = _salvar_frame(
            arquivo,
            sequencia,
            queda.frame,
            pasta / f"queda_{numero:02d}.jpg",
            titulo + (f" junto a {' e '.join(perto)}" if perto else ""),
            objetos_no_quadro=no_quadro,
        )
        evidencias.append(EvidenciaFrame(titulo, str(caminho), queda.t))
        alertas.append(
            alerta_queda(queda, paciente_id, clipe_id, inicio, _relativo(caminho), perto)
        )
    video = (
        gerar_video_anotado(arquivo, sequencia, pasta / "anotado.webm") if video_anotado else None
    )

    resultado = ResultadoVideo(
        clipe_id=clipe_id,
        arquivo=arquivo.name,
        paciente_id=paciente_id,
        exercicio=definicao,
        vista=str(vista),
        sequencia=sequencia,
        serie=serie,
        repeticoes=repeticoes,
        desvios=desvios,
        quedas=quedas,
        evidencias=evidencias,
        relatorio=pasta / report.NOME_RELATORIO,
        alertas=alertas,
        video=video,
    )
    report.gerar(resultado, pasta)
    return resultado
