"""Pacientes de Demonstração: uma internação simulada a partir de fontes públicas distintas.

Lê `data/demo_patients.json` (ESTRATEGIA.md, seção 8; docs/adr/0001), desloca cada fonte para o
relógio da internação, roda os Analisadores e funde os Alertas. A internação dura o que dura o
registro de Sinais Vitais; Alertas de prescrição fora desse trecho ficam de fora.

    internacao = demo.montar(demo.carregar_pacientes()[0])
    eventos = demo.fundir_eventos(internacao.eventos)
    demo.gravar(conn, demo.visiveis(eventos, agora))

`EventoAlerta.aparece_em` é quando a equipe vê o Alerta no replay: para Sinais Vitais, o minuto
em que o detector o emite.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.analyzers.audio import semantic
from src.analyzers.audio.azure import AzureConfig
from src.analyzers.prescriptions import load as load_prescricoes
from src.analyzers.prescriptions import rules
from src.analyzers.vitals import load as load_vitals
from src.analyzers.vitals import stream
from src.core import db, fusion, mock
from src.core.alerts import Alerta, Origem

_RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = _RAIZ / "data" / "demo_patients.json"
PASTA_VITALS = _RAIZ / "data" / "vitals" / "mimic3wdb"
CACHE_AUDIO = _RAIZ / "data" / "audio" / "cache_primock57"

_NAMESPACE = uuid.UUID("5e0c7a3b-1f6d-4c28-9e41-7b2d8f3a6c10")
_SEM_AZURE = AzureConfig("", "", "", "")


class DadosAusentes(FileNotFoundError):
    """Uma fonte do Paciente de Demonstração não está em `data/`."""


@dataclass(frozen=True)
class FontePrescricoes:
    subject_id: int
    hadm_id: int
    ancora: datetime  # instante no MIMIC-IV que cai em `ancora_min` da internação
    ancora_min: float


@dataclass(frozen=True)
class FonteConsulta:
    id: str
    inicio_min: float


@dataclass(frozen=True)
class FonteVideo:
    clipe: str | None
    inicio_min: float


@dataclass(frozen=True)
class PacienteDemo:
    id: str
    nome: str
    resumo: str
    internacao_inicio: datetime
    registro: str
    prescricoes: FontePrescricoes
    consulta: FonteConsulta
    video: FonteVideo

    def em(self, minutos: float) -> datetime:
        return self.internacao_inicio + timedelta(minutes=minutos)


@dataclass(frozen=True)
class EventoAlerta:
    alerta: Alerta
    aparece_em: datetime


@dataclass(frozen=True)
class Internacao:
    paciente: PacienteDemo
    sinais: pd.DataFrame
    prescricoes: pd.DataFrame
    consulta_fim: datetime
    eventos: list[EventoAlerta]

    @property
    def inicio(self) -> datetime:
        return self.sinais.index[0].to_pydatetime()

    @property
    def fim(self) -> datetime:
        return self.sinais.index[-1].to_pydatetime()


def _de_dict(dados: dict[str, Any]) -> PacienteDemo:
    prescricoes = dados["prescricoes"]
    return PacienteDemo(
        id=dados["id"],
        nome=dados["nome"],
        resumo=dados["resumo"],
        internacao_inicio=datetime.fromisoformat(dados["internacao_inicio"]),
        registro=dados["sinais_vitais"]["registro"],
        prescricoes=FontePrescricoes(
            subject_id=prescricoes["subject_id"],
            hadm_id=prescricoes["hadm_id"],
            ancora=datetime.fromisoformat(prescricoes["ancora"]),
            ancora_min=prescricoes["ancora_min"],
        ),
        consulta=FonteConsulta(**dados["consulta"]),
        video=FonteVideo(**dados["video"]),
    )


def carregar_pacientes(caminho: str | Path = CAMINHO_PADRAO) -> list[PacienteDemo]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [_de_dict(paciente) for paciente in dados["pacientes"]]


def _deslocar_janela(janela: pd.DataFrame, inicio: datetime) -> pd.DataFrame:
    return janela.set_axis(janela.index - janela.index[0] + pd.Timestamp(inicio), axis=0)


def eventos_de_sinais(janela: pd.DataFrame, paciente_id: str) -> list[EventoAlerta]:
    """Roda o replay sem pausa e anota o minuto em que cada Alerta foi emitido.

    `janela_analise` cobre o registro inteiro: com uma história menor, um episódio que sai da
    janela é reemitido como Alerta novo a cada minuto.
    """
    minutos_passados = [0]

    def contar_minuto(_: float) -> None:
        minutos_passados[0] += 1

    eventos: list[EventoAlerta] = []
    for alerta in stream.replay(
        janela, paciente_id, janela_analise=len(janela), esperar=contar_minuto
    ):
        emitido_em = janela.index[minutos_passados[0] - 1].to_pydatetime()
        eventos.append(EventoAlerta(alerta, emitido_em))
    return eventos


def _deslocar_prescricoes(
    paciente: PacienteDemo, prescricoes: pd.DataFrame, emar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fonte = paciente.prescricoes
    delta = pd.Timestamp(paciente.em(fonte.ancora_min)) - pd.Timestamp(fonte.ancora)
    da_internacao = prescricoes.loc[prescricoes["hadm_id"] == fonte.hadm_id].copy()
    da_internacao["starttime"] += delta
    da_internacao["stoptime"] += delta
    administracoes = emar.loc[emar["hadm_id"] == fonte.hadm_id].copy()
    administracoes["charttime"] += delta
    return da_internacao, administracoes


def eventos_de_prescricoes(
    paciente: PacienteDemo,
    prescricoes: pd.DataFrame,
    emar: pd.DataFrame,
    inicio: datetime,
    fim: datetime,
) -> list[EventoAlerta]:
    eventos: list[EventoAlerta] = []
    for alerta in rules.analisar(prescricoes, emar):
        if inicio <= alerta.detectado_em <= fim:
            renomeado = replace(
                alerta,
                id=str(uuid.uuid5(_NAMESPACE, f"{paciente.id}|{alerta.id}")),
                paciente_id=paciente.id,
            )
            eventos.append(EventoAlerta(renomeado, renomeado.detectado_em))
    return eventos


def _ler_consulta(
    consulta_id: str, cache_dir: Path
) -> tuple[list[semantic.Trecho], list[list[semantic.EntidadeClinica]]]:
    """Transcrição e entidades do cache versionado: a Consulta não gasta cota do Azure."""
    wav = Path(f"{consulta_id}_patient_16k.wav")
    for arquivo in (f"{wav.stem}.stt.json", f"{consulta_id}.ta_health_v2.json"):
        if not (cache_dir / arquivo).exists():
            raise DadosAusentes(f"Consulta {consulta_id} sem cache: {cache_dir / arquivo}")
    trechos = semantic.transcrever(wav, _SEM_AZURE, cache_dir)
    return trechos, semantic.extrair_entidades_saude(trechos, _SEM_AZURE, cache_dir, consulta_id)


def montar(
    paciente: PacienteDemo,
    *,
    pasta_vitals: Path = PASTA_VITALS,
    pasta_prescricoes: Path = load_prescricoes.PASTA_PADRAO,
    cache_audio: Path = CACHE_AUDIO,
) -> Internacao:
    """Carrega as fontes do paciente e devolve a internação com os Alertas de cada Analisador."""
    if not (pasta_vitals / f"{paciente.registro}.hea").exists():
        raise DadosAusentes("uv run python scripts/download_data.py mimic3wdb")
    if not (pasta_prescricoes / "hosp" / "prescriptions.csv.gz").exists():
        raise DadosAusentes("uv run python scripts/download_data.py mimic4demo")

    janela = _deslocar_janela(
        load_vitals.carregar(pasta_vitals / paciente.registro), paciente.internacao_inicio
    )
    dados = load_prescricoes.carregar(pasta_prescricoes)
    prescricoes, emar = _deslocar_prescricoes(paciente, dados.prescricoes, dados.emar)
    trechos, entidades = _ler_consulta(paciente.consulta.id, cache_audio)
    return montar_de_dados(paciente, janela, prescricoes, emar, trechos, entidades)


def montar_de_dados(
    paciente: PacienteDemo,
    janela: pd.DataFrame,
    prescricoes: pd.DataFrame,
    emar: pd.DataFrame,
    trechos: list[semantic.Trecho],
    entidades: list[list[semantic.EntidadeClinica]],
) -> Internacao:
    inicio = janela.index[0].to_pydatetime()
    fim = janela.index[-1].to_pydatetime()
    consulta_inicio = paciente.em(paciente.consulta.inicio_min)
    termos = semantic.termos_criticos(
        trechos, entidades, paciente.id, paciente.consulta.id, consulta_inicio
    )
    eventos = [
        *eventos_de_sinais(janela, paciente.id),
        *eventos_de_prescricoes(paciente, prescricoes, emar, inicio, fim),
        *(EventoAlerta(termo, termo.detectado_em) for termo in termos),
    ]
    duracao = trechos[-1].fim if trechos else 0.0
    return Internacao(
        paciente=paciente,
        sinais=janela,
        prescricoes=prescricoes,
        consulta_fim=consulta_inicio + timedelta(seconds=duracao),
        eventos=sorted(eventos, key=lambda e: e.aparece_em),
    )


def _distancia_das_origens(composto: Alerta, por_id: dict[str, EventoAlerta]) -> timedelta:
    instantes = [por_id[origem].alerta.detectado_em for origem in composto.alertas_origem]
    return max(instantes) - min(instantes)


def fundir_eventos(
    eventos: Iterable[EventoAlerta], video: Iterable[Alerta] = ()
) -> list[EventoAlerta]:
    """Eventos + Alertas de vídeo + um Alerta Composto por regra de Fusão.

    `fusion.fundir` gera um Composto por par de Alertas na janela de 24 h, e numa internação de
    poucas horas quase todo par coincide. Fica o par mais próximo no tempo de cada regra.
    """
    todos = [*eventos, *(EventoAlerta(alerta, alerta.detectado_em) for alerta in video)]
    por_id = {evento.alerta.id: evento for evento in todos}

    melhores: dict[str, Alerta] = {}
    for composto in fusion.fundir(evento.alerta for evento in todos):
        regra = composto.evidencia["regra"]
        atual = melhores.get(regra)
        chave = (_distancia_das_origens(composto, por_id), composto.detectado_em)
        if atual is None or chave < (_distancia_das_origens(atual, por_id), atual.detectado_em):
            melhores[regra] = composto

    compostos = [
        EventoAlerta(c, max(por_id[origem].aparece_em for origem in c.alertas_origem))
        for c in melhores.values()
    ]
    return sorted([*todos, *compostos], key=lambda e: e.aparece_em)


def visiveis(eventos: Iterable[EventoAlerta], agora: datetime) -> list[EventoAlerta]:
    return [evento for evento in eventos if evento.aparece_em <= agora]


def alertas_de_video(conn: sqlite3.Connection, paciente_id: str) -> list[Alerta]:
    """Alertas gravados pelo Analisador de Vídeo para este paciente (#6)."""
    return db.listar(conn, origem=Origem.VIDEO, paciente_id=paciente_id)


def gravar(conn: sqlite3.Connection, alertas: Iterable[Alerta]) -> int:
    """Grava os Alertas novos, tirando os mock na primeira vez e preservando Reconhecimentos."""
    novos = [alerta for alerta in alertas if db.obter(conn, alerta.id) is None]
    if novos:
        db.remover(conn, [alerta.id for alerta in mock.gerar_alertas_mock()])
    for alerta in novos:
        db.salvar(conn, alerta)
    return len(novos)
