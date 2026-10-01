"""Alertas mock no Contrato do Alerta, para o Streamlit rodar antes dos Analisadores reais.

Cobre todas as origens, tipos, severidades e status, incluindo Alertas Compostos que
apontam para Alertas do mesmo Paciente de Demonstração. Determinístico: mesmos ids a cada
execução, então popular o banco duas vezes não duplica nada.

    uv run python -m src.core.mock            # popula data/alertas.db
"""

from __future__ import annotations

import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any

from src.core import db
from src.core.alerts import Alerta, Origem, Severidade, Status, Tipo, reconhecer, resolver

INICIO_PADRAO = datetime(2026, 10, 1, 8, 0)


def gerar_alertas_mock(inicio: datetime = INICIO_PADRAO, seed: int = 4) -> list[Alerta]:
    rng = random.Random(seed)

    def alerta(
        minutos: float,
        paciente_id: str,
        origem: Origem,
        tipo: Tipo,
        severidade: Severidade,
        descricao: str,
        evidencia: dict[str, Any],
        **kw: Any,
    ) -> Alerta:
        return Alerta(
            id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            paciente_id=paciente_id,
            origem=origem,
            tipo=tipo,
            severidade=severidade,
            descricao=descricao,
            evidencia=evidencia,
            detectado_em=inicio + timedelta(minutes=minutos),
            **kw,
        )

    def composto(
        minutos: float, severidade: Severidade, descricao: str, origens: list[Alerta]
    ) -> Alerta:
        return alerta(
            minutos,
            origens[0].paciente_id,
            Origem.FUSAO,
            Tipo.COMPOSTO,
            severidade,
            descricao,
            {"regra": descricao, "janela_h": 24},
            alertas_origem=[a.id for a in origens],
        )

    # demo-01: clímax do vídeo, possível insuficiência respiratória (Composto crítico)
    spo2 = alerta(
        30,
        "demo-01",
        Origem.SINAIS_VITAIS,
        Tipo.SINAL_VITAL,
        Severidade.ALTA,
        "SpO2 abaixo de 90% por 6 minutos (regra clínica)",
        {
            "serie": "SpO2",
            "janela": [1800, 2160],
            "valores": [91, 89, 88, 87, 88, 89, 88],
            "tecnica": "regras",
        },
    )
    fr = alerta(
        34,
        "demo-01",
        Origem.SINAIS_VITAIS,
        Tipo.SINAL_VITAL,
        Severidade.MEDIA,
        "FR 3,4σ acima da história recente do paciente (z-score)",
        {"serie": "FR", "janela": [1920, 2040], "valores": [22, 26, 27], "tecnica": "zscore"},
    )
    vocal = alerta(
        45,
        "demo-01",
        Origem.AUDIO,
        Tipo.ALTERACAO_VOCAL,
        Severidade.MEDIA,
        "Voz com padrão de dificuldade respiratória (probabilidade 0,81)",
        {
            "trecho": "well, I just feel so tired lately",
            "inicio": 83.2,
            "fim": 97.0,
            "arquivo": "consulta_day1_c03.wav",
            "probabilidade": 0.81,
        },
    )
    dispneia = alerta(
        46,
        "demo-01",
        Origem.AUDIO,
        Tipo.TERMO_CRITICO,
        Severidade.ALTA,
        "Paciente relata falta de ar ao falar (Termo Crítico: dyspnea)",
        {
            "trecho": "I get really short of breath when I walk to the bathroom",
            "inicio": 121.5,
            "entidade": "SymptomOrSign",
            "termo": "dyspnea",
        },
    )
    insuficiencia = composto(
        47,
        Severidade.CRITICA,
        "Possível insuficiência respiratória",
        [spo2, vocal, dispneia],
    )

    # demo-02: Queda após início de opioide
    opioide = alerta(
        120,
        "demo-02",
        Origem.PRESCRICAO,
        Tipo.MUDANCA_TERAPEUTICA,
        Severidade.MEDIA,
        "Início de droga de alta vigilância: morfina 4 mg IV",
        {
            "droga": "Morphine Sulfate",
            "regra": "alta_vigilancia",
            "dose": "4 mg",
            "via": "IV",
            "inicio": "2026-10-01T10:00:00",
        },
    )
    queda = alerta(
        300,
        "demo-02",
        Origem.VIDEO,
        Tipo.QUEDA,
        Severidade.ALTA,
        "Queda detectada: paciente passou de em pé para o chão",
        {"frame": "results/video/quarto_02_f0412.jpg", "t": 13.7, "razao_bbox": 0.42},
    )
    queda_medicacao = composto(
        301,
        Severidade.ALTA,
        "Queda possivelmente associada a medicação",
        [opioide, queda],
    )

    # demo-03: hipotensão após anti-hipertensivo + agravamento em reabilitação
    anti_hipertensivo = alerta(
        60,
        "demo-03",
        Origem.PRESCRICAO,
        Tipo.MUDANCA_TERAPEUTICA,
        Severidade.BAIXA,
        "Aumento de dose > 50% em < 24h: metoprolol 25 → 50 mg",
        {
            "droga": "Metoprolol Tartrate",
            "regra": "salto_de_dose",
            "dose_anterior": "25 mg",
            "dose_nova": "50 mg",
        },
    )
    pas = alerta(
        150,
        "demo-03",
        Origem.SINAIS_VITAIS,
        Tipo.SINAL_VITAL,
        Severidade.ALTA,
        "PAS abaixo de 90 mmHg (Isolation Forest + regra clínica)",
        {
            "serie": "PAS",
            "janela": [8700, 9000],
            "valores": [98, 91, 86, 84],
            "tecnica": "isolation_forest",
        },
    )
    efeito_adverso = composto(
        152,
        Severidade.ALTA,
        "Possível efeito adverso de medicação",
        [anti_hipertensivo, pas],
    )
    desvio = alerta(
        200,
        "demo-03",
        Origem.VIDEO,
        Tipo.DESVIO_EXECUCAO,
        Severidade.BAIXA,
        "Agachamento com amplitude insuficiente: flexão do joelho de 62° (esperado 70°–120°)",
        {
            "frame": "results/video/agachamento_03_f0288.jpg",
            "t": 12.4,
            "angulo": 62,
            "articulacao": "joelho_esq",
            "faixa_esperada": [70, 120],
        },
    )
    desvio_2 = alerta(
        205,
        "demo-03",
        Origem.VIDEO,
        Tipo.DESVIO_EXECUCAO,
        Severidade.MEDIA,
        "Assimetria esquerda/direita de 18° no joelho",
        {
            "frame": "results/video/agachamento_03_f0510.jpg",
            "t": 21.3,
            "angulo": 18,
            "articulacao": "joelho",
            "limite": 10,
        },
    )
    dor = alerta(
        240,
        "demo-03",
        Origem.AUDIO,
        Tipo.TERMO_CRITICO,
        Severidade.MEDIA,
        "Paciente relata dor no joelho durante exercícios (Termo Crítico: pain)",
        {
            "trecho": "my knee hurts a lot when I do the squats",
            "inicio": 45.8,
            "entidade": "SymptomOrSign",
            "termo": "knee pain",
        },
    )
    agravamento = composto(
        241,
        Severidade.MEDIA,
        "Possível agravamento em reabilitação",
        [desvio, desvio_2, dor],
    )

    # avulso: uploads no Laboratório de Análise
    lab_video = alerta(
        400,
        "avulso",
        Origem.VIDEO,
        Tipo.DESVIO_EXECUCAO,
        Severidade.BAIXA,
        "Elevação lateral de braço com compensação de tronco (inclinação 14°)",
        {
            "frame": "results/video/upload_f0120.jpg",
            "t": 4.0,
            "angulo": 14,
            "articulacao": "tronco",
        },
    )
    lab_audio = alerta(
        410,
        "avulso",
        Origem.AUDIO,
        Tipo.TERMO_CRITICO,
        Severidade.BAIXA,
        "Menção a fadiga na Consulta enviada (Termo Crítico: fatigue)",
        {
            "trecho": "I've been feeling tired all the time",
            "inicio": 12.1,
            "entidade": "SymptomOrSign",
            "termo": "fatigue",
        },
    )

    def depois(alerta: Alerta, minutos: float) -> datetime:
        return alerta.detectado_em + timedelta(minutes=minutos)

    anti_hipertensivo = resolver(
        reconhecer(anti_hipertensivo, "farm. Carla", depois(anti_hipertensivo, 20))
    )
    queda = reconhecer(queda, "enf. Ana", depois(queda, 2))
    efeito_adverso = reconhecer(efeito_adverso, "dr. Bruno", depois(efeito_adverso, 8))
    lab_video = resolver(reconhecer(lab_video, "fisio. Diego", depois(lab_video, 30)))

    return [
        spo2,
        fr,
        vocal,
        dispneia,
        insuficiencia,
        opioide,
        queda,
        queda_medicacao,
        anti_hipertensivo,
        pas,
        efeito_adverso,
        desvio,
        desvio_2,
        dor,
        agravamento,
        lab_video,
        lab_audio,
    ]


def popular_mock(conn: sqlite3.Connection) -> list[Alerta]:
    """Grava os Alertas mock que ainda não estão no banco, sem desfazer Reconhecimentos."""
    alertas = gerar_alertas_mock()
    for alerta in alertas:
        if db.obter(conn, alerta.id) is None:
            db.salvar(conn, alerta)
    return alertas


if __name__ == "__main__":
    with db.conectar() as conexao:
        gravados = popular_mock(conexao)
    novos = sum(a.status is Status.NOVO for a in gravados)
    print(f"{len(gravados)} Alertas mock gravados em {db.CAMINHO_PADRAO} ({novos} novos)")
