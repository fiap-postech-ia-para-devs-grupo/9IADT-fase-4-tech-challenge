"""Luz de chamada e acionamento da equipe por WhatsApp (Evolution API).

Quando um Alerta exige ajuda imediata — severidade crítica, ou uma Queda —, a equipe de plantão
que optou por receber alertas é acionada no WhatsApp. Se o Alerta continua sem Reconhecimento
depois de `MINUTOS_ESCALONAMENTO` minutos do plantão, o acionamento escala para a equipe médica.
Cada (Alerta, nível, profissional) é enviado uma vez só; tudo fica registrado na tabela
`notificacoes`, que a tela de Administração mostra.

Sem `EVOLUTION_URL`/`EVOLUTION_APIKEY`/`EVOLUTION_INSTANCE` no `.env`, o envio é **simulado**:
registra o que seria enviado e não chama nada externo.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.core import cadastro
from src.core.alerts import Alerta, Severidade, Status, Tipo
from src.core.ficha import ROTULO_ORIGEM, ROTULO_SEVERIDADE

MINUTOS_ESCALONAMENTO = 5
JANELA_RECENTE = timedelta(minutes=15)  # Alerta mais antigo que isso no relógio não aciona

SCHEMA = """
CREATE TABLE IF NOT EXISTS notificacoes (
    id              TEXT PRIMARY KEY,
    alerta_id       TEXT NOT NULL,
    nivel           INTEGER NOT NULL,      -- 1: plantão; 2: escalonado para a equipe médica
    profissional_id TEXT NOT NULL,
    numero          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('enviado', 'simulado', 'falhou')),
    detalhe         TEXT NOT NULL DEFAULT '',
    criado_em       TEXT NOT NULL,         -- relógio do plantão
    enviado_em      TEXT NOT NULL,         -- relógio real
    UNIQUE (alerta_id, nivel, profissional_id)
);
"""


@dataclass(frozen=True)
class ConfigWhatsApp:
    url: str
    apikey: str
    instancia: str
    app_url: str

    @property
    def ativo(self) -> bool:
        return bool(self.url and self.apikey and self.instancia)


@dataclass(frozen=True)
class Notificacao:
    alerta_id: str
    nivel: int
    profissional_id: str
    numero: str
    status: str
    detalhe: str
    criado_em: datetime


def config() -> ConfigWhatsApp:
    load_dotenv()
    return ConfigWhatsApp(
        url=os.environ.get("EVOLUTION_URL", "").strip().rstrip("/"),
        apikey=os.environ.get("EVOLUTION_APIKEY", "").strip(),
        instancia=os.environ.get("EVOLUTION_INSTANCE", "").strip(),
        app_url=os.environ.get("APP_URL", "http://localhost:8501").strip().rstrip("/"),
    )


TECNICAS_DE_CHAMADA = {"news2_tendencia"}
EVENTOS_SONOROS_DE_CHAMADA = {"grito", "chamado"}


def exige_chamada(alerta: Alerta) -> bool:
    """Sirene vermelha + WhatsApp: instabilidade crítica, paciente no chão (Queda), piora do
    NEWS2, ou o paciente gritando/chamando a equipe (monitoramento de áudio do leito)."""
    ev = alerta.evidencia
    return (
        alerta.severidade is Severidade.CRITICA
        or alerta.tipo is Tipo.QUEDA
        or ev.get("tecnica") in TECNICAS_DE_CHAMADA
        or ev.get("evento") in EVENTOS_SONOROS_DE_CHAMADA
    )


def numero_valido(numero: str) -> str | None:
    """Só dígitos, com DDI e DDD (ex.: 5511999999999); None se não parecer um celular."""
    digitos = "".join(c for c in numero if c.isdigit())
    if len(digitos) == 11:  # sem DDI: assume Brasil
        digitos = "55" + digitos
    return digitos if 12 <= len(digitos) <= 13 else None


def mensagem(alerta: Alerta, paciente: str, nivel: int, app_url: str) -> str:
    titulo = "⏫ ESCALONADO — sem Reconhecimento" if nivel == 2 else "🚨 CHAMADA DE EMERGÊNCIA"
    return (
        f"{titulo}\n"
        f"*{ROTULO_SEVERIDADE[alerta.severidade]}* · {paciente}\n"
        f"{alerta.descricao}\n"
        f"Origem: {ROTULO_ORIGEM[alerta.origem]} · {alerta.detectado_em:%d/%m %H:%M}\n"
        f"Reconheça na Central de Alertas: {app_url}/alertas\n"
        "_Monitoramento Multimodal — demonstração acadêmica (FIAP), pacientes fictícios._"
    )


def enviar_whatsapp(cfg: ConfigWhatsApp, numero: str, texto: str) -> tuple[str, str]:
    """(status, detalhe). Sem configuração, simula; erro de rede vira 'falhou', sem exceção."""
    if not cfg.ativo:
        return "simulado", "Evolution API não configurada no .env"
    pedido = urllib.request.Request(
        f"{cfg.url}/message/sendText/{cfg.instancia}",
        data=json.dumps({"number": numero, "text": texto}).encode(),
        headers={"Content-Type": "application/json", "apikey": cfg.apikey},
        method="POST",
    )
    try:
        with urllib.request.urlopen(pedido, timeout=10) as resposta:
            corpo = json.loads(resposta.read() or b"{}")
        return "enviado", str(corpo.get("key", {}).get("id", "ok"))
    except (urllib.error.URLError, TimeoutError, ValueError) as erro:
        return "falhou", str(erro)[:200]


def _preparar(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _ja_enviado(conn: sqlite3.Connection, alerta_id: str, nivel: int, profissional_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM notificacoes WHERE alerta_id = ? AND nivel = ? AND profissional_id = ?",
            (alerta_id, nivel, profissional_id),
        ).fetchone()
        is not None
    )


def destinatarios(
    equipe: Iterable[cadastro.Profissional], nivel: int
) -> list[cadastro.Profissional]:
    com_numero = [p for p in equipe if p.recebe_alertas and numero_valido(p.telefone)]
    if nivel == 2:
        medicos = [p for p in com_numero if p.funcao == "Medicina"]
        return medicos or com_numero
    return com_numero


def acionar(
    conn: sqlite3.Connection,
    alertas: Iterable[Alerta],
    agora: datetime,
    rotulo_paciente: dict[str, str],
    cfg: ConfigWhatsApp | None = None,
) -> list[Notificacao]:
    """Aciona a equipe para os Alertas que pedem chamada e ainda não foram reconhecidos.

    **Um disparo por leito:** enquanto o leito tem chamado acionado e não reconhecido, outros
    Alertas do mesmo paciente não geram nova mensagem (o mais grave vai primeiro). Depois do
    Reconhecimento, um evento novo volta a acionar. O escalonamento também é um por leito.
    """
    _preparar(conn)
    cfg = cfg or config()
    equipe = cadastro.equipe(conn)
    acionados = leitos_acionados(conn)
    feitas: list[Notificacao] = []
    pesos = {s: i for i, s in enumerate(Severidade)}
    ordenados = sorted(alertas, key=lambda a: (pesos[a.severidade], a.detectado_em), reverse=True)
    for alerta in ordenados:
        if alerta.status is not Status.NOVO or not exige_chamada(alerta):
            continue
        if not (agora - JANELA_RECENTE <= alerta.detectado_em <= agora):
            continue
        atraso = agora - alerta.detectado_em
        niveis = [1] + ([2] if atraso >= timedelta(minutes=MINUTOS_ESCALONAMENTO) else [])
        for nivel in niveis:
            if (alerta.paciente_id, nivel) in acionados:
                continue
            acionados.add((alerta.paciente_id, nivel))
            for profissional in destinatarios(equipe, nivel):
                if _ja_enviado(conn, alerta.id, nivel, profissional.id):
                    continue
                numero = numero_valido(profissional.telefone) or ""
                paciente = rotulo_paciente.get(alerta.paciente_id, alerta.paciente_id)
                texto = mensagem(alerta, paciente, nivel, cfg.app_url)
                status, detalhe = enviar_whatsapp(cfg, numero, texto)
                feita = Notificacao(
                    alerta.id, nivel, profissional.id, numero, status, detalhe, agora
                )
                with conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO notificacoes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            alerta.id,
                            nivel,
                            profissional.id,
                            numero,
                            status,
                            detalhe,
                            agora.isoformat(),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                feitas.append(feita)
    return feitas


def leitos_acionados(conn: sqlite3.Connection) -> set[tuple[str, int]]:
    """(paciente, nível) com mensagem já enviada para um Alerta ainda sem Reconhecimento."""
    _preparar(conn)
    linhas = conn.execute(
        "SELECT DISTINCT a.paciente_id, n.nivel FROM notificacoes n "
        "JOIN alertas a ON a.id = n.alerta_id WHERE a.status = 'novo'"
    )
    return {(paciente, nivel) for paciente, nivel in linhas}


def historico(conn: sqlite3.Connection, limite: int = 50) -> list[dict]:
    _preparar(conn)
    linhas = conn.execute(
        "SELECT n.enviado_em, n.criado_em, n.nivel, p.nome, n.numero, n.status, n.detalhe, "
        "n.alerta_id FROM notificacoes n LEFT JOIN profissionais p ON p.id = n.profissional_id "
        "ORDER BY n.enviado_em DESC LIMIT ?",
        (limite,),
    )
    colunas = ("enviado_em", "plantao", "nivel", "profissional", "numero", "status", "detalhe")
    return [
        dict(zip(colunas, linha[:7], strict=True)) | {"alerta_id": linha[7]} for linha in linhas
    ]


def limpar_historico(conn: sqlite3.Connection) -> None:
    """Permite reapresentar a demonstração: os mesmos Alertas voltam a acionar a equipe."""
    _preparar(conn)
    with conn:
        conn.execute("DELETE FROM notificacoes")


def mascarar(numero: str) -> str:
    return f"{numero[:4]}•••••{numero[-4:]}" if len(numero) > 8 else numero
