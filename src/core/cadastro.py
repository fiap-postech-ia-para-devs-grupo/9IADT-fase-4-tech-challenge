"""Cadastros da unidade: leitos, equipe, admissões (internações) e evolução clínica.

Ficam no mesmo SQLite dos Alertas, em tabelas próprias; o Contrato do Alerta não muda. A
admissão liga um paciente a um leito e às suas **fontes de monitoramento**: o monitor de Sinais
Vitais (registro do MIMIC-III), a prescrição (internação do MIMIC-IV Demo), a Consulta gravada
(PriMock57) e a câmera do leito (webcam ou clipe). Como nos Pacientes de Demonstração
(docs/adr/0001), as fontes são dados públicos de pessoas diferentes; o paciente é composto.

Na primeira conexão, os Pacientes de Demonstração de `data/demo_patients.json` entram como
admissões já feitas, e a unidade recebe leitos e uma equipe de exemplo.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from src.core import demo
from src.core.demo import FonteConsulta, FontePrescricoes, FonteVideo, PacienteDemo
from src.core.ficha import Ficha, carregar_fichas

SCHEMA = """
CREATE TABLE IF NOT EXISTS leitos (
    id        TEXT PRIMARY KEY,
    unidade   TEXT NOT NULL,
    tipo      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profissionais (
    id        TEXT PRIMARY KEY,
    nome      TEXT NOT NULL,
    funcao    TEXT NOT NULL,
    registro  TEXT NOT NULL DEFAULT '',
    ativo     INTEGER NOT NULL DEFAULT 1,
    telefone  TEXT NOT NULL DEFAULT '',     -- WhatsApp, só dígitos com DDI: 5511999999999
    recebe_alertas INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS internacoes (
    id            TEXT PRIMARY KEY,
    nome          TEXT NOT NULL,
    idade         INTEGER,
    sexo          TEXT,
    leito_id      TEXT NOT NULL REFERENCES leitos(id),
    motivo        TEXT NOT NULL DEFAULT '',
    admitido_em   TEXT NOT NULL,            -- ISO 8601, relógio do plantão
    admitido_por  TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL CHECK (status IN ('internado', 'alta')),
    fontes        TEXT NOT NULL             -- JSON
);
CREATE TABLE IF NOT EXISTS evolucoes (
    id          TEXT PRIMARY KEY,
    paciente_id TEXT NOT NULL,
    autor       TEXT NOT NULL,
    tipo        TEXT NOT NULL CHECK (tipo IN ('evolucao', 'conduta', 'admissao', 'alta')),
    texto       TEXT NOT NULL,
    criado_em   TEXT NOT NULL,              -- ISO 8601, relógio do plantão
    alerta_id   TEXT
);
"""

FUNCOES = ("Enfermagem", "Medicina", "Fisioterapia", "Técnico de enfermagem")
TIPOS_LEITO = ("UTI", "Unidade Coronariana", "Enfermaria")

LEITOS_EXEMPLO = (
    ("UTI-01", "UTI adulto", "UTI"),
    ("UTI-02", "UTI adulto", "UTI"),
    ("UTI-04", "UTI adulto", "UTI"),
    ("UCO-12", "Unidade Coronariana", "Unidade Coronariana"),
    ("ENF-207", "Clínica médica", "Enfermaria"),
    ("ENF-208", "Clínica médica", "Enfermaria"),
)
EQUIPE_EXEMPLO = (
    ("enf-ana", "Ana Souza", "Enfermagem", "COREN-SP 123456"),
    ("med-bruno", "Bruno Lima", "Medicina", "CRM-SP 98765"),
    ("fis-carla", "Carla Mendes", "Fisioterapia", "CREFITO-3 4567-F"),
    ("tec-diego", "Diego Alves", "Técnico de enfermagem", "COREN-SP 654321"),
)


@dataclass(frozen=True)
class Leito:
    id: str
    unidade: str
    tipo: str


@dataclass(frozen=True)
class Profissional:
    id: str
    nome: str
    funcao: str
    registro: str = ""
    telefone: str = ""
    recebe_alertas: bool = False

    @property
    def rotulo(self) -> str:
        prefixo = {
            "Enfermagem": "Enf.",
            "Medicina": "Dr(a).",
            "Fisioterapia": "Ft.",
            "Técnico de enfermagem": "Téc.",
        }.get(self.funcao, "")
        return f"{prefixo} {self.nome}".strip()


@dataclass(frozen=True)
class Fontes:
    """O que monitora o paciente: cada campo é uma fonte de dados de um Analisador."""

    registro_vitais: str
    subject_id: int
    hadm_id: int
    ancora_prescricao: str  # ISO: instante do MIMIC-IV que cai em `ancora_min`
    ancora_min: float
    consulta_id: str
    consulta_inicio_min: float
    camera: str | None = None  # id de clipe em data/video/clipes.json, ou "webcam"
    camera_inicio_min: float = 0.0


@dataclass(frozen=True)
class Internacao:
    id: str
    nome: str
    idade: int | None
    sexo: str | None
    leito_id: str
    motivo: str
    admitido_em: datetime
    admitido_por: str
    status: str
    fontes: Fontes

    def como_paciente(self) -> PacienteDemo:
        """No formato que `demo.montar` entende, para reaproveitar Analisadores e Fusão."""
        f = self.fontes
        return PacienteDemo(
            id=self.id,
            nome=self.nome,
            resumo=self.motivo,
            internacao_inicio=self.admitido_em,
            registro=f.registro_vitais,
            prescricoes=FontePrescricoes(
                f.subject_id, f.hadm_id, datetime.fromisoformat(f.ancora_prescricao), f.ancora_min
            ),
            consulta=FonteConsulta(f.consulta_id, f.consulta_inicio_min),
            video=FonteVideo(
                None if f.camera in (None, "webcam") else f.camera, f.camera_inicio_min
            ),
            leito=self.leito_id,
        )


@dataclass(frozen=True)
class Evolucao:
    paciente_id: str
    autor: str
    tipo: str
    texto: str
    criado_em: datetime
    alerta_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def preparar(conn: sqlite3.Connection) -> None:
    """Cria as tabelas e, numa unidade vazia, os exemplos e as admissões de demonstração."""
    conn.executescript(SCHEMA)
    colunas = {linha[1] for linha in conn.execute("PRAGMA table_info(profissionais)")}
    with conn:  # bancos criados antes do WhatsApp
        if "telefone" not in colunas:
            conn.execute("ALTER TABLE profissionais ADD COLUMN telefone TEXT NOT NULL DEFAULT ''")
        if "recebe_alertas" not in colunas:
            conn.execute(
                "ALTER TABLE profissionais ADD COLUMN recebe_alertas INTEGER NOT NULL DEFAULT 0"
            )
        if not conn.execute("SELECT 1 FROM leitos LIMIT 1").fetchone():
            conn.executemany("INSERT INTO leitos VALUES (?, ?, ?)", LEITOS_EXEMPLO)
        if not conn.execute("SELECT 1 FROM profissionais LIMIT 1").fetchone():
            conn.executemany(
                "INSERT INTO profissionais (id, nome, funcao, registro) VALUES (?, ?, ?, ?)",
                EQUIPE_EXEMPLO,
            )
    if not conn.execute("SELECT 1 FROM internacoes LIMIT 1").fetchone():
        pacientes = demo.carregar_pacientes()
        fichas = carregar_fichas(pacientes)
        for paciente in pacientes:
            admitir(conn, _de_paciente_demo(paciente, fichas.get(paciente.id)), "carga inicial")


# Identidade fictícia dos Pacientes de Demonstração (idade e sexo vêm do MIMIC-IV, em `ficha`).
PERSONAS_DEMO = {
    "demo-01": ("José Almeida (fictício)", "Dispneia em observação"),
    "demo-02": ("Antônio Ferreira (fictício)", "Taquiarritmia em ajuste de beta-bloqueador"),
    "demo-03": ("Carlos Nogueira (fictício)", "Dor pós-operatória em analgesia com opioide"),
}


def _de_paciente_demo(p: PacienteDemo, ficha: Ficha | None = None) -> Internacao:
    nome, motivo = PERSONAS_DEMO.get(p.id, (p.nome, p.resumo))
    return Internacao(
        id=p.id,
        nome=nome,
        idade=ficha.idade if ficha else None,
        sexo=ficha.sexo if ficha else None,
        leito_id=p.leito or p.id,
        motivo=motivo,
        admitido_em=p.internacao_inicio,
        admitido_por="carga inicial",
        status="internado",
        fontes=Fontes(
            registro_vitais=p.registro,
            subject_id=p.prescricoes.subject_id,
            hadm_id=p.prescricoes.hadm_id,
            ancora_prescricao=p.prescricoes.ancora.isoformat(),
            ancora_min=p.prescricoes.ancora_min,
            consulta_id=p.consulta.id,
            consulta_inicio_min=p.consulta.inicio_min,
            camera=p.video.clipe,
            camera_inicio_min=p.video.inicio_min,
        ),
    )


# --- leitos e equipe ------------------------------------------------------------------------


def leitos(conn: sqlite3.Connection) -> list[Leito]:
    return [Leito(*linha) for linha in conn.execute("SELECT * FROM leitos ORDER BY id")]


def salvar_leito(conn: sqlite3.Connection, leito: Leito) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO leitos VALUES (?, ?, ?)", tuple(asdict(leito).values())
        )


def leitos_livres(conn: sqlite3.Connection) -> list[Leito]:
    ocupados = {i.leito_id for i in internacoes(conn)}
    return [leito for leito in leitos(conn) if leito.id not in ocupados]


def equipe(conn: sqlite3.Connection) -> list[Profissional]:
    linhas = conn.execute(
        "SELECT id, nome, funcao, registro, telefone, recebe_alertas FROM profissionais "
        "WHERE ativo = 1 ORDER BY nome"
    )
    return [Profissional(*linha[:5], bool(linha[5])) for linha in linhas]


def salvar_profissional(conn: sqlite3.Connection, profissional: Profissional) -> None:
    dados = asdict(profissional)
    dados["recebe_alertas"] = int(profissional.recebe_alertas)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO profissionais "
            "(id, nome, funcao, registro, telefone, recebe_alertas) VALUES (?, ?, ?, ?, ?, ?)",
            tuple(dados.values()),
        )


def desativar_profissional(conn: sqlite3.Connection, profissional_id: str) -> None:
    with conn:
        conn.execute("UPDATE profissionais SET ativo = 0 WHERE id = ?", (profissional_id,))


# --- admissão, alta e evolução --------------------------------------------------------------


def _de_linha(linha: sqlite3.Row | tuple) -> Internacao:
    (id_, nome, idade, sexo, leito_id, motivo, admitido_em, por, status, fontes) = tuple(linha)
    return Internacao(
        id=id_,
        nome=nome,
        idade=idade,
        sexo=sexo,
        leito_id=leito_id,
        motivo=motivo,
        admitido_em=datetime.fromisoformat(admitido_em),
        admitido_por=por,
        status=status,
        fontes=Fontes(**json.loads(fontes)),
    )


def internacoes(conn: sqlite3.Connection, *, incluir_altas: bool = False) -> list[Internacao]:
    sql = "SELECT * FROM internacoes"
    if not incluir_altas:
        sql += " WHERE status = 'internado'"
    return [_de_linha(linha) for linha in conn.execute(sql + " ORDER BY leito_id")]


def obter(conn: sqlite3.Connection, paciente_id: str) -> Internacao | None:
    linha = conn.execute("SELECT * FROM internacoes WHERE id = ?", (paciente_id,)).fetchone()
    return _de_linha(linha) if linha else None


class LeitoOcupado(ValueError):
    pass


def admitir(conn: sqlite3.Connection, internacao: Internacao, autor: str) -> Internacao:
    ocupante = next((i for i in internacoes(conn) if i.leito_id == internacao.leito_id), None)
    if ocupante is not None and ocupante.id != internacao.id:
        raise LeitoOcupado(f"Leito {internacao.leito_id} ocupado por {ocupante.nome}")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO internacoes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                internacao.id,
                internacao.nome,
                internacao.idade,
                internacao.sexo,
                internacao.leito_id,
                internacao.motivo,
                internacao.admitido_em.isoformat(),
                autor,
                "internado",
                json.dumps(asdict(internacao.fontes)),
            ),
        )
    registrar(
        conn,
        Evolucao(
            internacao.id,
            autor,
            "admissao",
            f"Admissão no leito {internacao.leito_id}. {internacao.motivo}".strip(),
            internacao.admitido_em,
        ),
    )
    return internacao


def dar_alta(conn: sqlite3.Connection, paciente_id: str, autor: str, em: datetime) -> None:
    with conn:
        conn.execute("UPDATE internacoes SET status = 'alta' WHERE id = ?", (paciente_id,))
    registrar(conn, Evolucao(paciente_id, autor, "alta", "Alta do monitoramento.", em))


def registrar(conn: sqlite3.Connection, evolucao: Evolucao) -> None:
    conn.executescript(SCHEMA)
    with conn:
        conn.execute(
            "INSERT INTO evolucoes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                evolucao.id,
                evolucao.paciente_id,
                evolucao.autor,
                evolucao.tipo,
                evolucao.texto,
                evolucao.criado_em.isoformat(),
                evolucao.alerta_id,
            ),
        )


def evolucoes(conn: sqlite3.Connection, paciente_id: str) -> list[Evolucao]:
    linhas = conn.execute(
        "SELECT paciente_id, autor, tipo, texto, criado_em, alerta_id, id FROM evolucoes "
        "WHERE paciente_id = ? ORDER BY criado_em DESC",
        (paciente_id,),
    )
    return [
        Evolucao(p, autor, tipo, texto, datetime.fromisoformat(em), alerta, id_)
        for p, autor, tipo, texto, em, alerta, id_ in linhas
    ]


def novo_id() -> str:
    return f"pac-{uuid.uuid4().hex[:6]}"


def dados(internacao: Internacao) -> dict[str, Any]:
    return asdict(internacao)
