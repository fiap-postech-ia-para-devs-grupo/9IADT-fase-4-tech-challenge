from __future__ import annotations

import io
import json
from datetime import datetime, timedelta
from email.message import Message

import pytest

from src.core import cadastro, db, mock, notificacao
from src.core.alerts import Severidade, Status

SIMULADO = notificacao.ConfigWhatsApp("", "", "", "http://localhost:8501")


@pytest.fixture
def conn(tmp_path):
    conexao = db.conectar(tmp_path / "alertas.db")
    cadastro.preparar(conexao)
    yield conexao
    conexao.close()


def test_unidade_nasce_com_leitos_equipe_e_admissoes_de_demonstracao(conn):
    internados = cadastro.internacoes(conn)

    assert {i.id for i in internados} == {"demo-01", "demo-02", "demo-03"}
    assert all("fictício" in i.nome for i in internados)
    assert len(cadastro.equipe(conn)) == 4
    assert len(cadastro.leitos_livres(conn)) == len(cadastro.leitos(conn)) - 3


def test_admissao_ocupa_o_leito_e_gera_evolucao(conn):
    livre = cadastro.leitos_livres(conn)[0]
    modelo = cadastro.internacoes(conn)[0]
    nova = cadastro.Internacao(
        id="pac-x",
        nome="Maria Teste",
        idade=70,
        sexo="feminino",
        leito_id=livre.id,
        motivo="Pneumonia",
        admitido_em=datetime(2026, 10, 1, 9, 0),
        admitido_por="Enf. Ana Souza",
        status="internado",
        fontes=modelo.fontes,
    )

    cadastro.admitir(conn, nova, "Enf. Ana Souza")

    assert livre.id not in {leito.id for leito in cadastro.leitos_livres(conn)}
    internacao = cadastro.obter(conn, "pac-x")
    assert internacao is not None
    paciente = internacao.como_paciente()
    assert paciente.leito == livre.id
    assert paciente.internacao_inicio == datetime(2026, 10, 1, 9, 0)
    assert cadastro.evolucoes(conn, "pac-x")[0].tipo == "admissao"
    with pytest.raises(cadastro.LeitoOcupado):
        cadastro.admitir(conn, nova.__class__(**{**nova.__dict__, "id": "pac-y"}), "x")


def test_alta_libera_o_leito(conn):
    cadastro.dar_alta(conn, "demo-01", "Dr(a). Bruno Lima", datetime(2026, 10, 1, 12))

    assert "demo-01" not in {i.id for i in cadastro.internacoes(conn)}
    assert "UTI-04" in {leito.id for leito in cadastro.leitos_livres(conn)}


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("(79) 99609-7411", "5579996097411"), ("+55 11 98888-7777", "5511988887777"), ("123", None)],
)
def test_numero_de_whatsapp(entrada, esperado):
    assert notificacao.numero_valido(entrada) == esperado


def _critico():
    return next(
        a
        for a in mock.gerar_alertas_mock()
        if a.severidade is Severidade.CRITICA and a.status is Status.NOVO
    )


def test_chamada_aciona_o_plantao_e_escalona_uma_vez_so(conn):
    ana = next(p for p in cadastro.equipe(conn) if p.id == "enf-ana")
    cadastro.salvar_profissional(
        conn, cadastro.Profissional(ana.id, ana.nome, ana.funcao, "", "11999990000", True)
    )
    critico = _critico()

    primeira = notificacao.acionar(
        conn, [critico], critico.detectado_em + timedelta(minutes=1), {}, SIMULADO
    )
    depois = notificacao.acionar(
        conn, [critico], critico.detectado_em + timedelta(minutes=6), {}, SIMULADO
    )
    de_novo = notificacao.acionar(
        conn, [critico], critico.detectado_em + timedelta(minutes=7), {}, SIMULADO
    )

    assert [(n.nivel, n.status) for n in primeira] == [(1, "simulado")]
    assert depois == []  # sem médico com WhatsApp: ninguém recebe a segunda mensagem
    assert de_novo == []
    assert len(notificacao.historico(conn)) == 1


def test_escalonamento_vai_para_o_medico(conn):
    for p in cadastro.equipe(conn):
        if p.id in {"enf-ana", "med-bruno"}:
            telefone = "11999990000" if p.id == "enf-ana" else "11988880000"
            cadastro.salvar_profissional(
                conn, cadastro.Profissional(p.id, p.nome, p.funcao, "", telefone, True)
            )
    critico = _critico()

    notificacao.acionar(conn, [critico], critico.detectado_em, {}, SIMULADO)
    escalado = notificacao.acionar(
        conn, [critico], critico.detectado_em + timedelta(minutes=6), {}, SIMULADO
    )

    assert [(n.nivel, n.profissional_id) for n in escalado] == [(2, "med-bruno")]


def test_um_disparo_por_leito_ate_o_reconhecimento(conn):
    ana = next(p for p in cadastro.equipe(conn) if p.id == "enf-ana")
    cadastro.salvar_profissional(
        conn, cadastro.Profissional(ana.id, ana.nome, ana.funcao, "", "11999990000", True)
    )
    criticos = [
        a
        for a in mock.gerar_alertas_mock()
        if a.severidade is Severidade.CRITICA and a.status is Status.NOVO
    ]
    primeiro = criticos[0]
    outro = primeiro.__class__(
        **{**primeiro.para_dict(), "id": "outro", "descricao": "outra chamada"}
        | {
            "origem": primeiro.origem,
            "tipo": primeiro.tipo,
            "severidade": primeiro.severidade,
            "status": primeiro.status,
            "detectado_em": primeiro.detectado_em,
            "reconhecido_em": None,
        }
    )
    for alerta in (primeiro, outro):
        db.salvar(conn, alerta)
    agora = primeiro.detectado_em + timedelta(minutes=1)

    enviadas = notificacao.acionar(conn, [primeiro, outro], agora, {}, SIMULADO)
    assert len(enviadas) == 1  # mesmo leito: uma mensagem só

    db.reconhecer(conn, enviadas[0].alerta_id, por="Enf. Ana")
    restante = next(a for a in (primeiro, outro) if a.id != enviadas[0].alerta_id)
    assert len(notificacao.acionar(conn, [restante], agora, {}, SIMULADO)) == 1


def test_sem_ninguem_com_whatsapp_nada_e_enviado(conn):
    critico = _critico()

    assert notificacao.acionar(conn, [critico], critico.detectado_em, {}, SIMULADO) == []


def test_alerta_antigo_no_relogio_nao_aciona(conn):
    ana = cadastro.equipe(conn)[0]
    cadastro.salvar_profissional(
        conn, cadastro.Profissional(ana.id, ana.nome, ana.funcao, "", "11999990000", True)
    )
    critico = _critico()

    tarde = critico.detectado_em + timedelta(hours=2)
    assert notificacao.acionar(conn, [critico], tarde, {}, SIMULADO) == []


def test_mensagem_tem_sirene_paciente_e_link():
    texto = notificacao.mensagem(_critico(), "Leito UTI-04 · José", 1, "http://app")

    assert texto.startswith("🚨")
    assert "Leito UTI-04" in texto
    assert "http://app/alertas" in texto


class _RespostaFalsa:
    def __init__(self, corpo: bytes) -> None:
        self.corpo = corpo

    def read(self) -> bytes:
        return self.corpo

    def __enter__(self) -> _RespostaFalsa:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_sem_evolution_o_telegram_recebe_a_chamada_com_o_destinatario(monkeypatch):
    pedidos = []

    def urlopen_falso(pedido, timeout):
        pedidos.append(pedido)
        return _RespostaFalsa(b'{"ok": true, "result": {"message_id": 42}}')

    monkeypatch.setattr(notificacao.urllib.request, "urlopen", urlopen_falso)
    cfg = notificacao.ConfigWhatsApp("", "", "", "http://app", "TOKEN", "-100123")

    assert cfg.ativo and not cfg.evolution
    status, detalhe = notificacao.enviar_whatsapp(cfg, "5511999998888", "🚨 oi", "Dra. Ana")

    assert (status, detalhe) == ("enviado", "telegram:42")
    assert pedidos[0].full_url == "https://api.telegram.org/botTOKEN/sendMessage"
    corpo = json.loads(pedidos[0].data)
    assert corpo["chat_id"] == "-100123"
    assert corpo["text"].startswith("📱 Para: Dra. Ana (5511•••••8888)\n🚨 oi")


_ERRO_MARKDOWN = b'{"ok": false, "description": "Bad Request: can\'t parse entities"}'


def test_telegram_recusa_o_markdown_e_reenvia_sem_formatacao(monkeypatch):
    formatos = []

    def urlopen_falso(pedido, timeout):
        corpo = json.loads(pedido.data)
        formatos.append(corpo.get("parse_mode"))
        if "parse_mode" in corpo:
            raise notificacao.urllib.error.HTTPError(
                pedido.full_url, 400, "bad", Message(), io.BytesIO(_ERRO_MARKDOWN)
            )
        return _RespostaFalsa(b'{"ok": true, "result": {"message_id": 7}}')

    monkeypatch.setattr(notificacao.urllib.request, "urlopen", urlopen_falso)
    cfg = notificacao.ConfigWhatsApp("", "", "", "http://app", "TOKEN", "1")

    assert notificacao.enviar_telegram(cfg, "leito_04 *") == ("enviado", "telegram:7")
    assert formatos == ["Markdown", None]
