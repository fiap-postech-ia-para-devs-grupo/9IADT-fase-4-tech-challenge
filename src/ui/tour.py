"""Tour guiado do sistema inteiro, aberto uma vez na primeira visita e pelo botão 🧭 da barra.

"Não exibir novamente" fica gravado no banco (tabela `preferencias`), então vale para as
próximas sessões neste computador. O botão da barra continua abrindo o tour quando quiser.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.ui import navegacao
from src.ui import plantao_estado as plantao

PASSOS = (
    (
        "🏥 Para que serve",
        "Ajuda",
        "Central de vigilância da unidade de internação. Junta sinais vitais, prescrições, áudio "
        "e vídeo de cada leito, detecta sinais precoces de risco e chama a equipe antes que o "
        "paciente piore. Os dados são públicos e os pacientes, fictícios e compostos.",
    ),
    (
        "👥 Configure a unidade",
        "Equipe e Leitos",
        "Cadastre leitos e profissionais. Informe o WhatsApp de quem recebe as chamadas de "
        "emergência (🔔) e envie uma mensagem de teste.",
    ),
    (
        "👤 Assuma o plantão",
        None,
        "Na barra lateral, escolha o plantonista: é quem reconhece os Alertas e assina a "
        "evolução. Ali fica também o relógio do plantão (▶ Iniciar, velocidade, ir para o minuto).",
    ),
    (
        "➕ Dê entrada no paciente",
        "Admissão",
        "Escolha um leito livre (bolinha verde) e ligue as fontes de monitoramento: monitor "
        "multiparamétrico, prescrição eletrônica, consulta gravada e câmera do leito.",
    ),
    (
        "🛏️ Acompanhe os leitos",
        "Mapa de Leitos",
        "Cada cartão mostra sinais atuais, NEWS2 e a luz do leito: apagada (estável), âmbar "
        "(atenção) ou vermelha piscando (chamado). Clique em Abrir prontuário.",
    ),
    (
        "🩺 Prontuário",
        "Prontuário",
        "Monitor de beira de leito, tendências com NEWS2, linha do tempo, medicações em uso, "
        "Alertas do paciente, evolução e alta.",
    ),
    (
        "📡 Câmera e microfone do leito",
        "Prontuário",
        "Na seção Câmera e microfone, marque vídeo e/ou áudio e inicie. O YOLOv8-pose vigia "
        "Quedas; o AST (AudioSet) e o Whisper reconhecem alarme, grito, gemido e o paciente "
        "chamando a enfermagem. Use a webcam/microfone ou uma gravação.",
    ),
    (
        "🚨 Sirene e WhatsApp",
        "Central de Alertas",
        "Instabilidade crítica, Queda, piora do NEWS2, grito ou chamado acendem a sirene da "
        "barra e a luz do leito e acionam a equipe no WhatsApp. Sem Reconhecimento em 5 min, "
        "escala para a equipe médica.",
    ),
    (
        "✋ Central de Alertas",
        "Central de Alertas",
        "Fila única, mais graves primeiro, com Alertas repetidos agrupados em episódios. Abra, "
        "veja a Evidência (frame, áudio, sinal), reconheça e registre a conduta: ela vai para "
        "a evolução do paciente.",
    ),
    (
        "🔀 Fusão multimodal",
        "Central de Alertas",
        "Quando modalidades diferentes se confirmam, nasce um Alerta Composto: queda de SpO2 + "
        "falta de ar na consulta, Queda + opioide, Queda + grito, PA/FC + anti-hipertensivo.",
    ),
    (
        "🔬 Laboratório de Análise",
        "Laboratório de Análise",
        "Envie um vídeo de sessão de fisioterapia (Desvio de Execução, Relatório de Sessão) ou "
        "um áudio de consulta (Azure Speech to Text e Text Analytics) para análise avulsa.",
    ),
)

_SCHEMA = "CREATE TABLE IF NOT EXISTS preferencias (chave TEXT PRIMARY KEY, valor TEXT)"


def _preferencia(conn: sqlite3.Connection, chave: str) -> str | None:
    conn.execute(_SCHEMA)
    linha = conn.execute("SELECT valor FROM preferencias WHERE chave = ?", (chave,)).fetchone()
    return linha[0] if linha else None


def _gravar(conn: sqlite3.Connection, chave: str, valor: str) -> None:
    conn.execute(_SCHEMA)
    with conn:
        conn.execute("INSERT OR REPLACE INTO preferencias VALUES (?, ?)", (chave, valor))


def _fechar(nao_exibir: bool) -> None:
    if nao_exibir:
        _gravar(plantao.conexao(), "tour_oculto", "1")
    st.session_state.update(tour_passo=0, tour_aberto=False)
    st.rerun()


@st.dialog("🧭 Tour do sistema", width="large")
def _dialogo() -> None:
    passo = st.session_state.get("tour_passo", 0)
    titulo, pagina, texto = PASSOS[passo]
    st.progress((passo + 1) / len(PASSOS), text=f"Passo {passo + 1} de {len(PASSOS)}")
    st.markdown(f"### {titulo}")
    st.write(texto)
    if pagina:
        st.caption(f"Onde: menu › {pagina}")
    nao_exibir = st.checkbox(
        "Não exibir novamente ao abrir o sistema",
        value=_preferencia(plantao.conexao(), "tour_oculto") == "1",
        key="tour_nao_exibir",
        help="O botão 🧭 Tour na barra superior continua abrindo o tour.",
    )
    voltar, pular, avancar = st.columns(3)
    if passo > 0 and voltar.button("← Anterior", width="stretch"):
        st.session_state["tour_passo"] = passo - 1
        st.rerun()
    if pular.button("Fechar", width="stretch"):
        _fechar(nao_exibir)
    if passo < len(PASSOS) - 1:
        if avancar.button("Próximo →", type="primary", width="stretch"):
            st.session_state["tour_passo"] = passo + 1
            st.rerun()
    elif avancar.button("Concluir", type="primary", width="stretch"):
        _fechar(nao_exibir)


def abrir() -> None:
    st.session_state.update(tour_aberto=True, tour_passo=0)
    st.rerun()


def mostrar_se_pedido() -> None:
    """Chamado pelo `app.py`: abre o tour uma vez na primeira visita (se não foi ocultado) ou
    quando pedido pelo botão da barra."""
    if not st.session_state.get("tour_visto"):
        st.session_state["tour_visto"] = True
        if _preferencia(plantao.conexao(), "tour_oculto") != "1":
            st.session_state.update(tour_aberto=True, tour_passo=0)
    if st.session_state.get("tour_aberto"):
        _dialogo()


__all__ = ["PASSOS", "abrir", "mostrar_se_pedido", "navegacao"]
