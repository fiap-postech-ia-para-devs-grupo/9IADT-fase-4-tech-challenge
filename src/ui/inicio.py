"""Início: o painel do plantão (indicadores e gráficos) e, logo abaixo, como o sistema
funciona — fluxo multimodal, o que é real e o que é simulado e o passo a passo do plantão. O
tour guiado (barra superior) percorre o sistema inteiro.
"""

from __future__ import annotations

import streamlit as st

from src.ui import cabecalho, dashboard, navegacao, tour
from src.ui import plantao_estado as plantao
from src.ui.tour import PASSOS

FLUXO = """
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica"
  fontsize=11 color="#0b2545" fillcolor="#ffffff"]; edge [color="#5b6b7b"];
  subgraph cluster_fontes { label="Fontes ligadas na Admissão"; color="#d9e1ea"; fontsize=11;
    mon [label="🫀 Monitor multiparamétrico\\nMIMIC-III"];
    rx  [label="💊 Prescrição eletrônica\\nMIMIC-IV Demo"];
    aud [label="🎙️ Consulta gravada\\nPriMock57"];
    cam [label="📷 Câmera do leito /\\nsessão de fisioterapia"];
  }
  subgraph cluster_analisadores { label="Analisadores"; color="#d9e1ea"; fontsize=11;
    av [label="Sinais Vitais\\nregras · z-score · Isolation Forest\\n+ tendência do NEWS2"];
    ap [label="Prescrições\\nregras de Mudança\\nTerapêutica Atípica"];
    aa [label="Áudio\\nAzure Speech to Text\\nAzure Text Analytics\\n+ classificador acústico"];
    vv [label="Vídeo\\nYOLOv8-pose (postura)\\nYOLOv8 (objetos/áreas críticas)"];
  }
  alertas [label="Alertas\\n(Contrato único, SQLite)" fillcolor="#e3f2fd"];
  fusao [label="Fusão\\nAlertas Compostos" fillcolor="#fff3e0"];
  central [label="🚨 Central de Alertas\\nluz de chamada · WhatsApp\\nReconhecimento · evolução"
           fillcolor="#ffebee"];
  mon -> av; rx -> ap; aud -> aa; cam -> vv;
  av -> alertas; ap -> alertas; aa -> alertas; vv -> alertas;
  alertas -> fusao -> central; alertas -> central;
}
"""


def _conteudo() -> None:
    cabecalho.barra()
    st.title("Painel do plantão")
    dashboard.painel()


def _como_funciona() -> None:
    st.subheader("Como o sistema funciona")
    st.markdown(
        "Central de vigilância da unidade de internação: junta **sinais vitais, prescrições, "
        "áudio e vídeo** de cada leito, detecta sinais precoces de risco e **chama a equipe** "
        "— no painel, na luz do leito e no WhatsApp — antes que o paciente piore."
    )
    botao, _ = st.columns([1, 3])
    if botao.button("🧭 Fazer o tour do sistema", width="stretch"):
        tour.abrir()
    fluxo, dados = st.columns([3, 2], gap="large")
    with fluxo:
        st.markdown('<div class="secao">Como os dados viram chamado</div>', unsafe_allow_html=True)
        st.graphviz_chart(FLUXO, width="stretch")
    with dados:
        st.markdown(
            '<div class="secao">O que é dado real e o que é simulado</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "- **Reais e públicos:** sinais vitais de UTI (MIMIC-III), prescrições (MIMIC-IV "
            "Demo), consultas gravadas (PriMock57), vídeos de exercício e de queda (Wikimedia "
            "Commons, UR Fall), sons (AudioSet via AST). Modelos e métricas são medidos nesses "
            "dados.\n"
            "- **Composto:** cada paciente junta fontes de pessoas diferentes; nomes são "
            "fictícios (ADR 0001).\n"
            "- **Simulado:** o turno toca os registros em velocidade acelerada; câmera e "
            "microfone do leito podem ser gravações tocadas em tempo real.\n"
            "- **Ao vivo:** webcam e microfone deste computador e o envio de WhatsApp."
        )
    st.markdown('<div class="secao">Passo a passo do plantão</div>', unsafe_allow_html=True)
    colunas = st.columns(4)
    for i, (titulo, pagina_destino, texto) in enumerate(PASSOS):
        with colunas[i % 4].container(border=True, height=230):
            st.markdown(f"**{i + 1}. {titulo}**")
            st.caption(texto)
            if pagina_destino:
                navegacao.link(pagina_destino, f"Ir para {pagina_destino}")
    st.caption(
        "Projeto acadêmico (FIAP, Tech Challenge Fase 4). Apoio à decisão; não substitui a "
        "avaliação clínica."
    )


def pagina() -> None:
    st.fragment(_conteudo, run_every=plantao.intervalo())()


def pagina_ajuda() -> None:
    """Ajuda: como o sistema funciona, o que é real e o que é simulado, e o passo a passo."""
    cabecalho.barra(avancar=False)
    st.title("Ajuda")
    _como_funciona()
