"""Administração da unidade: equipe (com WhatsApp para a luz de chamada), leitos e histórico de
acionamentos. É onde o plantão é configurado antes de admitir pacientes."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
import streamlit as st

from src.core import cadastro, notificacao
from src.ui import cabecalho
from src.ui import plantao_estado as plantao


def _slug(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sem_acento.lower()).strip("-")


def _equipe() -> None:
    conn = plantao.conexao()
    equipe = cadastro.equipe(conn)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Profissional": p.rotulo,
                    "Função": p.funcao,
                    "Registro": p.registro,
                    "WhatsApp": notificacao.mascarar(notificacao.numero_valido(p.telefone) or "")
                    or "—",
                    "Recebe chamadas": "🔔 sim" if p.recebe_alertas else "—",
                }
                for p in equipe
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    novo, editar = st.columns(2, gap="large")
    with novo.form("novo_profissional", clear_on_submit=True, border=True):
        st.markdown("**Cadastrar profissional**")
        nome = st.text_input("Nome completo")
        funcao = st.selectbox("Função", cadastro.FUNCOES)
        registro = st.text_input("Registro profissional", placeholder="COREN-SP 123456")
        telefone = st.text_input("WhatsApp", placeholder="(11) 99999-9999")
        recebe = st.checkbox("Recebe chamadas de emergência no WhatsApp", value=True)
        if st.form_submit_button("Cadastrar", type="primary"):
            if not nome.strip():
                st.error("Informe o nome.")
            elif telefone and not notificacao.numero_valido(telefone):
                st.error("WhatsApp inválido: use DDD + número, ex.: (11) 99999-9999.")
            else:
                cadastro.salvar_profissional(
                    conn,
                    cadastro.Profissional(
                        _slug(nome),
                        nome.strip(),
                        funcao,
                        registro.strip(),
                        notificacao.numero_valido(telefone) or "",
                        recebe and bool(telefone),
                    ),
                )
                st.success(f"{nome} cadastrado(a).")
                st.rerun()

    with editar.container(border=True):
        st.markdown("**Editar WhatsApp / chamadas**")
        if not equipe:
            st.caption("Nenhum profissional cadastrado.")
            return
        escolhido = st.selectbox(
            "Profissional", equipe, format_func=lambda p: p.rotulo, key="adm_editar"
        )
        telefone = st.text_input(
            "WhatsApp",
            value=escolhido.telefone,
            key=f"adm_tel_{escolhido.id}",
            placeholder="(11) 99999-9999",
        )
        recebe = st.checkbox(
            "Recebe chamadas de emergência",
            value=escolhido.recebe_alertas,
            key=f"adm_recebe_{escolhido.id}",
        )
        salvar, remover = st.columns(2)
        if salvar.button("Salvar", type="primary", key="adm_salvar", width="stretch"):
            numero = notificacao.numero_valido(telefone) if telefone else ""
            if telefone and not numero:
                st.error("WhatsApp inválido.")
            else:
                cadastro.salvar_profissional(
                    conn,
                    cadastro.Profissional(
                        escolhido.id,
                        escolhido.nome,
                        escolhido.funcao,
                        escolhido.registro,
                        numero or "",
                        recebe and bool(numero),
                    ),
                )
                st.rerun()
        if remover.button("Desativar", key="adm_remover", width="stretch"):
            cadastro.desativar_profissional(conn, escolhido.id)
            st.rerun()


def _leitos() -> None:
    conn = plantao.conexao()
    ocupacao = {i.leito_id: i for i in cadastro.internacoes(conn)}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Leito": leito.id,
                    "Unidade": leito.unidade,
                    "Tipo": leito.tipo,
                    "Situação": f"🛏️ {ocupacao[leito.id].nome}" if leito.id in ocupacao else "livre",
                }
                for leito in cadastro.leitos(conn)
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.form("novo_leito", clear_on_submit=True):
        identificador, unidade, tipo = st.columns(3)
        id_leito = identificador.text_input("Identificação", placeholder="UTI-05")
        nome_unidade = unidade.text_input("Unidade", placeholder="UTI adulto")
        tipo_leito = tipo.selectbox("Tipo", cadastro.TIPOS_LEITO)
        if st.form_submit_button("Cadastrar leito", type="primary"):
            if not id_leito.strip():
                st.error("Informe a identificação do leito.")
            else:
                cadastro.salvar_leito(
                    conn,
                    cadastro.Leito(id_leito.strip().upper(), nome_unidade.strip(), tipo_leito),
                )
                st.rerun()


def _whatsapp() -> None:
    conn = plantao.conexao()
    cfg = notificacao.config()
    if cfg.ativo:
        st.success(f"Chamadas conectadas: {cfg.canal}.")
    else:
        st.warning(
            "Nenhum canal configurado: as chamadas ficam **simuladas** (registradas, sem envio). "
            "Preencha `EVOLUTION_URL`, `EVOLUTION_APIKEY` e `EVOLUTION_INSTANCE` (WhatsApp) ou "
            "`TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` (Telegram) no `.env`."
        )
    # Chave própria + callback: o valor sobrevive à troca de tela (widget fora da tela some).
    st.toggle(
        "Acionar a equipe no WhatsApp durante o plantão",
        value=st.session_state.get("whatsapp_ativo", True),
        key="whatsapp_toggle",
        help="Desligue para ensaiar a demonstração sem enviar mensagens.",
        on_change=lambda: st.session_state.update(
            whatsapp_ativo=st.session_state["whatsapp_toggle"]
        ),
    )
    st.caption(
        f"Quem é acionado: profissionais com 🔔. Alerta crítico ou Queda aciona o plantão; sem "
        f"Reconhecimento em {notificacao.MINUTOS_ESCALONAMENTO} min do plantão, escala para a "
        "equipe médica. Cada chamada vai uma vez por pessoa."
    )

    destinatarios = notificacao.destinatarios(cadastro.equipe(conn), 1)
    if destinatarios and st.button("🚨 Enviar mensagem de teste", key="adm_teste"):
        for profissional in destinatarios:
            numero = notificacao.numero_valido(profissional.telefone) or ""
            status, detalhe = notificacao.enviar_whatsapp(
                cfg,
                numero,
                "🚨 [TESTE] Luz de chamada do Monitoramento Multimodal configurada para "
                f"{profissional.rotulo}. Demonstração acadêmica (FIAP).",
                profissional.rotulo,
            )
            st.write(f"{profissional.rotulo} ({notificacao.mascarar(numero)}): {status}")

    historico = notificacao.historico(conn)
    st.markdown('<div class="secao">Histórico de acionamentos</div>', unsafe_allow_html=True)
    if not historico:
        st.caption("Nenhum acionamento ainda.")
    else:
        tabela = pd.DataFrame(historico).drop(columns=["alerta_id", "detalhe"])
        tabela["numero"] = tabela["numero"].map(notificacao.mascarar)
        tabela["nivel"] = tabela["nivel"].map(lambda n: {1: "plantão", 2: "⏫ escalonado"}[n])
        st.dataframe(tabela, hide_index=True, width="stretch")
    if st.button("Limpar histórico (reapresentar a demonstração)", key="adm_limpar"):
        notificacao.limpar_historico(conn)
        st.rerun()


def pagina() -> None:
    cabecalho.barra(avancar=False)
    st.title("Equipe e Leitos")
    secao = st.segmented_control(
        "Seção",
        ["👥 Equipe", "🛏️ Leitos", "🚨 Chamadas (WhatsApp)"],
        default="👥 Equipe",
        key="adm_secao",
        label_visibility="collapsed",
    )
    if secao == "🛏️ Leitos":
        _leitos()
    elif secao == "🚨 Chamadas (WhatsApp)":
        _whatsapp()
    else:
        _equipe()
