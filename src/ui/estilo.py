"""Identidade visual do sistema: CSS da interface hospitalar e blocos HTML reutilizáveis.

Cores por severidade seguem a convenção de monitorização (vermelho crítico, laranja alto,
amarelo médio, verde baixo). Todo texto que entra no HTML passa por `html.escape`.
"""

from __future__ import annotations

import html

import streamlit as st

from src.core.alerts import Severidade

COR_SEVERIDADE = {
    Severidade.CRITICA: "#d32f2f",
    Severidade.ALTA: "#ef6c00",
    Severidade.MEDIA: "#f9a825",
    Severidade.BAIXA: "#2e7d32",
}
COR_RISCO = {"baixo": "#2e7d32", "baixo-médio": "#f9a825", "médio": "#ef6c00", "alto": "#d32f2f"}
COR_PONTOS = {0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#d32f2f"}
# Nos blocos escuros do monitor, as mesmas faixas em tons claros (contraste sobre o fundo).
COR_MONITOR = {0: "#69f0ae", 1: "#ffe57f", 2: "#ffab40", 3: "#ff5252"}
COR_MONITOR_RISCO = {
    "baixo": "#69f0ae",
    "baixo-médio": "#ffe57f",
    "médio": "#ffab40",
    "alto": "#ff5252",
}

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
h1 {font-size: 1.6rem !important; font-weight: 700 !important; color: #0b2545;}
h2, h3 {color: #0b2545;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
.barra-sistema {display:flex; align-items:center; gap:1.2rem; background:#0b2545; color:#e8eef5;
  padding:.5rem 1rem; border-radius:8px; margin-bottom:.8rem; font-size:.85rem;
  flex-wrap:wrap; white-space:nowrap}
.barra-sistema .marca {font-weight:700; font-size:1rem; color:#fff; letter-spacing:.3px;}
.barra-sistema .item {opacity:.9}
.barra-sistema .direita {margin-left:auto; display:flex; gap:.6rem; align-items:center}
.chip {display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; color:#fff; white-space:nowrap}
.chip-claro {background:#e8eef5; color:#1b2b3a}
.relogio {font-family: ui-monospace, Menlo, Consolas, monospace; font-size:1.05rem;
  background:#13315c; padding:.15rem .6rem; border-radius:6px; color:#fff}
.leito {border:1px solid #d9e1ea; border-radius:10px; background:#fff; padding:.8rem 1rem;
  border-left:8px solid var(--cor); margin-bottom:.3rem}
.leito .topo {display:flex; justify-content:space-between; align-items:center; gap:.4rem;
  flex-wrap:wrap}
.leito .nome {font-weight:700; font-size:1.15rem; color:#0b2545; white-space:nowrap}
.leito .sub {color:#5b6b7b; font-size:.82rem}
.sinais {display:grid; grid-template-columns:repeat(4,1fr); gap:.4rem; margin:.6rem 0}
.sinal {background:#0f1b2b; border-radius:6px; padding:.35rem .5rem; color:#cfd8e3}
.sinal .rot {font-size:.68rem; text-transform:uppercase; letter-spacing:.5px; opacity:.8}
.sinal .val {font-family: ui-monospace, Menlo, Consolas, monospace; font-size:1.35rem;
  font-weight:700; color:var(--cor)}
.sinal .uni {font-size:.7rem; opacity:.7}
.monitor {display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; margin:.4rem 0 1rem}
.monitor .sinal {padding:.7rem .8rem}
.monitor .sinal .val {font-size:2.3rem}
.monitor .sinal .tend {font-size:.75rem; opacity:.75}
.alerta-cartao {border:1px solid #d9e1ea; border-left:6px solid var(--cor); border-radius:8px;
  background:#fff; padding:.5rem .75rem}
.alerta-cartao .linha1 {display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;
  font-size:.82rem; color:#5b6b7b}
.alerta-cartao .desc {font-weight:600; color:#1b2b3a; margin-top:.2rem; font-size:.95rem}
.secao {font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; color:#5b6b7b;
  font-weight:700; margin:.6rem 0 .3rem}
</style>
"""


def aplicar() -> None:
    st.html(CSS)


def e(texto: object) -> str:
    return html.escape(str(texto))


def chip(texto: str, cor: str) -> str:
    return f'<span class="chip" style="background:{cor}">{e(texto)}</span>'


def chip_claro(texto: str) -> str:
    return f'<span class="chip chip-claro">{e(texto)}</span>'


def bloco_sinal(rotulo: str, valor: str, unidade: str, cor: str, tendencia: str = "") -> str:
    tend = f'<div class="tend">{e(tendencia)}</div>' if tendencia else ""
    return (
        f'<div class="sinal" style="--cor:{cor}"><div class="rot">{e(rotulo)}</div>'
        f'<span class="val">{e(valor)}</span> <span class="uni">{e(unidade)}</span>{tend}</div>'
    )
