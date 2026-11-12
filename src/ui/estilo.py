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
@keyframes piscar {0%,100% {opacity:1; box-shadow:0 0 18px 6px rgba(255,23,68,.85)}
  50% {opacity:.35; box-shadow:0 0 4px 1px rgba(255,23,68,.3)}}
.luz-chamada {display:flex; align-items:center; gap:.8rem; background:#3b0d12; color:#fff;
  border:2px solid #ff1744; border-radius:10px; padding:.55rem 1rem; margin-bottom:.8rem}
.luz-chamada .lampada {width:22px; height:22px; border-radius:50%; background:#ff1744;
  animation: piscar 1s infinite; flex:none}
.luz-chamada .sirene {font-size:1.6rem; animation: piscar 1s infinite; border-radius:8px}
.luz-chamada .texto b {color:#ff8a80}
.luz-apagada {display:flex; align-items:center; gap:.6rem; color:#5b6b7b; font-size:.82rem;
  margin:-.4rem 0 .6rem}
.luz-apagada .lampada {width:12px; height:12px; border-radius:50%; background:#9e9e9e}
.sirene-barra {display:inline-flex; align-items:center; gap:.4rem; background:#13315c;
  padding:.15rem .6rem .15rem .35rem; border-radius:999px}
.sirene-barra .cupula {width:18px; height:14px; border-radius:9px 9px 3px 3px;
  background:#546e7a}
.sirene-barra .txt {font-size:.75rem; font-weight:700; text-transform:uppercase}
.sirene-barra.critico .cupula {background:#ff1744; animation: piscar-sirene .8s infinite;
  box-shadow:0 0 14px 5px rgba(255,23,68,.9)}
.sirene-barra.alto .cupula {background:#ffab00; animation: piscar-sirene 1.4s infinite;
  box-shadow:0 0 12px 4px rgba(255,171,0,.8)}
@keyframes piscar-sirene {0%,100% {opacity:1; filter:brightness(1.3)}
  50% {opacity:.15; box-shadow:none; filter:brightness(.6)}}
.lampada-leito {display:inline-block; width:14px; height:14px; border-radius:50%;
  background:#ff1744; animation: piscar 1s infinite; margin-left:.3rem}
.leito-livre {border:1px dashed #b0bec5; border-radius:14px; padding:.8rem 1rem; color:#607d8b;
  background:#fafbfc; margin-bottom:.4rem; min-height:86px}
.ponto {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.35rem;
  vertical-align:middle}
.ponto.livre {background:#2e7d32; box-shadow:0 0 0 3px rgba(46,125,50,.15)}
.ponto.ocupado {background:#d32f2f; box-shadow:0 0 0 3px rgba(211,47,47,.15)}
.cl-ocupacao {font-size:.72rem; color:#6b7b8b; font-weight:600}
.cl {position:relative; background:linear-gradient(180deg,#ffffff 0%,#fbfcfe 100%);
  border:1px solid #e3e9f0; border-radius:14px; padding:1rem 1.1rem .9rem;
  box-shadow:0 1px 2px rgba(16,42,67,.06), 0 6px 18px rgba(16,42,67,.06);
  border-top:5px solid var(--cor); transition:box-shadow .2s, transform .2s; margin-bottom:.4rem;
  min-height:330px; display:flex; flex-direction:column}
.cl .cl-rodape {margin-top:auto}
.cl.faixa {min-height:0; padding:.7rem 1rem; margin-bottom:.6rem}
.cl.faixa .cl-leito {font-size:1.2rem}
.cl.faixa:hover {transform:none}
.cl:hover {box-shadow:0 2px 4px rgba(16,42,67,.08), 0 12px 28px rgba(16,42,67,.12);
  transform:translateY(-1px)}
.cl-topo {display:flex; justify-content:space-between; align-items:flex-start}
.cl-leito {font-size:1.45rem; font-weight:800; color:#0b2545; letter-spacing:.3px; line-height:1}
.cl-unid {font-size:.72rem; text-transform:uppercase; letter-spacing:.8px; color:#7b8a99;
  margin-top:.2rem}
.cl-paciente {font-weight:700; color:#1b2b3a; margin-top:.7rem; font-size:1rem}
.cl-sub {color:#6b7b8b; font-size:.8rem; margin-top:.1rem}
.cl-news {display:inline-flex; align-items:center; gap:.4rem; margin-top:.55rem;
  font-size:.78rem; font-weight:700; color:var(--cor-news); background:var(--fundo-news);
  padding:.15rem .6rem; border-radius:999px}
.cl-rodape {border-top:1px solid #edf1f5; margin-top:.6rem; padding-top:.5rem}
.cl-ultimo {color:#6b7b8b; font-size:.76rem; margin-top:.35rem; line-height:1.3}
.sl {display:flex; flex-direction:column; align-items:center; gap:1px; min-width:44px}
.sl .cupula {width:30px; height:22px; border-radius:15px 15px 4px 4px; background:#cfd8dc;
  border:1px solid rgba(0,0,0,.08)}
.sl .base {width:40px; height:7px; border-radius:2px; background:#37474f}
.sl .txt {font-size:.6rem; font-weight:700; color:#90a4ae; text-transform:uppercase;
  letter-spacing:.5px; margin-top:2px}
.sl.critico .cupula {background:#ff1744; animation: piscar-sirene .8s infinite;
  box-shadow:0 0 22px 8px rgba(255,23,68,.85)}
.sl.critico .txt {color:#d50000}
.sl.alto .cupula {background:#ffab00; animation: piscar-sirene 1.4s infinite;
  box-shadow:0 0 18px 6px rgba(255,171,0,.7)}
.sl.alto .txt {color:#e65100}
.sl.grande .cupula {width:44px; height:32px; border-radius:22px 22px 6px 6px}
.sl.grande .base {width:56px; height:9px}
.kpis {display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:.7rem;
  margin:.2rem 0 1rem}
.kpi {background:#fff; border:1px solid #e3e9f0; border-left:5px solid var(--cor);
  border-radius:12px; padding:.7rem .9rem; box-shadow:0 1px 2px rgba(16,42,67,.05),
  0 4px 12px rgba(16,42,67,.05)}
.kpi-rot {font-size:.72rem; text-transform:uppercase; letter-spacing:.6px; color:#6b7b8b;
  font-weight:700}
.kpi-val {font-size:1.9rem; font-weight:800; color:var(--cor); line-height:1.15; margin-top:.2rem}
.kpi-sub {font-size:.75rem; color:#7b8a99}
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


ROTULO_LUZ = {"critico": "chamado", "alto": "atenção", "apagada": "estável"}


def sirene_leito(estado: str, grande: bool = False) -> str:
    """Luz do leito: vermelha piscando (chamado), âmbar (atenção) ou apagada (estável)."""
    classe = f"sl {estado}{' grande' if grande else ''}"
    return (
        f'<div class="{classe}" title="Luz de chamada do leito"><div class="cupula"></div>'
        f'<div class="base"></div><div class="txt">{ROTULO_LUZ[estado]}</div></div>'
    )
