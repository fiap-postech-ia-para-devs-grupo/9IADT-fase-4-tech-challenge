"""Credenciais dos recursos Azure (Speech e Language, tier F0) lidas do `.env`.

Setup das contas, região e cotas em `docs/azure-setup.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureConfig:
    speech_key: str
    speech_region: str
    language_key: str
    language_endpoint: str


VARIAVEIS = (
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION",
    "AZURE_LANGUAGE_KEY",
    "AZURE_LANGUAGE_ENDPOINT",
)


class ConfigAzureIncompleta(Exception):
    def __init__(self, faltando: list[str]) -> None:
        self.faltando = faltando
        super().__init__(
            f"Variáveis Azure ausentes ou vazias no .env: {', '.join(faltando)} "
            "(ver .env.example e docs/azure-setup.md)"
        )


def carregar_config(env: Mapping[str, str]) -> AzureConfig:
    valores = {nome: env.get(nome, "").strip() for nome in VARIAVEIS}
    faltando = [nome for nome, valor in valores.items() if not valor]
    if faltando:
        raise ConfigAzureIncompleta(faltando)
    return AzureConfig(
        speech_key=valores["AZURE_SPEECH_KEY"],
        speech_region=valores["AZURE_SPEECH_REGION"],
        language_key=valores["AZURE_LANGUAGE_KEY"],
        language_endpoint=valores["AZURE_LANGUAGE_ENDPOINT"],
    )
