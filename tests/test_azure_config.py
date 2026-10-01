import pytest

from src.analyzers.audio.azure import AzureConfig, ConfigAzureIncompleta, carregar_config

ENV_COMPLETO = {
    "AZURE_SPEECH_KEY": "chave-speech",
    "AZURE_SPEECH_REGION": "eastus",
    "AZURE_LANGUAGE_KEY": "chave-language",
    "AZURE_LANGUAGE_ENDPOINT": "https://fase4-language.cognitiveservices.azure.com/",
}


def test_carrega_as_quatro_variaveis_do_env():
    assert carregar_config(ENV_COMPLETO) == AzureConfig(
        speech_key="chave-speech",
        speech_region="eastus",
        language_key="chave-language",
        language_endpoint="https://fase4-language.cognitiveservices.azure.com/",
    )


def test_aponta_todas_as_variaveis_ausentes_ou_vazias():
    env = {**ENV_COMPLETO, "AZURE_SPEECH_KEY": "  "}
    del env["AZURE_LANGUAGE_ENDPOINT"]

    with pytest.raises(ConfigAzureIncompleta) as erro:
        carregar_config(env)

    assert erro.value.faltando == ["AZURE_SPEECH_KEY", "AZURE_LANGUAGE_ENDPOINT"]
    assert "AZURE_SPEECH_KEY" in str(erro.value)
    assert "AZURE_LANGUAGE_ENDPOINT" in str(erro.value)
