"""`analyze_audio`: função chamável pela Tela 3 — Laboratório de Análise (ESTRATEGIA.md,
seções 5 e 9). Roda a camada semântica (Azure STT + TA for Health + Sentiment) e a camada
acústica (classificador local) sobre um áudio qualquer e devolve tudo que a tela mostra —
transcrição, Termos Críticos, Sentimento, métricas acústicas e probabilidade de Alteração
Vocal — junto com os Alertas já no Contrato, prontos para `db.salvar`.

    from src.analyzers.audio.lab import analyze_audio
    resultado = analyze_audio(Path("gravacao.wav"), config)
    for alerta in resultado.alertas:
        db.salvar(conn, alerta)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.analyzers.audio import acoustic, semantic
from src.analyzers.audio.azure import AzureConfig
from src.core.alerts import Alerta

_RAIZ = Path(__file__).resolve().parents[3]
CACHE_DIR_PADRAO = _RAIZ / "data" / "audio" / "cache_laboratorio"
MODELO_PADRAO = _RAIZ / "data" / "audio" / "modelo_acustico.joblib"


@dataclass(frozen=True)
class ResultadoLaboratorio:
    arquivo: str
    transcricao: list[semantic.Trecho]
    sentimentos: list[semantic.SentimentoTrecho]
    features_acusticas: dict[str, float]
    probabilidade_alteracao_vocal: float | None
    alertas: list[Alerta]

    @property
    def texto_completo(self) -> str:
        return " ".join(t.texto for t in self.transcricao)


def analyze_audio(
    arquivo: Path,
    config: AzureConfig,
    *,
    paciente_id: str = "avulso",
    cache_dir: Path = CACHE_DIR_PADRAO,
    caminho_modelo: Path = MODELO_PADRAO,
) -> ResultadoLaboratorio:
    """Pipeline completo de um upload de áudio (qualquer formato que o `ffmpeg` leia).

    `caminho_modelo` ausente (classificador acústico ainda não treinado, ver
    `scripts/treinar_acustico.py`) não impede o resto: `probabilidade_alteracao_vocal` fica
    `None` e só os Alertas de Termo Crítico são gerados.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    identificador = arquivo.stem

    audio_pronto = semantic.converter_wav(arquivo, cache_dir / f"{identificador}_16k.wav")

    transcricao = semantic.transcrever(audio_pronto, config, cache_dir)
    entidades = semantic.extrair_entidades_saude(transcricao, config, cache_dir, identificador)
    alertas_termo = semantic.termos_criticos(
        transcricao, entidades, paciente_id, identificador, agora
    )
    sentimentos = semantic.analisar_sentimento(transcricao, config, cache_dir, identificador)

    features = acoustic.extrair_features(audio_pronto)
    probabilidade: float | None = None
    alertas_vocal: list[Alerta] = []
    if caminho_modelo.exists():
        modelo_salvo = acoustic.carregar_modelo(caminho_modelo)
        probabilidade = acoustic.prever_com_features(features, modelo_salvo)
        alerta = acoustic.alerta_alteracao_vocal(
            paciente_id,
            identificador,
            probabilidade,
            modelo_salvo["limiar"],
            trecho=" ".join(t.texto for t in transcricao) or None,
            detectado_em=agora,
        )
        if alerta is not None:
            alertas_vocal.append(alerta)

    return ResultadoLaboratorio(
        arquivo=arquivo.name,
        transcricao=transcricao,
        sentimentos=sentimentos,
        features_acusticas=features,
        probabilidade_alteracao_vocal=probabilidade,
        alertas=[*alertas_termo, *alertas_vocal],
    )
