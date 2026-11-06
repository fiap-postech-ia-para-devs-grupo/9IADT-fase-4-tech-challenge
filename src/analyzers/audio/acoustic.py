"""Camada acústica do Analisador de Áudio (ESTRATEGIA.md, seção 5): features acústicas sobre
o subset do Coswara, classificador (Regressão Logística e Random Forest) e inferência sobre a
fala do paciente (PriMock57 ou upload no Laboratório) → Alerta `alteracao_vocal`.

Pipeline (`scripts/treinar_acustico.py`):

    1. `carregar_metadados`     rótulo por pessoa (`bd`/`ftg` → sintomático, `healthy` → saudável)
    2. `montar_dataset`         features de cada gravação (`counting-normal.wav`, `vowel-a.wav`)
    3. `dividir_por_pessoa`     split estratificado sem misturar as duas gravações da mesma pessoa
    4. `treinar_avaliar`        Regressão Logística + Random Forest → AUC, F1, matriz de confusão
    5. `inferir`                probabilidade sobre um áudio qualquer (mesmas features)

As duas gravações (fala contada e vogal sustentada) são tratadas como amostras independentes
do mesmo rótulo, não concatenadas por pessoa: é o que permite aplicar o mesmo extrator de
features num áudio de fala livre na inferência (seção 7 do relatório tem a limitação disso —
treino e inferência vêm de tipos de fala diferentes).
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
# (stubs do pandas: itertuples() tipa linhas como tuple genérica, e indexação por máscara
# booleana `df[df["col"].isin(lista)]` às vezes infere Series em vez de DataFrame; o idioma
# pandas de acessar por nome/indexar por máscara é mais claro do que contornar o stub)

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.alerts import Alerta, Origem, Severidade, Tipo

ARQUIVOS_UTEIS = ("counting-normal.wav", "vowel-a.wav")
N_MFCC = 13
TAXA_AMOSTRAGEM = 16_000
N_DOBRAS_LIMIAR = 5

_NAMESPACE = uuid.UUID("3f7a9c1e-6b2d-4e5f-9a1c-8d4e6f0b2c7a")


# --- 1. features acústicas -----------------------------------------------------------------


def extrair_features(caminho_wav: Path) -> dict[str, float]:
    """MFCC (média/desvio), F0, jitter, shimmer, HNR (`parselmouth`), proporção de pausas e
    taxa de fala (proxy por transições silêncio→fala sobre a energia RMS)."""
    import librosa
    import parselmouth
    from parselmouth.praat import call

    y, sr = librosa.load(str(caminho_wav), sr=TAXA_AMOSTRAGEM, mono=True)
    duracao = len(y) / sr if sr else 0.0

    features: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc{i + 1}_media"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i + 1}_desvio"] = float(np.std(mfcc[i]))

    som = parselmouth.Sound(str(caminho_wav))
    pitch = som.to_pitch()
    f0 = pitch.selected_array["frequency"]
    f0_validos = f0[f0 > 0]
    features["f0_media"] = float(np.mean(f0_validos)) if f0_validos.size else 0.0
    features["f0_desvio"] = float(np.std(f0_validos)) if f0_validos.size else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parselmouth avisa quando o áudio é curto/silencioso
        try:
            pp = call(som, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([som, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            jitter = shimmer = float("nan")
    features["jitter_local"] = 0.0 if np.isnan(jitter) else float(jitter)
    features["shimmer_local"] = 0.0 if np.isnan(shimmer) else float(shimmer)

    harmonicidade = som.to_harmonicity_cc()
    valores_hnr = harmonicidade.values[harmonicidade.values != -200]
    features["hnr_media"] = float(np.mean(valores_hnr)) if valores_hnr.size else 0.0

    rms = librosa.feature.rms(y=y)[0]
    limiar_silencio = float(np.median(rms) * 0.3) if rms.size else 0.0
    silencio = rms < limiar_silencio
    features["proporcao_pausas"] = float(np.mean(silencio)) if silencio.size else 0.0
    transicoes = int(np.sum(np.diff(silencio.astype(int)) == -1))
    features["taxa_fala"] = transicoes / duracao if duracao > 0 else 0.0

    return features


# --- 2. metadados e dataset -----------------------------------------------------------------


def _rotulo(linha: pd.Series) -> int | None:
    """1 = sintomático (Breathing Difficulties ou Fatigue), 0 = saudável sem nenhum dos dois,
    None = fora do subset (ex.: positivo para COVID sem esses sintomas)."""
    sintomatico = str(linha.get("bd")) == "True" or str(linha.get("ftg")) == "True"
    saudavel = linha.get("covid_status") == "healthy" and not sintomatico
    if sintomatico:
        return 1
    if saudavel:
        return 0
    return None


def carregar_metadados(caminho_combined_csv: Path) -> pd.DataFrame:
    """`combined_data.csv` do Coswara-Data, com a coluna `rotulo` (1/0/NaN) adicionada."""
    bruto = pd.read_csv(caminho_combined_csv, dtype=str)
    bruto["rotulo"] = bruto.apply(_rotulo, axis=1)
    return bruto


def localizar_arquivos(pasta_extraida: Path) -> dict[str, dict[str, Path]]:
    """`{participant_id: {"counting-normal.wav": caminho, "vowel-a.wav": caminho}}` para quem
    `scripts/download_data.py coswara` já extraiu (ver `Extracted_data/<pasta>/<id>/`)."""
    resultado: dict[str, dict[str, Path]] = {}
    if not pasta_extraida.exists():
        return resultado
    for pasta_lote in pasta_extraida.iterdir():
        if not pasta_lote.is_dir():
            continue
        for pessoa in pasta_lote.iterdir():
            if not pessoa.is_dir():
                continue
            arquivos = {
                nome: caminho
                for nome in ARQUIVOS_UTEIS
                if (caminho := pessoa / nome).exists()
            }
            if arquivos:
                resultado[pessoa.name] = arquivos
    return resultado


def montar_dataset(metadados: pd.DataFrame, pasta_extraida: Path) -> pd.DataFrame:
    """Uma linha por gravação (não por pessoa): colunas `id`, `arquivo`, `rotulo` + features.

    Pessoas sem rótulo (`rotulo` nulo) ou sem os áudios extraídos ficam fora. Arquivos
    corrompidos/curtos demais para o `parselmouth` processar são pulados (raro, registrado
    como diferença entre pessoas disponíveis e linhas no dataset final).
    """
    disponiveis = localizar_arquivos(pasta_extraida)
    linhas: list[dict[str, Any]] = []
    for meta in metadados.itertuples():
        if pd.isna(meta.rotulo):
            continue
        arquivos = disponiveis.get(meta.id)
        if not arquivos:
            continue
        for nome, caminho in arquivos.items():
            try:
                features = extrair_features(caminho)
            except Exception:
                continue
            linhas.append({"id": meta.id, "arquivo": nome, "rotulo": int(meta.rotulo), **features})
    return pd.DataFrame(linhas)


def balancear_por_pessoa(metadados: pd.DataFrame, por_classe: int, seed: int = 0) -> pd.DataFrame:
    """Amostra até `por_classe` pessoas de cada rótulo (sem reposição, seed fixa)."""
    rng = np.random.default_rng(seed)
    partes = []
    for rotulo in (0, 1):
        pessoas = metadados.loc[metadados["rotulo"] == rotulo, "id"].drop_duplicates()
        n = min(por_classe, len(pessoas))
        escolhidas = rng.choice(pessoas.to_numpy(), size=n, replace=False).tolist()
        partes.append(metadados[metadados["id"].isin(escolhidas)])
    return pd.concat(partes, ignore_index=True)


COLUNAS_NAO_FEATURE = ("id", "arquivo", "rotulo")


def colunas_features(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in COLUNAS_NAO_FEATURE]


def dividir_por_pessoa(
    dataset: pd.DataFrame, teste: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado por pessoa: as duas gravações de uma pessoa nunca ficam uma no
    treino e outra no teste (vazaria informação do mesmo indivíduo entre os dois)."""
    from sklearn.model_selection import train_test_split

    pessoas = dataset[["id", "rotulo"]].drop_duplicates()
    treino_ids, teste_ids = train_test_split(
        pessoas["id"].tolist(), test_size=teste, stratify=pessoas["rotulo"], random_state=seed
    )
    treino = dataset[dataset["id"].isin(treino_ids)].reset_index(drop=True)
    teste_df = dataset[dataset["id"].isin(teste_ids)].reset_index(drop=True)
    return treino, teste_df


# --- 3. treino e avaliação -------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoModelo:
    nome: str
    auc: float
    f1: float
    matriz_confusao: tuple[tuple[int, int], tuple[int, int]]
    limiar: float


def _limiar_youden(y_teste: np.ndarray, probas: np.ndarray) -> float:
    """Limiar que maximiza sensibilidade + especificidade − 1 (estatística J de Youden)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, limiares = roc_curve(y_teste, probas)
    j = tpr - fpr
    return float(limiares[int(np.argmax(j))])


def treinar_avaliar(
    treino: pd.DataFrame, teste: pd.DataFrame, seed: int = 0
) -> tuple[dict[str, ResultadoModelo], dict[str, Any]]:
    """Treina os dois modelos em `treino` e avalia em `teste`. Devolve as métricas por modelo
    e os próprios modelos ajustados (para `salvar_modelo`/inferência).

    O limiar vem de previsões fora da dobra no treino (validação cruzada por pessoa), nunca do
    teste: escolhê-lo no teste infla o F1 e a matriz de confusão reportados.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    colunas = colunas_features(treino)
    X_treino, y_treino = treino[colunas], treino["rotulo"].to_numpy()
    X_teste, y_teste = teste[colunas], teste["rotulo"].to_numpy()

    modelos: dict[str, Any] = {
        "regressao_logistica": Pipeline(
            [("escala", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }

    resultados: dict[str, ResultadoModelo] = {}
    dobras = StratifiedGroupKFold(n_splits=N_DOBRAS_LIMIAR, shuffle=True, random_state=seed)
    for nome, modelo in modelos.items():
        probas_fora_da_dobra = np.asarray(
            cross_val_predict(
                modelo, X_treino, y_treino, groups=treino["id"], cv=dobras, method="predict_proba"
            )
        )
        limiar = _limiar_youden(y_treino, probas_fora_da_dobra[:, 1])
        modelo.fit(X_treino, y_treino)
        probas = modelo.predict_proba(X_teste)[:, 1]
        auc = roc_auc_score(y_teste, probas)
        preds = (probas >= limiar).astype(int)
        (tn, fp), (fn, tp) = confusion_matrix(y_teste, preds, labels=[0, 1]).tolist()
        resultados[nome] = ResultadoModelo(
            nome=nome,
            auc=float(auc),
            f1=float(f1_score(y_teste, preds)),
            matriz_confusao=((int(tn), int(fp)), (int(fn), int(tp))),
            limiar=limiar,
        )
    return resultados, {"modelos": modelos, "colunas": colunas}


# --- 4. inferência e Alerta -------------------------------------------------------------------


def salvar_modelo(
    nome_escolhido: str,
    resultados: dict[str, ResultadoModelo],
    contexto: dict,
    caminho: Path,
) -> None:
    import joblib

    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "nome": nome_escolhido,
            "modelo": contexto["modelos"][nome_escolhido],
            "colunas": contexto["colunas"],
            "limiar": resultados[nome_escolhido].limiar,
        },
        caminho,
    )


def carregar_modelo(caminho: Path) -> dict[str, Any]:
    import joblib

    return joblib.load(caminho)


def prever_com_features(features: dict[str, float], modelo_salvo: dict[str, Any]) -> float:
    """Probabilidade de Alteração Vocal a partir de features já extraídas (evita recalculá-las
    quando quem chama também precisa mostrar as features, como a Tela 3)."""
    linha = pd.DataFrame([features])[modelo_salvo["colunas"]]
    return float(modelo_salvo["modelo"].predict_proba(linha)[0, 1])


def inferir(caminho_wav: Path, modelo_salvo: dict[str, Any]) -> float:
    """Probabilidade de Alteração Vocal para um áudio qualquer (mesmas features do treino)."""
    return prever_com_features(extrair_features(caminho_wav), modelo_salvo)


def alerta_alteracao_vocal(
    paciente_id: str,
    arquivo: str,
    probabilidade: float,
    limiar: float,
    *,
    trecho: str | None = None,
    inicio: float = 0.0,
    detectado_em: datetime | None = None,
) -> Alerta | None:
    """`None` quando a probabilidade não passa do limiar (ESTRATEGIA.md, seção 5: "Alteração
    Vocal se a probabilidade passar do limiar")."""
    if probabilidade < limiar:
        return None
    evidencia: dict[str, object] = {
        "arquivo": arquivo,
        "inicio": inicio,
        "probabilidade": round(probabilidade, 3),
        "limiar": round(limiar, 3),
    }
    if trecho is not None:
        evidencia["trecho"] = trecho
    severidade = Severidade.ALTA if probabilidade >= 0.8 else Severidade.MEDIA
    return Alerta(
        id=str(uuid.uuid5(_NAMESPACE, f"{paciente_id}|{arquivo}|alteracao_vocal")),
        paciente_id=paciente_id,
        origem=Origem.AUDIO,
        tipo=Tipo.ALTERACAO_VOCAL,
        severidade=severidade,
        descricao=(
            "Voz com padrão de dificuldade respiratória/fadiga "
            f"(probabilidade {probabilidade:.2f})"
        ),
        evidencia=evidencia,
        detectado_em=detectado_em or datetime.now(),
    )
