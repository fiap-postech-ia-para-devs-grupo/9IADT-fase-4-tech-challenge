from src.analyzers.audio import termos


def test_localiza_termo_respiratorio_exato():
    termo = termos.localizar("shortness of breath", "SymptomOrSign")
    assert termo is not None
    assert termo.termo == "shortness of breath"
    assert termo.categoria == "respiratorio"
    assert termo.entidade_categoria == "SymptomOrSign"


def test_localiza_termo_dentro_de_entidade_mais_longa():
    termo = termos.localizar("mild dyspnea on exertion", "SymptomOrSign")
    assert termo is not None
    assert termo.termo == "dyspnea"


def test_termo_fora_da_lista_de_categorias_da_fusao_fica_sem_categoria():
    termo = termos.localizar("fatigue", "SymptomOrSign")
    assert termo is not None
    assert termo.categoria is None


def test_termo_desconhecido_nao_casa():
    assert termos.localizar("broken arm cast", "SymptomOrSign") is None


def test_substring_solta_nao_casa_por_fronteira_de_palavra():
    # "pain" sozinho não deve casar com "chest pain" nem "back pain" (falso positivo de substring)
    assert termos.localizar("pain", "SymptomOrSign") is None


def test_casamento_ignora_maiusculas():
    termo = termos.localizar("Chest Pain", "SymptomOrSign")
    assert termo is not None
    assert termo.termo == "chest pain"
    assert termo.categoria == "dor"


def test_texto_vazio_nao_casa():
    assert termos.localizar("   ", "SymptomOrSign") is None


def test_plural_casa_com_o_termo_no_singular():
    assert termos.localizar("chest pains", "SymptomOrSign").termo == "chest pain"
    assert termos.localizar("headaches", "SymptomOrSign").termo == "headache"
    assert termos.localizar("palpitations", "SymptomOrSign").termo == "palpitation"


def test_variantes_vistas_nas_consultas_reais_casam():
    assert termos.localizar("can't breathe", "SymptomOrSign").categoria == "respiratorio"
    assert termos.localizar("breathless", "SymptomOrSign").categoria == "respiratorio"
    assert termos.localizar("nauseous", "SymptomOrSign").termo == "nauseous"
