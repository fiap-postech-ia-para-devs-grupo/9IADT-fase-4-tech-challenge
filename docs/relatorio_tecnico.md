# Relatório Técnico — Tech Challenge Fase 4

> Esqueleto criado no #5. Cada dono escreve a sua seção até o feature freeze (04/11/2026); Renato revisa e costura.
> Vocabulário oficial em [CONTEXT.md](../CONTEXT.md). Estrutura definida em [ESTRATEGIA.md](../ESTRATEGIA.md), seção 13.

| Seção | Dono | Issue |
| --- | --- | --- |
| 1. Contexto | Renato | #17 |
| 2. Fluxo Multimodal | Antonio | #11 |
| 3.1 Vídeo | Marcelo | #6, #13 |
| 3.2 Áudio | V. Geizler | #9, #10, #14 |
| 3.3 Sinais Vitais | Antonio | #7 |
| 3.4 Prescrições | V. Blasque | #8 |
| 4. Fusão e Pacientes de Demonstração | V. Blasque | #12 |
| 5. Resultados e Exemplos de Anomalias Detectadas | cada dono, nas suas seções; Renato consolida | #17 |
| 6. Integração com Azure | V. Geizler | #14, #21 |
| 7. Limitações | Renato, com insumo de cada dono | #17 |
| 8. Conclusão e Trabalhos Futuros | Renato | #17 |

## 1. Contexto (continuidade com o assistente da Fase 3)

_Dono: Renato._ <!-- TODO -->

## 2. Fluxo Multimodal

_Dono: Antonio._ Diagrama: fontes → Analisadores → Alertas → Fusão → Central de Alertas. <!-- TODO -->

## 3. Modelos por Tipo de Dado

### 3.1 Vídeo — YOLOv8-pose, Desvio de Execução, Queda

_Dono: Marcelo._ <!-- TODO -->

### 3.2 Áudio — Azure STT + TA for Health + classificador acústico

_Dono: V. Geizler._ <!-- TODO -->

### 3.3 Sinais Vitais — regras × z-score × Isolation Forest

_Dono: Antonio._

**Dado.** 10 registros *numerics* do MIMIC-III Waveform Database (`mimic3wdb/1.0`, acesso aberto), lidos com `wfdb` e reamostrados para 1 minuto (`src/analyzers/vitals/load.py`). Cada registro tem entre 100 e 330 minutos. Foram escolhidos pelo critério: até 50% de lacunas em FC, FR, SpO2 e PAS, e pelo menos 3 janelas limpas de 40 minutos sem lacuna. Lacunas de até 5 minutos são interpoladas; as maiores ficam como ausência de dado. Os registros são curtos e irregulares, o que limita a amostra (ver seção 7).

**Técnicas** (`src/analyzers/vitals/detectors.py`, mesma interface `detect(janela) -> Alerta[]`):

- **Regras clínicas:** SpO2 < 90 (crítica), FC > 120 ou < 45, PAS > 180 ou < 90, FR > 25.
- **z-score móvel:** |x − média| > 3σ sobre os 60 minutos anteriores, com desvio mínimo de 5 bpm (FC), 3 rpm (FR), 2 pontos (SpO2) e 10 mmHg (PAS). O desvio mínimo evita que séries muito estáveis gerem falsos positivos.
- **Isolation Forest:** sobre médias móveis de 5 minutos das quatro séries, 200 árvores, contaminação de 1%.

**Avaliação** (`scripts/avaliar_vitals.py`). Em cada registro, injetamos uma Anomalia de cada tipo (pico de FC, queda gradual de SpO2, hipotensão súbita) com 20 minutos de duração, fora de trechos já marcados pelas regras e pelo z-score. Foram 23 injeções no total, pois nem todos os tipos couberam em todos os registros. Um Alerta cobre uma injeção quando a série bate e os intervalos se cruzam, com tolerância de 5 minutos.

| Técnica | Recall | Precisão | Alertas | Atraso mediano |
| --- | --- | --- | --- | --- |
| Regras clínicas | 1,00 (23/23) | 0,41 | 64 | 0 min |
| z-score móvel | 0,70 (16/23) | 0,19 | 90 | 0 min |
| Isolation Forest | 0,48 (11/23) | 0,61 | 18 | 18 min |

Resultados em `results/vitals/avaliacao.csv` (agregado) e `results/vitals/avaliacao_por_registro.csv`.

**Leitura dos números.** As regras acertam todas as injeções porque os eventos foram desenhados para cruzar exatamente os limites clínicos, então esse recall é otimista. A precisão baixa das regras não é só erro: os Alertas "falsos" contam Anomalias reais que já estavam nos registros (ex.: SpO2 de 76% no registro 3000100n). O z-score detecta mais tipos de desvio, mas gera muitos alertas sobre variações pequenas. O Isolation Forest é o mais preciso, com menos alertas, mas detecta menos e com atraso de cerca de 18 minutos, porque só marca um padrão conjunto depois que a média móvel se afasta da história.

No replay (`scripts/replay_vitals.py`), os 10 registros geraram 239 Alertas `sinal_vital` gravados no SQLite.

### 3.4 Prescrições — regras sobre MIMIC-IV Demo

_Dono: V. Blasque._

**Dado.** MIMIC-IV Clinical Database Demo 2.2 (ODbL, acesso aberto): `hosp/prescriptions` (18.087 linhas, 100 pacientes, 250 internações com prescrição) e `hosp/emar`, mais `admissions` para a alta de cada internação. Só entram prescrições de terapia (`drug_type = MAIN`), sem as soluções de manutenção de acesso (_flush_/_lock_); prescrição sem `stoptime` termina na alta. `scripts/download_data.py mimic4demo` baixa tudo; os dados brutos não vão para o git.

**Técnica.** Regras determinísticas, sem aprendizado: o MIMIC não traz rótulo de "prescrição atípica", então não há precisão/recall a medir. O valor está em regras auditáveis, cada Alerta citando a regra e os dados que o dispararam (`evidencia`). Isolation Forest ficou como trabalho futuro (ESTRATEGIA.md, seção 7).

| Regra | Dispara quando | Severidade |
| --- | --- | --- |
| `salto_de_dose` | mesma droga, via e unidade, dose nova > 1,5× a anterior com início em < 24h | alta se a droga é de alta vigilância, senão média |
| `alta_vigilancia` | primeira prescrição de cada classe de alta vigilância (anticoagulante, insulina, opioide, sedativo) na internação | alta para opioide e anticoagulante, média nas demais |
| `trocas_excessivas` | mais de 4 trocas de princípio ativo dentro da mesma classe em 48h (um Alerta por episódio) | baixa |
| `duplicidade` | dois princípios ativos diferentes da mesma classe com vigências sobrepostas (um Alerta por par e internação) | alta se envolve alta vigilância, senão média |

**Listas versionadas** (`src/analyzers/prescriptions/listas.py`, `VERSAO = "1.0.0"`; toda Evidência registra a versão): drogas de alta vigilância, 10 classes terapêuticas para duplicidade/trocas, lista de reposição de eletrólitos e lista de rotina. A unidade de comparação é o princípio ativo, então metoprolol _tartrate_ e _succinate_ não são duplicidade.

**Decisões de projeto** (todas declaradas aqui porque mudam a contagem):

- Reposição de eletrólitos e glicose (Mg, Ca, K, fosfato, NaCl, bicarbonato, dextrose) fica fora de `salto_de_dose`: é titulada por exame, então a dose oscila por rotina. Sem essa exclusão, a regra gerava 674 Alertas (93% dos pacientes), dominados por magnésio, cálcio e potássio; com ela, 444.
- Heparina + varfarina (ponte de anticoagulação), insulina basal + rápida e AAS + clopidogrel são esquemas padrão e não entram em `duplicidade`.
- "Suspensão" não entra em `trocas_excessivas`: o MIMIC só registra o `stoptime` prescrito, não o momento real da suspensão. Conta-se a troca de princípio ativo dentro da classe.
- O eMAR confirma `alta_vigilancia`: a Evidência traz `primeira_administracao`, a primeira dose de fato administrada.
- Os ids dos Alertas são determinísticos (UUIDv5), então reprocessar não duplica nada no SQLite.

**Resultado nos 100 pacientes** (`results/prescriptions/contagem_por_regra.csv`, listas v1.0.0):

| Regra | Alertas | Pacientes com ao menos 1 |
| --- | --- | --- |
| `salto_de_dose` | 444 | 81 |
| `alta_vigilancia` | 652 | 99 |
| `trocas_excessivas` | 16 | 11 |
| `duplicidade` | 243 | 69 |
| **Total** | **1.355** | |

**Linhas do tempo** (`results/prescriptions/linha_do_tempo_*.html`, gráfico interativo com as prescrições monitoradas e os Alertas sobre a droga de cada um): internações com mais regras distintas disparadas.

- Paciente 10014729, internação 28889419: os quatro tipos de Alerta, por exemplo "Duplicidade terapêutica (opioide): Meperidine e Oxycodone-Acetaminophen ativas ao mesmo tempo" e "5 trocas de medicação em 48h".
- Paciente 10032725, internação 25177949: "Aumento de dose > 50% em < 24h: HYDROmorphone (Dilaudid) 2 → 6 mg" e "Duplicidade terapêutica (opioide): HYDROmorphone e Oxycodone SR ativas ao mesmo tempo".

**Leitura dos números e limitações.** O demo é majoritariamente de UTI, então quase todo paciente inicia uma droga de alta vigilância (99%): a regra mostra o mecanismo, não discrimina risco, e numa enfermaria teria taxa bem menor. Os limiares (1,5×, 24h, 4 trocas, 48h) vêm da ESTRATEGIA.md ou foram escolhidos olhando a distribuição do próprio demo; não houve validação clínica. As listas de drogas são curadas à mão para o escopo do trabalho, não um formulário completo. As datas do MIMIC são deslocadas (anos 21xx); o alinhamento com as outras modalidades é feito pelos offsets dos Pacientes de Demonstração (seção 4).

**Como rodar.**

```bash
uv run python scripts/download_data.py mimic4demo
uv run python -m scripts.analisar_prescricoes --gravar   # contagem, linhas do tempo e Alertas no SQLite
uv run pytest tests/test_prescriptions.py
```

## 4. Fusão e Pacientes de Demonstração

_Dono: V. Blasque._ Declarar que os pacientes são compostos para demonstração ([ADR 0001](adr/0001-paciente-de-demonstracao-composto.md)). <!-- TODO -->

## 5. Resultados e Exemplos de Anomalias Detectadas

_Dono: cada um nas suas seções; Renato consolida._ <!-- TODO -->

## 6. Integração com Azure (serviços, custos, cotas)

_Dono: V. Geizler._ Ver [azure-setup.md](azure-setup.md). <!-- TODO -->

## 7. Limitações (datasets distintos, pacientes compostos, dados em inglês)

_Dono: Renato, com insumo de cada dono._ <!-- TODO -->

## 8. Conclusão e Trabalhos Futuros

_Dono: Renato._ AWS Textract/Comprehend, OpenAI, TORGO, notificação externa. <!-- TODO -->
