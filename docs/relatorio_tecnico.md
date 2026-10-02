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

_Dono: Antonio._ <!-- TODO -->

### 3.4 Prescrições — regras sobre MIMIC-IV Demo

_Dono: V. Blasque._ <!-- TODO -->

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
