# Teste do upload de áudio no Laboratório

Roteiro para conferir, em cerca de 5 minutos, o caminho completo do upload de áudio com o Azure real: transcrição, Termos Críticos e Alteração Vocal.

**Estado:** testado com o Azure real por Renato em novembro de 2026, com sucesso.

## Antes de começar

- Dependências instaladas com `uv sync` e `ffmpeg` disponível (o devcontainer já traz o `ffmpeg`).
- `.env` na raiz com as 4 variáveis `AZURE_*`. Veja o passo 5 de [azure-setup.md](azure-setup.md). As chaves são passadas por canal privado, nunca por issue, PR ou commit.

## Passo a passo

1. Atualize a branch com `git pull`.
2. Rode `uv run streamlit run app.py`.
3. No menu, abra **Laboratório de Análise**.
4. Em **Áudio da Consulta**, envie [`data/audio/teste_upload/teste_falta_de_ar.wav`](../data/audio/teste_upload/teste_falta_de_ar.wav). O seletor abre no seu computador: se você usa devcontainer e não encontrar o arquivo, baixe-o pelo painel do VS Code (botão direito → *Download*).
5. Clique em **Analisar** e aguarde.

## Resultado esperado

O clipe tem 60 s, em inglês, e é a fala de um paciente relatando falta de ar.

| Onde olhar | O que deve aparecer |
| --- | --- |
| Transcrição | Texto em inglês, com "short of breath" e "wheezing" (ou "wheezing a bit") |
| Alertas | Um Alerta de cada um desses dois Termos Críticos |
| Probabilidade de Alteração Vocal | Um percentual; acima do limiar, gera também um Alerta de Alteração Vocal |

Ter Alerta de Alteração Vocal não indica sucesso nem falha do teste: o classificador acusa alteração em quase todas as Consultas do PriMock57 (17 de 18), e o resultado neste clipe curto pode ser diferente.

O botão **Gravar Alertas no Banco** é opcional e só altera o banco local.

## Teste opcional: sintoma negado

Grave no celular, em inglês, a frase "I have no fever and no chest pain" e envie o arquivo (M4A e MP3 também são aceitos).

Não deve surgir Alerta de Termo Crítico, porque sintoma negado não conta. Se surgir, anote a frase transcrita: o filtro depende da marcação do Azure e já deixou passar alguns casos no lote.

## Consumo de cota

As chaves e a cota F0 são as mesmas para o time inteiro.

- A primeira análise deste clipe em cada máquina gasta cerca de 1 minuto de Speech e poucos registros de Language.
- Repetir o mesmo arquivo na mesma máquina usa o cache local em `data/audio/cache_laboratorio/` e não gasta cota. Por isso, repetir também não testa o Azure de novo.
- Evite enviar Consultas inteiras: cada uma gasta cerca de 9 minutos de Speech.

## Se algo der errado

| Sintoma | O que verificar |
| --- | --- |
| Mensagem de configuração incompleta | Falta variável no `.env`, ou ele não está na raiz |
| "Falha ao analisar o áudio" | Leia o erro. Rode `uv run python -m scripts.azure_smoke_test` para ver qual serviço falha (gasta cerca de 10 s de Speech e 2 registros de Language) |
| "Nenhuma fala reconhecida" | Confira se o arquivo tem fala em inglês; o reconhecimento usa `en-US` |
| Sem o percentual de Alteração Vocal | Falta `data/audio/modelo_acustico.joblib`; atualize a branch |

## Origem do clipe

Trecho de 35 s a 95 s da fala do paciente em `day3_consultation04`, do PriMock57. As condições de uso estão em [`data/audio/SOURCES.md`](../data/audio/SOURCES.md).
