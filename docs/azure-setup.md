# Setup Azure (Speech + Language, tier F0)

Serviços usados pelo Analisador de Áudio (ESTRATEGIA.md, seções 1, 5, 12 e 15): **Speech to Text** para transcrever a Consulta e **AI Language** para Text Analytics for Health (Termos Críticos) e Sentiment.

## Estado

| Item | Valor |
| --- | --- |
| Conta | _a preencher: Azure for Students de quem? (ou Plano B)_ |
| Região (Speech e Language) | _a preencher_ — sugestão: `eastus` |
| Recurso Speech | _nome_ · F0 |
| Recurso Language | _nome_ · F0 · endpoint `https://<nome>.cognitiveservices.azure.com/` |
| Smoke test passou em | _data_ |

## Passo a passo

1. **Conta.** Criar em <https://azure.microsoft.com/free/students> com o e-mail acadêmico (FIAP). Não pede cartão; vem com crédito, mas tudo aqui é F0, então não consome crédito.
2. **Grupo de recursos.** Portal → *Resource groups* → criar `rg-fase4` na região escolhida.
3. **Speech.** Portal → *Create a resource* → *Speech* → mesmo grupo e região → *Pricing tier* **Free F0**. Em *Keys and Endpoint*, copiar `KEY 1` e a *Location/Region*.
4. **Language.** Portal → *Create a resource* → *Language service* → manter as features padrão (Text Analytics for Health e Sentiment fazem parte do serviço base) → mesmo grupo e região → **Free F0** → aceitar o aviso de IA responsável. Em *Keys and Endpoint*, copiar `KEY 1` e o *Endpoint*.
5. **`.env`.** `cp .env.example .env` na raiz e preencher:

   ```text
   AZURE_SPEECH_KEY=<KEY 1 do Speech>
   AZURE_SPEECH_REGION=<região, ex.: eastus>
   AZURE_LANGUAGE_KEY=<KEY 1 do Language>
   AZURE_LANGUAGE_ENDPOINT=https://<nome>.cognitiveservices.azure.com/
   ```

   O `.env` está no `.gitignore`. **Nunca commitar chaves.** Compartilhar com o time só por canal privado (DM, gerenciador de senhas), nunca em issue/PR/commit.
6. **Smoke test.**

   ```bash
   uv run python -m scripts.azure_smoke_test
   ```

   Sintetiza uma frase em inglês com o próprio Speech, transcreve de volta (STT `en-US`) e chama TA for Health e Sentiment. Sai com código 0 e três `[OK]` quando está tudo certo; código 2 se faltar variável no `.env`; código 1 se algum serviço falhar (a mensagem mostra qual). Para testar com um áudio real: `--audio caminho.wav` (WAV 16 kHz mono).

Se a região escolhida não oferecer Text Analytics for Health, o smoke test falha no `[FALHA] TA for Health`: recriar o recurso Language em outra região (`eastus`, `westeurope`) e atualizar o `.env`.

## Cotas F0 (orçamento de cota)

Conferir os números no portal (*Pricing tier* de cada recurso) no momento da criação; a Microsoft muda limites sem aviso.

| Recurso | Cota F0 mensal | Observação |
| --- | --- | --- |
| Speech to Text | 5 h de áudio | Compartilhada pelo time inteiro (uma chave só). Uma Consulta do PriMock57 tem ~9 min → ~15–20 Consultas em batch ≈ 2,5–3 h; o resto fica para as demos de upload e a gravação do vídeo. |
| Speech — TTS neural | 0,5 M caracteres | Usado só pelo smoke test (uma frase). |
| Language (TA for Health + Sentiment) | 5.000 registros de texto | 1 registro = até 1.000 caracteres de um documento. Transcrições longas consomem vários registros: dividir por fala e cachear a resposta em JSON. |

Regras para não estourar (ESTRATEGIA.md, seção 15):

- **Cachear toda resposta do Azure em JSON** e nunca chamar de novo o que já está em cache.
- Rodar o batch do PriMock57 **uma vez**, num subset de 15–20 Consultas.
- Reservar cota para o dia da gravação; o smoke test custa ~10 s de STT e 2 registros de Language.
- F0 é limitado a **um recurso por tipo por assinatura**: se alguém já tiver um Speech F0 na mesma assinatura, reutilizar.

## Plano B

Se o Azure for Students não for liberado (ou exigir cartão), em paralelo e na primeira semana:

1. Uma segunda pessoa tenta o Azure for Students com o próprio e-mail acadêmico.
2. Se nenhuma conta for liberada: **conta Azure free com cartão** de um integrante. Os recursos continuam F0 (não cobram); registrar aqui de quem é a conta.
3. Em qualquer caso, só uma conta abastece o time: as chaves vão para o `.env` de cada um.
