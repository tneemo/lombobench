# LomboBench

Observabilidade **caixa-preta** dos *controles externos de inferência* do Ollama —
Fase 1 do LomboAI.

Para cada prompt enviado ao modelo, o LomboBench registra: quais parâmetros foram
**solicitados**, quais valores efetivos foram **inferidos** (e de onde vieram:
request, Modelfile ou default do Ollama), o que o servidor efetivamente
**confirmou**, o diff em relação ao baseline do modelo, o comportamento observável
(texto, thinking separado, métricas, time-to-first-token) e avaliações
comportamentais. Esses dados alimentam a Fase 2 (regulador autônomo dos controles).

> **Enquadramento honesto:** este sistema não acessa pesos, ativações nem estados
> internos do modelo — os pesos são imutáveis durante a inferência. Ele observa os
> controles externos (amostragem, template, system, contexto) e o comportamento.
> Não substitui ferramentas de interpretabilidade mecanística (NNsight,
> TransformerLens); a Fase 3 fará essa ponte reexecutando casos registrados.

## Arquitetura

```
Qualquer cliente (curl, LangChain, Open WebUI…)
        │  http://127.0.0.1:11435  (drop-in: mesma API do Ollama)
        ▼
   LomboProxy (FastAPI + httpx)
        │── repassa tudo ao Ollama :11434 (streaming transparente)
        │── captura /api/generate, /api/chat, /v1/chat/completions
        │── Lab + análise em /_lombo/
        ▼
   lombo.db (SQLite, escritor único em fila)
```

## Uso

```powershell
.venv\Scripts\python.exe -m lombo            # proxy em :11435, Lab em /_lombo/
# flags: --port --host --ollama-url --db --no-capture
```

Aponte qualquer cliente para `http://127.0.0.1:11435` no lugar do Ollama.
Metadados experimentais via headers (removidos antes do repasse):
`X-Lombo-Experiment`, `X-Lombo-Condition`, `X-Lombo-Pair`, `X-Lombo-Replicate`,
`X-Lombo-No-Log` (não grava a requisição).

### Engines de inferência

O contrato em `lombo.engines.Engine` separa o benchmark do runtime. Ele expõe
`generate`, `tokenize`, `get_model_info` e o gancho `supports_trace`; os
resultados usam os mesmos campos normalizados da captura atual. O proxy Ollama
existente continua sendo o caminho padrão e não foi alterado.

Para inferência local com pesos Hugging Face/PyTorch (não GGUF):

```powershell
pip install -e ".[transformers]"
```

```python
from lombo.engines import TransformersEngine

engine = TransformersEngine("Qwen/Qwen2.5-0.5B-Instruct")
result = engine.generate(
    [{"role": "user", "content": "Explique entropia em uma frase."}],
    seed=42, temperature=0.7, top_p=0.9, top_k=40, max_new_tokens=80,
)
print(result["response_text"])
```

O engine seleciona dispositivos com `device_map="auto"`, usa BF16 quando a
GPU oferece suporte (FP16 em outras GPUs e FP32 em CPU) e carrega as
dependências pesadas somente ao ser instanciado.

O primeiro suporte opt-in ao NNsight é instalado separadamente:

```powershell
pip install -e ".[transformers,nnsight]"
```

```python
# trace de camada única (opt-in NNsight):
if engine.supports_trace():
    trace = engine.trace("Explique entropia.", layer_index=0)

# diagnóstico correlacional multi-camada (logit lens, sem NNsight):
if engine.supports_diagnostics():
    diag = engine.trace_all_layers("Explique entropia.", target_token="ordem")
```

`trace` executa um único forward pass e captura somente a saída da camada
indicada e os logits da última posição. O retorno serializável contém o índice
da camada, shape e dtype da ativação, IDs dos tokens, logits do último token e
metadados do modelo.

`trace_all_layers` (diagnóstico de camadas, exposto no Lab pela aba
**Diagnóstico**) faz o **logit lens** numa única passada **read-only** — um
forward hook por bloco captura o resíduo, projetado pela norma final + `lm_head`
(não usa NNsight; a leitura é passiva). Reporta, por camada, os top-k tokens, as
normas do hidden state (‖h‖ e Δ residual) e — se um token-alvo for informado —
seu rank e probabilidade. Projetar o resíduo da última camada reproduz os logits
reais do modelo, ancorando o lens à predição verdadeira. Isso mostra **em qual
camada uma predição fica *legível*** (leitura representacional), o que **não** é
o mesmo que provar que a camada *computa* ou *usa* a feature. A resposta causal
("utilizada") exige intervenção — activation patching, onde o NNsight de fato
brilha — prevista para a próxima fase. Attention, MLP, cache, ablation, patching
e SAEs permanecem fora do escopo desta fase.

### Lombo Lab (`http://127.0.0.1:11435/_lombo/`)

Painéis de chat lado a lado com **uma variável manipulada por painel**
(temperature, seed, modelo, system…). Dois modos por sessão:

- **Experimento controlado (padrão)** — cada rodada é independente: todos os
  painéis recebem o mesmo histórico-base e só a variável difere. Comparável;
  é o modo do paper.
- **Conversa ramificada** — cada painel segue seu próprio histórico
  (`branch_id`/`parent_request_id` registram a árvore).

Cada envio gera um `pair_id`; reenvios incrementam `replicate`. Sem seed
definida, o Lab injeta uma seed explícita por rodada (igual em todos os
painéis) — todo registro é **replayável** (gancho do "Inspecionar execução"
da Fase 3). Sob cada resposta: diff palavra a palavra, métricas e rótulos
epistêmicos humanos (*correto, alucinou, reconheceu entidade inexistente,
corrigiu premissa falsa, pediu esclarecimento, demonstrou incerteza, resposta
inconclusiva*), gravados em `evaluations` com `source='human'`.

### Análise e export

Aba **Análise** (ou `/_lombo/api/*`): timeline, detalhe com as três camadas de
parâmetros (nunca fundidas — `/api/ps` fica à parte em `observed_runtime_json`),
`GET /summary`, `GET /compare?pair_id=`, `GET /params/impact?param=temperature`,
`GET /export?format=csv|json`.

## Overhead do proxy

Medido, não alegado: `python scripts/overhead_bench.py`. Na máquina de
desenvolvimento (Ollama 0.31.2, llama3.2:3b, n=8), a diferença de mediana ficou
dentro do ruído de medição (−10,8 ms total; −4,6 ms TTFT).

## Privacidade

O proxy grava prompts e respostas localmente. Por padrão escuta **apenas
127.0.0.1**; `X-Lombo-No-Log` exclui uma requisição; `--no-capture` desliga a
captura; o Lab exibe aviso permanente.

## Desenvolvimento

```powershell
.venv\Scripts\python.exe -m pytest    # camadas de params, parsing de stream, heurísticas
```

Fora do escopo desta fase (preparado no schema): regulador autônomo (Fase 2),
logprobs por token, avaliações por modelo juiz. A interpretabilidade via NNsight
(Fase 3) já entrega o diagnóstico **correlacional** de camadas (logit lens
multi-camada, aba Diagnóstico); o diagnóstico **causal** por activation patching
é o próximo passo (`requests.request_body_raw` + seed + digest garantem o replay).
