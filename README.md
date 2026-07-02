# Hate Speech Classifier — Prototype

Adaptable multi-label hate speech classifier using an LLM + RAG (Master's thesis, TU Darmstadt / PEASEC).

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`.env` file (not committed):
```
TUDAGPT_TOKEN=<your token>
TUDAGPT_URL=https://tudagpt.hrz.tu-darmstadt.de/api/ai-req
```

## LLM backend: TUDaGPT (HAWKI)

### Endpoints
- `POST /api/ai-req`
- `GET /api/user`

### Auth
Bearer token, from `.env`.

### Models

| Tier | Slug |
|------|------|
| Strong | `mistral-large-3-675b-instruct-2512` |
| Medium | `qwen-3-5-122b-a10b` *(confirm slug from dev tools)* |
| Fast | `meta-llama-3-3-70b-instruct` *(confirm slug from dev tools)* |

Get exact slugs from the web chat: F12 → Network → send a message → open the `streamAI` request → `model` field in the payload.

### Request format

Non-standard `payload` wrapper, nested `content.text`. Keep the payload minimal — only `model`, `temperature`, `messages`. Extra fields (`stream`, `tools`) copied from the web app have caused server-side errors.

```json
{
  "payload": {
    "model": "mistral-large-3-675b-instruct-2512",
    "temperature": 1.0,
    "messages": [
      { "role": "system", "content": { "text": "<system prompt>" } },
      { "role": "user", "content": { "text": "<input>" } }
    ]
  }
}
```

### Response format

```json
{ "success": true, "content": { "text": "<output>" } }
```

Read output from `response["content"]["text"]`.

### Errors

| Code | Meaning |
|------|---------|
| 401 | bad/missing token |
| 403 | external API disabled or no permission |
| 422 | validation error |
| 500 | server error — retry with backoff |

