# TUDaGPT (HAWKI) API

LLM backend for the classifier. Runs on TU Darmstadt infrastructure.

## Base URL

```
https://tudagpt.hrz.tu-darmstadt.de
```

Endpoints:
- `POST /api/ai-req` — model completion
- `GET /api/user` — verify token

## Auth

Bearer token (Laravel Sanctum). Create it in the TUDaGPT web profile under "API Tokens" (shown once). Store in `.env`, never commit.

```
Authorization: Bearer <TOKEN>
```

## Models

| Tier | Slug |
|------|------|
| Strong | `mistral-large-3-675b-instruct-2512` |
| Medium | `qwen-3-5-122b-a10b` *(confirm slug from dev tools)* |
| Fast | `meta-llama-3-3-70b-instruct` *(confirm slug from dev tools)* |

Get exact slugs from the web chat: F12 → Network → send a message → open the `streamAI` request → `model` field in the payload.

## Request format

Note the non-standard `payload` wrapper and nested `content.text`:

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

`temperature` is supported (confirmed: produces run-to-run variation).

## Response format

```json
{ "success": true, "content": { "text": "<output>" } }
```

Read the output from `response["content"]["text"]`.

## Sample call (curl)

```bash
curl -X POST https://tudagpt.hrz.tu-darmstadt.de/api/ai-req \
  -H "Authorization: Bearer $HAWKI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"payload":{"model":"mistral-large-3-675b-instruct-2512","temperature":1.0,"messages":[{"role":"user","content":{"text":"Give me a random word."}}]}}'
```

## Errors

| Code | Meaning |
|------|---------|
| 401 | bad/missing token |
| 403 | external API disabled or no permission |
| 422 | validation error (returns field detail) |
| 500 | server error (retry with backoff) |

## Notes

- All requests are usage-tracked and may be rate-limited.
- Self-consistency multiplies calls per input — factor into quota.
- HTTPS only.