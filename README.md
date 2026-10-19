# Zen LLM Engine

<p align="center">
  <img src="web/zen-llm-engine-logo.png" alt="Zen LLM Engine Logo" width="250">
</p>

**Zen LLM Engine — A hassle-free Web LLM engine**
> Disclaimer: this software has been vibe-coded.

>  [!NOTE]
> This project was previously named **selenium-llm-engine**. It has been renamed to **Zen LLM Engine** to better reflect its zendriver-based architecture. For backward compatibility, the Docker image is also published under the old name `xargonwan/selenium-llm-engine:latest` (which will be deprecated in a future release). The legacy `SELENIUM_DEBUG` environment variable still works as an alias for `ZEN_DEBUG`.

![Docker Pulls](https://img.shields.io/docker/pulls/xargonwan/zen-llm-engine)
| Branch    | Build Status                                                                                                                                         | Docs Status                                                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `main`    | [![CI Status](https://img.shields.io/github/actions/workflow/status/XargonWan/zen-llm-engine/build-and-publish.yml)](https://github.com/xargonwan/zen-llm-engine/actions)

This repository provides a standalone, Docker-friendly LLM engine proxy. It automates browser access to Web LLMs via [zendriver](https://github.com/stephanlensky/zendriver) (a Chrome DevTools Protocol driver — no Selenium/chromedriver in the loop) to expose a simple OpenAI-compatible API and web admin UI.

Feel free to submit pull requests with improvements, new engines, or engine definitions (`engines/*.json` or `engines/*.py`).

</br>
> [!NOTE]
> The API (including the web admin UI at `/ui`) listens on port `8000` by
> default over plain **HTTP**. Docker Compose maps this to `14848` in this
> repository, but you can change this mapping freely in `docker-compose.yml`.
> The webtop remote desktop is a separate service served by Selkies on port
> `3001` over **HTTPS** — it does not expose the API routes (there is no
> `/ui` on `3001`). Access the admin UI at `http://<host>:14848/ui`, not on
> `3001`.


## Features

- zendriver-based (Chrome DevTools Protocol) support for web-based LLMs.
- Unified REST API endpoints:
  - `/api/ping` (liveness only: proves the HTTP server is up)
  - `/api/status` and `/api/health` (real health: per-engine breaker/queue/login state, RSS, zombies; never blocks on the browser)
  - `/api/engines`
  - `/api/engines/reload`
  - `/api/engines/{engine}/recover` (close an engine's circuit breaker)
  - `/models` and `/models/{engine}`
  - `/login/{engine}` and `/login/{engine}/state`
  - `/engine/{engine}/prompt`
  - `/v1/models`
  - `/v1/models/{model_id:path}`
  - `/v1/chat/completions` (OpenAI-like compatibility)
  - `/chat/completions` (OpenAI-like legacy compatibility)
  - `/stats` (aggregated counters + response time averages)
  - `/api/logs/app` (incremental app log polling)
  - `/api/debug` (get/set debug mode) and `/api/debug/log` (debug trace polling)
  - `/api/engines/selector-hints` (runtime selector hints)
  - `/reset` and `/api/reset` (clears engine state and stats counters)
  - `/api/session/kill` (force-kills frozen browser session processes)
  - `/logs` and `/api/history`
  - `/ui`
- SQLite storage for prompt logs and counters
- Web UI for simple login, prompt sending and metrics
- Docker + docker-compose support

## Quickstart

The easiest way to run Zen LLM Engine is to pull the Docker image from Docker Hub.

**New name (recommended):**
```bash
docker pull xargonwan/zen-llm-engine:latest

docker run -d --name zen-llm-engine \
  -p 14848:8000 \
  -p 3001:3001 \
  -v data:/app/data \
  -v config:/config \
  xargonwan/zen-llm-engine:latest
```

**Legacy name (for backward compatibility):**
```bash
docker pull xargonwan/selenium-llm-engine:latest

docker run -d --name selenium-llm-engine \
  -p 14848:8000 \
  -p 3001:3001 \
  -v data:/app/data \
  -v config:/config \
  xargonwan/selenium-llm-engine:latest
```

> [!NOTE]
> Both image names point to the same image. The legacy `selenium-llm-engine` name is published for backward compatibility and will be deprecated in a future release.

3. OpenAI-compatible endpoints:

- `http://localhost:14848/chatgpt/prompt`
- `http://localhost:14848/gemini/prompt`
- `http://localhost:14848/stepfun/prompt`
- `http://localhost:14848/claude/prompt`
- `http://localhost:14848/v1/chat/completions`
- `http://localhost:14848/ui`

4. OpenAPI / OpenAI client

This application exposes an OpenAI-compatible endpoint for clients and SDKs:

- Endpoint: `POST http://localhost:14848/v1/chat/completions`
- `model`: `chatgpt` or `gemini`
- `messages`: standard OpenAI array

Example using `openai` (Python):

```python
from openai import OpenAI

client = OpenAI(api_key="dummy")  # the proxy does not require a real key, values can be dummy
client.api_base = "http://localhost:14848"
client.api_type = "openai"
client.api_version = ""

resp = client.chat.completions.create(
    model="chatgpt",
    messages=[{"role": "user", "content": "Write a short poem in English about a robot learning to sing."}]
)
print(resp)
```

If the client does not directly support base URL configuration, use a manual request with `requests`:

```python
import requests

url = "http://localhost:14848/v1/chat/completions"

payload = {
    "model": "chatgpt",
    "messages": [{"role": "user", "content": "Scrivi una breve poesia in italiano."}]
}

r = requests.post(url, json=payload)
print(r.json())
```

## Health, circuit breaker and manual sign-in

A browser-driven engine that is signed out (or whose page changed) cannot recover
by retrying, and every retry resets the shared Chromium. That is destructive: a
sign-in you complete by hand in that browser (the web UI on port `3001`) is lost
if Chromium is restarted before it flushes its cookies to disk. So each engine has
a **circuit breaker**:

- It opens after `ZEN_BREAKER_THRESHOLD` (default `3`) consecutive failed
  requests, or immediately when a failed attempt coincides with a visible
  "signed out" marker (`login_detection.login_button_css`).
- While open, requests are rejected at once with **HTTP 503**, a `Retry-After`
  header and `{"error": {"type": "engine_unavailable", "reason": "login" | "failures"}}`.
  Queued requests are failed immediately and **the browser is left alone**.
- `reason: "failures"` allows one trial request after `ZEN_BREAKER_COOLDOWN`
  seconds (default `300`, doubling on each failed trial up to
  `ZEN_BREAKER_MAX_COOLDOWN`, default `3600`).
- **To sign in by hand**, first `POST /api/engines/<engine>/pause` (or
  `POST /login/<engine>`, which pauses too) so requests stop navigating the tab
  while you type; resume with the watcher or `.../recover`.
- `reason: "login"` never retries on its own. Sign in through the web UI on
  port `3001`; a read-only watcher (`ZEN_LOGIN_WATCH_INTERVAL`, default `15` s)
  closes the breaker once the page looks signed in, or call
  `POST /api/engines/{engine}/recover`.

`GET /api/status` reports it all (it never takes the browser lock, never touches
the browser or the database, so it answers even when a request is wedged):

```json
{"status": "ok|degraded|down",
 "process": {"rss_mb": 812.4, "rss_warning": false, "zombie_processes": 0},
 "browser": {"launches": 3, "last_launch_at": 1760000000.0, "pid": 1234},
 "engines": [{"name": "gemini", "state": "down", "queue_depth": 0,
              "last_success_at": 1760000000.0, "login_state": "unlogged",
              "breaker": {"open": true, "reason": "login", "retry_in_seconds": null}}]}
```

`/api/health` returns the same body with HTTP 503 when any engine is `down`.
`browser.launches` is a smoke detector: it should grow by a handful per day, not
by hundreds. `ZEN_RSS_WARN_MB` (default `2048`) sets the `rss_warning` threshold.

## Debug mode

Debug mode traces everything that flows through the engine and is **disabled by
default**. When enabled it records, into an in-memory ring buffer, the full
prompt lifecycle:

- the input prompt,
- the chunks generated when an oversized prompt is split,
- the output reply,
- the target engine,
- the total time from input to output (`elapsed_ms`).

It can be enabled in two ways:

- **Environment variable** at startup — set `ZEN_DEBUG` (or legacy `SELENIUM_DEBUG`) to one of
  `1`/`true`/`yes`/`on` (case-insensitive):

  ```bash
  docker run -d --name zen-llm-engine \
    -e ZEN_DEBUG=true \
    -p 14848:8000 -p 3001:3001 \
    -v data:/app/data -v config:/config \
    xargonwan/zen-llm-engine:latest
  ```

- **Web UI / API** at runtime — toggle it from the *Debug Mode* card in `/ui`,
  or via the API:

  ```bash
  # read state
  curl http://localhost:14848/api/debug
  # enable
  curl -X POST http://localhost:14848/api/debug -H 'Content-Type: application/json' -d '{"enabled": true}'
  # read trace events (incremental)
  curl 'http://localhost:14848/api/debug/log?since=0'
  ```

A runtime toggle overrides the environment default until the process restarts.

## Media upload support

`/v1/chat/completions` (and legacy per-engine `/chatgpt/prompt`, `/gemini/prompt`, etc.) accepts multimodal `messages` payloads where a message `content` can be an array of parts:

- `type: "text"` for text content
- `type: "image_url"` for an inline image using a data URI or raw base64 payload
- `type: "input_audio"` for audio upload in supported engines

Example `curl` request with an inline image:

```bash
curl -X POST "http://localhost:14848/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "content": "Here is an image:"},
          {
            "type": "image_url",
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=",
            "filename": "example.png"
          }
        ]
      }
    ]
  }'
```

If the chosen engine supports audio uploads, use `type: "input_audio"` with a base64 audio payload and optional `mime_type`:

```json
{
  "type": "input_audio",
  "data": "<base64-audio-data>",
  "mime_type": "audio/mpeg",
  "filename": "example.mp3"
}
```

## Agentic (tool-calling) mode

`/v1/chat/completions` automatically switches to an **agentic lane** when the
request carries any OpenAI agentic field:

- `tools` (non-empty list of function definitions)
- `tool_choice`
- `response_format` (e.g. `{"type": "json_object"}`)

You can also force it explicitly with `"mode": "agent"`.

In this mode the server injects a JSON-only harness into the prompt so the
browser-driven model answers with a single structured JSON object, then parses
that reply back into standard OpenAI `tool_calls`.

Example request with tools:

```bash
curl -X POST "http://localhost:14848/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt",
    "messages": [{"role": "user", "content": "What is the weather in Rome?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

When the model decides to call a tool the response uses the standard OpenAI
shape (`content` is `null`, `finish_reason` is `tool_calls`):

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

When the model gives a final answer instead, `finish_reason` is `stop` and the
text is returned in `message.content` as usual.

**Notes and limits:**

- The agentic lane is **stateless**: the caller must re-send the full
  conversation (including previous `tool_calls` and `tool` results) on every
  request, exactly as with the OpenAI API.
- Output is **best-effort**. If the model's first reply cannot be parsed, the
  server issues a bounded number of *reformulation* retries nudging it back to
  valid JSON. If it still fails, the raw text is returned as `content`.
- Real token streaming is not supported for tool calls; the response is
  delivered once parsing completes.
- The feature is fully **engine-agnostic** — it works with any configured
  engine and requires no engine-specific configuration.

3. Prompt example (API call)

Before sending a prompt, make sure you have logged in using `/login/chatgpt` or `/login/gemini`.

```bash
curl -X POST "http://localhost:14848/chatgpt/prompt" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short poem in English about a robot learning to sing."}'
```

Example response:

```json
{
  "id": "llm_1680390000",
  "object": "chat.completion",
  "created": 1680390000,
  "model": "chatgpt",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "engine": "chatgpt",
  "prompt": "Write a short poem...",
  "elapsed_ms": 850
}
```

4. Login flow (before `/prompt`):

- `POST http://localhost:14848/login/chatgpt`
- `POST http://localhost:14848/login/gemini`

## Local build (from source)

If you want to build locally from this repository, use Docker Compose:

```bash
git clone https://github.com/XargonWan/zen-llm-engine.git
cd zen-llm-engine
docker compose up --build
```

This runs the same service under `http://localhost:14848` (and `https://localhost:3001` for the webtop remote desktop).

## Directory structure

- `app.py` - FastAPI entrypoint
- `core/` - zendriver engine wrappers and manager
- `db/` - SQLite persistence helpers
- `web/` - minimal static UI
- `tests/` - API tests and `stress_test.py` (manual, excluded from CI)
- `Dockerfile`, `docker-compose.yml` - container setup

## Stress Test Results

A ramp-up stress test was run against the live system with the following setup:
- 10 long prompts (~3,000–8,000 chars each) sent to Gemini with delays ramping from 30s down to 0s
- 10 long prompts sent to ChatGPT (same ramp-up pattern)
- 5 prompts each sent in burst to: claude, copilot, perplexity, stepfun, grok

### Average Response Times

| Engine | Avg Response | Notes |
|--------|-------------|-------|
| copilot | 16.9s | Fastest, but unstable under load |
| perplexity | 35.2s | Moderately stable |
| chatgpt (unlogged) | 71.1s | Slow, high variance (42–426s) |
| claude (unlogged) | 79.0s | Error-prone in unlogged mode |
| gemini | 94.8s | Slowest but most reliable |

### System Soliditude

- **FIFO queue**: No race conditions detected; requests to each engine are processed sequentially and correctly.
- **Per-engine parallelism**: Each engine processes independently; no cross-engine interference.
- **Failure modes**: The dominant failure cause is `fill_input` verification rejecting long prompts that the target site reformats (e.g., ChatGPT truncating or reformatting prompt text before sending). A secondary issue is a `Connection pool is full` warning indicating a single-connection pool to the Chrome driver, which creates a bottleneck during burst traffic.
- **Engines requiring login**: stepfun, grok, and (partially) claude are unreliable or non-functional without authenticated sessions.

### Overall Assessment

The system handles concurrent load across multiple engines without deadlocks or queue corruption. The FIFO queue is the strongest component. Browser stability under sustained burst traffic is the main fragility — driven by input verification strictness and connection pool limitations rather than the queue itself.

## Legal / Terms of Service (ToS) Notice

- This software is intended for research and testing purposes only.
- Before using this tool with any online service, verify and comply with that service's Terms of Service (ToS) and usage policies.
- If the target service does not allow automated access or use via browser automation, do not use this software against it.
- Always respect copyright and service contract requirements.
