# Measured limits — 2026-07-28

Everything here was measured by replaying a **real captured Claude Code request**
against each backend, not by sending toy prompts. That distinction is the whole
point of this file: a toy prompt says every backend works, and every one of the
findings below is invisible at toy size.

---

## 0. The one setting that makes any of this work

```
MAX_THINKING_TOKENS=0
```

Without it, every Claude Code turn against a LiteLLM-proxied backend fails with:

```
API Error: Content block is not a thinking block
```

Claude Code's captured request carries `"thinking": {"type": "adaptive"}`. LiteLLM
synthesises Anthropic `thinking` content blocks from whatever reasoning field the
backend emits, and Claude Code rejects the result.

Toggling only this field on one captured payload, everything else identical:

| `thinking` | reported input | output | text returned |
|---|---|---|---|
| on | **7** | 42 | `''` — empty |
| off | **18,823** | 54 | `'E-OK'` |

Note the input count. With thinking on, the prompt is not merely mis-reported, it
is barely processed. An earlier reading of this file blamed a suspiciously low
`input_tokens=126` on Cerebras silently truncating prompts. **That was wrong** —
same cause, and Cerebras was innocent (see §4).

Set in `_CC_MULTILLM_ENV` in `../multi-llm.sh`. This does not affect `/effort
xhigh` on real Anthropic models, which runs through `claude`, not `ccfree`.

**Verified end to end after the fix:** `claude -p` returned `CCOK`, and a turn
requiring real tool execution (`Read` on a file, then answer from its contents)
returned the correct value `4471`. Full agentic loop, not just text generation.

---

## 1. What a Claude Code request actually weighs

| | tools | request bytes | input tokens |
|---|---|---|---|
| plain `claude -p` | 23 | 109 KB | **28,702** |
| with tokensave MCP | 108 | 172 KB | **~43,000** |

The system prompt alone is 27.7 KB. Tool schemas are 63 KB (plain) / 127 KB (MCP).
Nothing about this is tunable — it is the floor for *every single turn*.

`--strict-mcp-config` drops MCP tools and saves ~15.6k tokens/request.

---

## 2. The real ceiling is requests per DAY, not tokens per minute

This is the finding that reshaped the config. Free tiers are capped daily, and the
caps are small enough to matter within a single working session.

| Provider | Cap | Evidence |
|---|---|---|
| **OpenRouter free** | **50 requests/day across ALL free models combined** | `X-RateLimit-Limit: 50`, `X-RateLimit-Remaining: 0`, reset `1785283200000` = midnight UTC. Error: `Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day` |
| **SambaNova free** | exhausted, not transient | Stayed `Rate limit exceeded` across three probes over two minutes |
| **Cloudflare** | 10,000 neurons/day | `you have used up your daily free allocation` |
| **Cerebras** | still serving | Throttles by latency rather than refusing (§4) |
| **ollama local** | none | The only unlimited backend |

**50 requests/day is a handful of Claude Code turns.** All three OpenRouter models
share that one bucket, so listing three of them adds redundancy, not capacity.

Two consequences, both now in the config:

- **`routing_strategy: simple-shuffle` with explicit weights**, not
  `usage-based-routing-v2`. Usage-based routing *equalises* usage across
  deployments — precisely wrong when one bucket is 50/day and another is large. It
  drains the scarce bucket first, then limps.
- **`cooldown_time: 300`**, not 60. A daily cap does not heal in a minute, so a
  short cooldown just buys a wasted round-trip every time it lapses.

Corollary: **do not enable `background_health_checks`.** Polling every deployment
on a timer spends the exact quota this setup exists to conserve — a 60s interval
would burn OpenRouter's entire daily allowance in under an hour, before any work
starts. Use `llmcheck` on demand instead.

---

## 3. Two LiteLLM behaviours that silently do the wrong thing

Both were found by observation, not documentation, and both are load-bearing.

### `"*"` in `fallbacks` is a *generic* fallback, not the `*` model group

`get_fallback_model_group()` in `router_utils/fallback_event_handlers.py` matches an
exact key first; failing that it uses the entry keyed `"*"` — the variable is
literally called `generic_fallback_idx`. So `{"*": [...]}` applies to **every model
group without its own entry**.

Symptom: the `dbg-*` aliases, whose entire purpose is to test one backend in
isolation, quietly acquired a fallback chain. `dbg-sambanova` answered *from
ollama*, and `dbg-or-ultra` answered *from Cerebras* — both reporting success.

Fix: an explicit `[]` entry per diagnostic alias (an exact match wins and breaks
before the generic check; an empty list is not `None`, so nothing is substituted).
`local` needs one too, or a failing local model falls back out to the cloud groups,
which fall back to local again.

### `weight` is all-or-nothing within a model group

`router_strategy/simple_shuffle.py` picks its metric by testing
`healthy_deployments[0]` for `"weight"`, then `"rpm"`, then `"tpm"` — then reads
that metric from every sibling **defaulting to 0**. A group where only some
deployments carry `weight` silently gives the rest weight 0 and never routes to
them again. Worse, which metric is chosen depends on which deployment happens to be
first and healthy, so the bug appears and disappears.

Leaving `weight` off entirely does work (rpm is used instead), but rpm is already
doing a different job — the pre-call rate filter — and one number cannot serve both.

---

## 4. Why most "working" free models cannot run Claude Code

| Backend | Verdict | Measured reason |
|---|---|---|
| `groq/*` (all) | **impossible** | Free TPM is per-model and tiny: `llama-3.3-70b` 12,000, `qwen3.6-27b` **8,000** (measured from the refusal itself: `Limit 8000, Used 7413, Requested 1464`), `llama-3.1-8b` 6,000. Groq bills input + `max_tokens` together, so one turn is charged at **73,889 tokens**. Claude Code's `max_tokens=32000` alone exceeds the TPM, so it fails *at any prompt size* — which is why Groq is excluded by group membership, not by `max_input_tokens`. |
| `cerebras/zai-glm-4.7` | **impossible** | Total context **8,192 tokens** (`Current length is 8948 while limit is 8192`). Fine for short `llm` prompts, which is where it lives now. |
| `cloudflare/*` | **exhausted** | Daily 10,000-neuron allocation gone. |
| `openrouter/cohere/north-mini-code:free` | broken | Returns pure reasoning, **empty content**. |

These are hard ceilings, not misconfigurations.

### Cerebras `gpt-oss-120b`: earlier suspicion retracted

It was flagged here as possibly truncating prompts silently. A needle-in-haystack
test disproves that — a codeword planted at the very start of the prompt was
recalled correctly:

| target size | reported input | needle | latency |
|---|---|---|---|
| 8,000 | 5,239 | FOUND | 0.6s |
| 20,000 | 12,979 | FOUND | 0.8s |
| 28,000 | 18,139 | FOUND | **60.4s** |

Usage reporting is accurate and nothing is dropped. The real defect is the **latency
cliff** as the TPM throttle engages: sub-second up to ~13k input, then a minute at
~18k. Hence `max_input_tokens: 32000` — plain turns (28.7k) fit, MCP-loaded turns
(43k) are filtered to a backend that will not stall.

---

## 5. What survives a real request

| Backend | Latency | tool_use | Notes |
|---|---|---|---|
| `sambanova/gpt-oss-120b` | 3.2–4.0s typical | correct | Most reliable when it has quota. Spikes to 141s. |
| `cerebras/gpt-oss-120b` | 0.6s small, 60s at 18k | correct | Fast until throttled. |
| `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | 4.9–8.8s | correct | 550B, but shares the 50/day bucket. |
| `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | 3.7s | correct | same bucket |
| `openrouter/inclusionai/ling-3.0-flash:free` | 6.7s | correct | same bucket |
| `ollama_chat/vibeforged-14b` (local) | 17–26s warm | correct | **Verified 90,048 input tokens with correct needle recall.** No quota, no network. |

An earlier note demoted SambaNova as "25s, too slow". That was a cold start at toy
size; at real size it is 3–4s.

**Graceful degradation, measured on a day when OpenRouter and SambaNova were both
spent** (replaying the captured payloads, `fb=1` meaning fallback was used):

| payload | result | time |
|---|---|---|
| 28.7k plain, sonnet tier | `E2E-OK` | 135.4s |
| 43k MCP, opus tier | `E2E-OK` | 12.6s |
| 43k, dated model ID via `*` catch-all | `E2E-OK` | 7.7s |

Nothing failed outright. That is the fallback chain terminating at `local` doing
its job — slowly, but correctly.

---

## 6. The `"E"` bug, resolved

`claude -p` returning a single character was not an SSE parsing bug. The chain was:
Groq 429s (TPM) → falls back → Cloudflare `llama-4-scout` answers → that model
genuinely emits **one token** (`output_tokens: 1`, `stop_reason: end_turn`) when
handed Claude Code's system prompt. Fixing the routing removes it.

---

## 7. Two workloads, one routing table

`llm` sends ~100-token prompts; Claude Code sends 28.7–43k. This used to require two
hand-maintained sets of aliases. It no longer does:

```yaml
router_settings:
  enable_pre_call_checks: true
```

The router counts the real prompt with tiktoken and drops every deployment whose
`model_info.max_input_tokens` is smaller, plus any already over its `rpm`, *before*
dispatching. Verified: a 146k-token prompt aimed at a 32k-declared deployment is
refused locally by `litellm._pre_call_checks: Context Window exceeded` without ever
leaving the machine.

So each group can hold fast small-context models *and* a big-context backstop:
`cat huge.log | llm "summarise"` automatically skips the small ones. Every
deployment must declare `max_input_tokens` — one that omits it is never filtered.

The `*` catch-all still matters: Claude Code sends dated model IDs
(`claude-opus-4-5-20251101`) matching no alias, so it must point at a big-context
backend. Verified working above.
