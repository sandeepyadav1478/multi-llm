# Multi-LLM controls for Claude Code.  Sourced from ~/.zshrc.
#
# Three ways to get work done, in descending order of quality:
#
#   claude / cc     Anthropic, your Pro subscription.  Opus + xhigh effort.
#   ccacct use X    switch to the other Pro subscription when the first is spent.
#   ccfree          Claude Code driven by free cloud models via LiteLLM.
#   cclocal         Claude Code driven by the local ollama model.  No quota, offline.
#   llm "..."       one-shot prompt to a free/local model, for chores.
#
# Services are on-demand:  olresume / llmresume  (see aliases further down in .zshrc)

# --- shared endpoint -------------------------------------------------------
export LITELLM_BASE="http://localhost:4000"
export LITELLM_KEY="sk-litellm-local"

# CLAUDE_CODE_MAX_OUTPUT_TOKENS is load-bearing, not a tuning knob.
# Claude Code otherwise asks for far more than free backends allow, and LiteLLM
# will not clamp an oversized client value -- verified, it forwards it and the
# request fails. 32000 is accepted by every backend in the Claude Code chain and
# is far more output than a turn ever needs.
#
# Providers also bill input + max_tokens together against their rate limits, so
# this number is not free: it is why Groq (free TPM 6k-12k per model) is out of
# the Claude Code chain entirely -- one turn bills at ~73,889 tokens there.
# Details: ~/ai-configs/multi-llm/docs/measurements.md
#
# MAX_THINKING_TOKENS=0 is REQUIRED, not a preference. With extended thinking on,
# every turn dies with:
#     API Error: Content block is not a thinking block
# LiteLLM synthesises Anthropic `thinking` blocks out of whatever reasoning field
# the backend emits, and Claude Code rejects the result. Setting the budget to 0
# stops it asking for thinking at all, and the same prompt then returns cleanly.
# This is the single switch that makes Claude Code work against these backends;
# without it nothing else in this file matters.
#
# Note this is thinking, not effort: /effort xhigh on real Anthropic models is
# unaffected, because that runs through `claude`, not `ccfree`.
_CC_MULTILLM_ENV=(
  "ANTHROPIC_BASE_URL=$LITELLM_BASE"
  "ANTHROPIC_AUTH_TOKEN=$LITELLM_KEY"
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000"
  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
  "MAX_THINKING_TOKENS=0"
)

# --- Claude Code on free cloud models --------------------------------------
# Tiers map onto the aliases in ~/ai-configs/litellm/config.yaml. Each one has a
# fallback chain ending at the local model, so a spent provider degrades instead
# of failing.
ccfree() {
  env "${_CC_MULTILLM_ENV[@]}" \
      ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-8 \
      ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-5-0 \
      ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5 \
      ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5 \
      claude "$@"
}

# --- Claude Code fully local ------------------------------------------------
# Everything, including the background/summarising model, on vibeforged-14b.
# Needs ollama up: olresume
cclocal() {
  env "${_CC_MULTILLM_ENV[@]}" \
      ANTHROPIC_DEFAULT_OPUS_MODEL=local \
      ANTHROPIC_DEFAULT_SONNET_MODEL=local \
      ANTHROPIC_DEFAULT_HAIKU_MODEL=local \
      ANTHROPIC_SMALL_FAST_MODEL=local \
      claude --model local "$@"
}

# --- chore session ----------------------------------------------------------
# A NORMAL Anthropic session -- same model, same /effort -- that treats offloading
# as the default rather than something to remember. For sessions that are mostly
# log reading, summarising and drafting, which is exactly the work the free models
# measure at 95-98% on (docs/model-quality.md).
#
# The parent stays Anthropic on purpose. The free models never run unsupervised;
# they get handed a chore and their answer is read before anything is done with it.
#
# This is the opt-in switch for the `offload` skill. Plain `claude` still has the
# skill available and will use it when it fits -- this just stops it being a
# judgement call every time.
ccchore() {
  claude --append-system-prompt "$(cat <<'EOF'
Offload mode is ON for this session.

Before reading any file, log, or command output larger than roughly 2000 tokens,
pipe it to `llm` and read only the summary -- do not read the raw text into
context. `cat big.log | llm "summarise the errors"` costs ~300 tokens; reading
big.log costs 50k, and keeps costing it on every later turn.

Follow the `offload` skill for tiers and, more importantly, for what must never
be offloaded (anything numeric, anything a decision depends on, anything needing
this conversation). You remain responsible for every result: read each summary
and sanity-check it before acting on it.
EOF
)" "$@"
}

# --- health check -----------------------------------------------------------
# Answers "what can I actually use right now?" without guessing.
#
# Probes the dbg-* aliases, NOT smart/fast/default. Those are model groups with
# fallback chains, so they answer happily while the backend you care about is
# dead -- an earlier version of this function checked them and reported four
# healthy tiers on a day when every OpenRouter model and SambaNova were spent.
# The dbg-* aliases have fallbacks explicitly disabled, so a dead route says so.
llmcheck() {
  print -r -- "litellm : $(systemctl --user is-active litellm-proxy 2>/dev/null)"
  print -r -- "ollama  : $(systemctl is-active ollama 2>/dev/null)"
  print -r -- "account : $(ccacct which 2>/dev/null)"
  print -r -- "backends (isolated, no fallback):"
  # $EPOCHREALTIME is empty without this module, which silently makes every
  # timing print 0.0s.
  zmodload -i zsh/datetime
  # Declared once, outside the loop: a bare `local got` on an already-set variable
  # makes zsh *display* it (typeset semantics), leaking a "got=ok" line per probe.
  # Asked of the running proxy, never hardcoded. This list used to be eight
  # names typed out by hand while the config held ten, and the two missing
  # groq variants stayed invisible for as long as nobody compared the two --
  # long enough to get "8 backends" written into a report about a 10-backend
  # sweep. A second copy of a list is a copy that drifts.
  local m t0 got
  local -a probes
  probes=(${(f)"$(curl -s --max-time 10 "$LITELLM_BASE/model/info" \
                    -H "Authorization: Bearer $LITELLM_KEY" \
                  | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)["data"]
except Exception:
    raise SystemExit
for m in d:
    n = m.get("model_name", "")
    if n.startswith("dbg-"):
        print(n)' 2>/dev/null)"})
  if (( ! ${#probes} )); then
    print -r -- "  (proxy not answering /model/info -- nothing to probe)"
    return 1
  fi
  for m in $probes; do
    t0=$EPOCHREALTIME
    got=$(curl -s --max-time 60 "$LITELLM_BASE/v1/chat/completions" \
            -H "Authorization: Bearer $LITELLM_KEY" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$m\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
          | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print("unreachable"); raise SystemExit
if "error" in d:
    msg=str(d["error"].get("message",""))
    # Three distinct states, only the last is a real problem:
    #   QUOTA SPENT  - provider refused, 429. Free tiers here are capped per DAY.
    #   COOLING DOWN - LiteLLM parked it locally after failures (cooldown_time).
    #                  It never left the machine, so this is our state, not theirs.
    if "No deployments available" in msg: print("COOLING DOWN")
    elif "ate limit" in msg or "429" in msg:  print("QUOTA SPENT")
    else: print("ERR "+msg[:40])
else:
    print("ok")' 2>/dev/null)
    printf '  %-17s %-14s %.1fs\n' "$m" "$got" "$(( EPOCHREALTIME - t0 ))"
  done
}
