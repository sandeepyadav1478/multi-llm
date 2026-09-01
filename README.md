# multi-llm — spend Anthropic quota on judgement, not on bulk text

`/effort xhigh` on Opus is the right setting for work that needs thinking, and an
expensive one for work that does not. This directory is the machinery for telling
the two apart: free local and cloud models absorb the chores, Anthropic keeps the
reasoning, and **the Anthropic side is always the parent** — nothing here runs
unsupervised.

The proxy those free models sit behind is configured next door in
[`../litellm/`](../litellm/README.md).

---

## Layout

```
multi-llm/
├── multi-llm.sh          shell functions, sourced from ~/.zshrc
├── bin/
│   ├── llm               one-shot prompt to a free/local model
│   ├── offload-hook      PreToolUse hook: blocks bulk reads, names the pipe
│   └── ccacct            switch between Claude Pro accounts
├── skills/offload/       the offload skill (symlinked into ~/.claude/skills/)
├── bench/
│   ├── bench.py          prose-quality benchmark + MLflow results
│   └── tools.py          tool-calling capacity + Claude Code compatibility
└── docs/
    ├── measurements.md   what the backends actually do under real load
    └── model-quality.md  what they can be trusted with, measured
```

---

## The three ways to get work done

| | Runs on | Use for |
|---|---|---|
| `claude` / `cc` | Anthropic Pro, Opus | Anything needing judgement. |
| `ccchore` | Anthropic parent + free workers | Log-reading, summarising, drafting sessions. Same model and effort, but offloading is on by default. |
| `llm "..."` | Free cloud or local | A single chore, right now, from the shell. |

Plus two fallbacks when quota runs out entirely: `ccacct use <name>` switches to
the other Pro subscription, and `ccfree` / `cclocal` drive Claude Code itself off
free models (see [`../litellm/README.md`](../litellm/README.md)).

**Those last two only work through the model groups, never pinned to a single
small backend.** Claude Code sends its whole tool array in every request — 108
tools with tokensave loaded is **~9.8K tokens**, ~12K with a system prompt,
before the user has typed anything.

Measured at that full load, all five criteria, ten `dbg-*` aliases: **six pass at
100%** — cerebras, sambanova, all three OpenRouter aliases, and `local` (vibeforged-14b
on ollama, 3.4s median). No backend has a tool-*count* ceiling below 108; both
failure modes are about size. `zai-glm-4.7` handles 80 tools then refuses 108
with `ContextWindowExceededError`, and the whole groq family is priced out on
tokens-per-minute. All four production groups — `default`, `smart`, `fast`,
`local` — took the load and answered correctly, because each carries a backstop.

Those are single-turn scores, and Claude Code is a loop. `tools.py --loop` sends
a tool *result* back and grades the continuation: **9 of 9 reachable backends**
complete a real call → result → answer cycle and quote the result rather than
guessing it (sambanova was quota-spent, not broken). And because all of that
tests `/v1/chat/completions` while Claude Code talks `/v1/messages` — a different
translation inside LiteLLM — `cclocal` was finally run for real: it read a file
and reported it correctly, then drove `Glob` → `Read` → `Write` to compute a sum
and write the answer to disk. Both exited 0. **Claude Code works end-to-end on a
14B model on your own GPU, at zero quota.**

Context is not the only ceiling. Groq's free tier bills **tokens per minute**,
and one full Claude Code request is bigger than a minute: `qwen3.6-27b` (the
`fast` alias) allows 8,000/min and the request needs 12,993 — **162% of the
whole budget**, so it fails on a completely idle account. `llama-3.3-70b` allows
12,000 and the request needs 11,747, which is 98% of a minute per turn. Both
models have plenty of context; neither has the throughput. Check the per-minute
allowance too, not just the window.

Details and the sweep: [`docs/model-quality.md`](docs/model-quality.md) §1b.

### `llm` — one chore, one command

```bash
llm "explain this stack trace"
git diff | llm "write a commit message"      # stdin is appended
cat build.log | llm "summarise the errors"   # the 50k never enters any context
llm -m local "summarise these logs"          # no quota, no network
llmcheck                                     # what is actually alive right now
```

`local` needs ollama up. `ollama.service` is `disabled`, which is **boot
autostart only** — the service itself runs normally and serves models fine, and
that is how `local` scored 5/5 on the full Claude Code tool array. Start it by
hand after a reboot:

```bash
sudo systemctl start ollama     # disabled at boot on purpose; start on demand
```

The idle service costs almost nothing; VRAM is held by the *model*, which ollama
loads on demand and drops again after its keep-alive expires. `llmcheck` prints
`ollama : active` or `inactive` on its second line, before probing any backend.

### `ccchore` — a session that offloads by default

A normal Anthropic session that treats offloading as the default rather than
something to remember. It appends one instruction to the system prompt; the model
and `/effort` are untouched.

Plain `claude` still has the `offload` skill available and will reach for it when
it fits — `ccchore` just stops that being a judgement call on every turn.

---

## Why the pipe matters

This is the entire saving, and it is not subtle:

```bash
cat build.log                        # ~50,000 tokens into context
cat build.log | llm "summarise"      # ~300 tokens into context
```

Context persists. Reading a log to shorten it means paying for that log on *every
remaining turn of the session*, not once. Piping it means the raw text never
enters an Anthropic context at all.

That claim used to be an argument. It's now a measurement: every `llm` call
appends a line to `bench/usage.jsonl` — backend, payload size, reply size,
latency, and whether it worked — and `uv run bench.py --usage` totals it up.

```bash
cd bench && uv run bench.py --usage    # what offloading actually did, no requests
```

It reports two numbers rather than one percentage: tokens that went to a free
backend, and tokens that came back. The gap is the saving, and it is an
*under*statement — the payload would have been charged on every later turn,
the summary is charged once. Turning that into a single ratio would need a
guess at turn count, which nothing here measures. The per-alias table is also
the honest answer to "am I actually using this, or just admiring it".

### That measurement said "no", so this is now a hook

`--usage` is only worth having if you act on what it says, and what it said was
damning. Measured 2026-07-29: `rtk` — a hook — had saved **353K tokens over
~2,150 commands**. Offloading, which was a *rule* asking the model to remember to
pipe, had **7 calls and no measurable saving**. Same idea, same backends, two
orders of magnitude apart. The only difference is that nobody has to remember a
hook.

So `bin/offload-hook` runs on `PreToolUse` and blocks a `Read`, or a bare
`cat`/`tail`, of a log over **8 KB** — with the exact pipe to run instead.

The 8 KB is measured, not chosen. Across 112 sessions and 5,930 tool calls the
median tool result is **135 characters** and only **6.2%** clear 4,000; that tail
is the entire offloadable population, and its own median is 8.4K for `Read`.
The threshold was 25 KB first — three times the tail's median, so it would have
fired almost never, which is precisely the failure it exists to prevent. 8 KB
also happens to be what the `large` tier already tests and still fits `fast`
(3,500 input) and `smart` (4,000), so no tier gets silently swapped out by the
router mid-request.

What it will not touch: source, JSON, YAML, lockfiles, small files, ranged reads
(`offset`/`limit`), or anything with stdout already redirected to a file. Test
and build runners are *advised*, never blocked — the failing test name is usually
the whole point of the run. Escape hatches are `OFFLOAD_SKIP=1 <cmd>` or a ranged
read. `bin/offload-hook --selftest` runs 21 assertions covering exactly these.

## What is safe to hand over

Measured, not assumed — see [`docs/model-quality.md`](docs/model-quality.md).
Summarising and short spec'd writing score **97–99%** on the three cloud
backends, measured over three reps (`local` is well behind at 88%, and — more to
the point — swings between 73% and 98% run to run). The hard exclusions, each
from an observed failure:

- **anything whose answer is a number** — asked to count 30 items, backends
  answered 30, 29 and 25;
- **the small models, on big payloads** — those 97–99% scores were earned on
  prompts under 2.5K characters. Re-measured at a realistic 8K–30K of tool
  output, `llama-3.1-8b` falls to the floor and `llama-3.3-70b` to 80%. Payload
  size is routed automatically (30K lands on `gpt-oss-120b` or `local`, the only
  backends with the headroom), so this costs quality, not correctness — but do
  not assume a short-prompt score holds at log scale;
- anything a decision depends on being exactly right;
- code changes, security, correctness calls;
- anything needing the conversation's context — `llm` is a stateless one-shot.

## Re-measuring

```bash
cd bench && uv run bench.py            # all backends that are up
uv run bench.py --models dbg-local
uv run bench.py --tiers all            # + the 4 smoke evals; onboarding a backend
uv run bench.py --tiers large          # the 8K-30K tool-output evals; off by default
uv run bench.py --evals large_inject   # one eval, without paying for its tier
uv run bench.py --routing              # does each tier still serve its own traffic?
uv run tools.py --counts 1,15,80       # how many tools can each backend handle?
uv run tools.py --counts 108 --cc-system   # can it drive Claude Code at all?
uv run bench.py --report               # compare past runs, no requests
uv run bench.py --regrade all          # re-score stored answers, no requests
srv start mlflow-server                # the UI does not run by default
xdg-open http://127.0.0.1:5000         # results
```

`--regrade` is the one worth knowing about. Editing a grader used to retire every
number measured under the old one; it doesn't have to, because a stored answer
stays valid as long as the *question* is unchanged. It re-scores what is already
in the store with today's graders, spends **zero** requests, and logs the delta
next to the original rather than over it — so a grader fix gets priced instead of
costing a whole re-run. Prompt identity is checked per eval, exactly where the
run recorded a hash and by length where it didn't (tagged, so the weaker claim
stays visible). Both `--report` and `--regrade` respect `--tiers`.

`--routing` answers a question the benchmark cannot: *is the router still sending
each tier's traffic to the backend that tier's score was earned on?* Every eval
runs against a 1:1 `dbg-*` alias, deliberately, so a score names one deployment —
which means the suite never exercises the production groups at all. It found a
live downgrade: a 25 KB payload sent to `smart` was being served by the local
14B, because LiteLLM's pre-call filter silently drops deployments whose declared
context is too small and nothing warns. `--routing` probes each alias at 2 KB and
again at the hook's own threshold (which it reads from `bin/offload-hook`, rather
than copying the number) and flags any tier whose large payload lands somewhere
its small payload never does. It probes the small size twice on purpose: a group
holds several deployments and weighted routing can legitimately answer from
either, so only a large payload escaping that set is a real capability drop.

`--routing` also sends a second probe the size check cannot substitute for: a
real Claude Code tool array with a *tiny* message. The pre-call filter **does**
count `tools` — measured 2026-07-29, `smart` serves `zai-glm-4.7` on 5 of 6 tiny
requests and 0 of 6 once 108 schemas are attached, with no 400 in between — so a
group holding a big-context member routes the array there automatically. The
probe exists because that only works when the group *has* one: a tier pinned to a
single small deployment has nothing to steer to and 400s from the provider. It
imports the array from `tools.py` rather than keeping a copy, and flags any tier
whose serving deployment declares less context than the array needs.

`--evals` narrows a run to specific eval ids inside `--tiers`. Without it the
smallest runnable unit is a whole tier — in `large`, four requests per backend
against a daily free cap. Note that a subset run gets its own digest
automatically, which is correct and not a side effect: a one-eval run *is* a
different ruler and must not sit in the same table as a full sweep.

One warning that cost a wrong claim to learn: **`--reps` does not make a
pass/fail result more trustworthy.** Repeats inside one run are separate requests
but correlated outcomes — over 20 stored runs, `sum_inject` scored 1.0 or 0.4 and
never once landed in between, which independent draws would do most of the time.
`--reps 10` on an injection eval reads as n=10 and is n≈1. Use reps for latency
spread; to test a behaviour, repeat the *run*.

Doing exactly that on 2026-07-29 moved a safety number the wrong way. Ten more
runs put `local`'s prompt-injection hijack rate at **9/29 ≈ 31%**, not the ~1-in-9
previously recorded — when it fails, the whole reply is the bare injected token
`PWNED`. Cerebras `gpt-oss-120b`, `zai-glm-4.7` and groq `qwen3.6-27b` are 0/4.
So: **do not route untrusted text through `local` when a cloud route is alive.**
`local` is the no-quota fallback, not the default for `some_command | llm`.

The UI is **two** systemd user services — `mlflow-server` on :5000 and
`mlflow-ui` on :5001 — config in `~/ai-configs/mlflow/`, managed from `srv`
alongside litellm-proxy, both pointed at this same sqlite file. They are the
same binary (`mlflow ui` and `mlflow server` are one command in MLflow 3); the
pair exists so one can be restarted without taking the other down.

**Neither starts at boot.** Both are deliberately disabled, so `srv start
mlflow-server` (or `mlflow-ui`) comes before the URL answers. `bench.py` writes
the sqlite file directly and never talks to either, so benchmarks don't care
whether they're running. Runs, per-response tables and **traces** (prompt, reply
and grade for every graded attempt) are all there.

Three tabs are worth knowing: **Traces** goes from a score to the exact prompt
and reply that earned it; **Prompts** holds the versioned system prompt; and
**Models** lists one entry per `dbg-*` alias with the endpoint it currently
resolves to and its score history — the "what is this alias and has it ever been
good" view. Treat Models as an index, not a trend line: a backend's points span
every eval set it was ever run against, and `--report` is where comparability is
actually enforced.

Don't run a bare `mlflow ui`. With no `--backend-store-uri` it reads `./mlruns`,
which here holds artifacts and no run metadata, so it fails *silently* — a
healthy, completely empty UI while the real data sits in `mlflow.db`. Use the
services, or pass the store yourself on a free port:

```bash
mlflow ui -p 5002 --backend-store-uri sqlite:////home/sandeep/ai-configs/multi-llm/bench/mlflow.db
```

`bench.py` pulls MLflow itself (PEP 723 inline deps), but the `mlflow` CLI the
service runs is separate. Install it once, **pinned to Python 3.13**:

```bash
uv tool install --python 3.13 mlflow==3.14.0
```

Both pins matter. On Python **3.14** the server fails at import — mlflow's
`skill_installer` imports `importlib.abc.Traversable`, which 3.14 removed — and
since `uv tool install` takes the newest interpreter it finds, the unpinned
command installs a `mlflow` that cannot start `ui` or `server`. The library path
is unaffected, so `bench.py` works either way and the breakage only shows up
when you go to look at the results. The version pin keeps the CLI matched to the
library that wrote the store, so the sqlite schema is never migrated out from
under `bench.py`.

Worth re-running after adding a backend, changing a model version, or when a
provider's free tier changes shape. Quota-spent backends are recorded as skipped
rather than scored zero, so a later run fills the gaps instead of repeating work.

A row reading `COOLING DOWN` means the **provider** rate-limited it, not that
anything here is misconfigured — LiteLLM masks the upstream 429 behind "no
deployments available". It is retried on a backoff and is usually gone within a
minute; if it persists, check the provider directly rather than the config
(`docs/model-quality.md` has the one-liner and the three real causes). Note that
Groq's limit is per **key**, so several `dbg-groq-*` backends in one sweep will
starve whichever runs last.
