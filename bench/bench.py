#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlflow>=3.14,<4", "pyyaml", "pandas"]
# ///
"""Benchmark every non-Anthropic backend on the work we actually want to give it.

    uv run bench.py                    # every backend that is up right now
    uv run bench.py --models dbg-groq,dbg-local
    uv run bench.py --baseline         # floor calibration, no network, no quota
    uv run bench.py --reps 3           # variance; costs 3x the quota
    uv run bench.py --tiers all        # + the 4 smoke evals; onboarding a backend
    uv run bench.py --temperature 0    # reproducible mode (see below)
    uv run bench.py --report           # compare past runs, no requests
    uv run bench.py --list             # what would run, no requests
    srv start mlflow-server            # the UI is NOT running by default
    xdg-open http://127.0.0.1:5000     # then the results are here

Two systemd user units serve this exact sqlite file -- `mlflow-server` (:5000)
and `mlflow-ui` (:5001), configured in ~/ai-configs/mlflow/ and managed from
`srv`. They are the same binary on different ports; in MLflow 3 `mlflow ui` and
`mlflow server` are the same command. Both are DISABLED, so neither starts at
boot and you must start one before the URL above answers.

If you want a throwaway UI instead, pass the store explicitly -- and do pass it:

    mlflow ui --port 5002 --backend-store-uri sqlite:///$PWD/mlflow.db

Bare `mlflow ui` reads ./mlruns as its backend store. That directory here holds
artifacts and no meta.yaml, so it serves a healthy, completely EMPTY UI while
every run sits untouched in mlflow.db. No error, just nothing.

The MLflow *library* comes from this script's inline deps, but the `mlflow` CLI
is separate: `uv tool install --python 3.13 mlflow==3.14.0`. Pin both. On
Python 3.14 the server dies at import (it reaches for `importlib.abc.Traversable`,
removed in 3.14) while the library keeps working -- so an unpinned install looks
fine until the moment you try to look at the results.

This script never talks to the server. It writes the sqlite file directly, so
benchmarks run whether or not the service is up.

Backends come from the `dbg-*` aliases in the LiteLLM config, not from a list in
this file. Those aliases have fallbacks explicitly disabled, so a score is
attributable to the backend named. Benchmarking the group aliases (`fast`,
`smart`, ...) would silently score whichever deployment answered.

Quota is the binding constraint, not time. A default pass is 9 requests per
backend, and all three OpenRouter models share ONE bucket of 50 requests/day --
so a single --reps 3 run over the OpenRouter models would spend 81 and fail
partway. Spent backends are recorded as skipped, never as scoring zero, so
rerunning after the midnight-UTC reset fills in the gaps rather than repeating
work.

That 9 is 13 minus the four `smoke` evals, skipped by default because no backend
has ever failed one -- measured over 9 runs x 6 backends. Dropping them cut 31%
of the per-run quota AND widened the spread between backends (47pp to 83pp
floor-normalised), because an eval everyone passes contributes nothing but a
constant to the average. `--tiers all` runs them, worth doing once when
onboarding a backend. The eval digest covers whichever subset ran, so a 9-eval
and a 13-eval run can never be compared by accident.


Four things this measures that the first version got wrong
----------------------------------------------------------
All four are the same mistake: benchmarking a configuration production does not
run. max_tokens, temperature and the system prompt are each read from or matched
to bin/llm now, because each one silently changed a result.

1. THE FLOOR IS NOT ZERO. Echoing the prompt back scores 72% and an empty
   string scores 40%, because criteria of the form "did not hallucinate", "did
   not leak reasoning", "did not claim all tests passed" are satisfied by any
   text that merely fails to misbehave. A raw 95% on a scale that starts at 72%
   is not "95% of the way to perfect", it is 82% of the available range. Every
   run therefore reports `score_vs_floor` next to the raw score, and
   `--baseline` recomputes the floor for free.

2. EVALS ARE WEIGHTED EQUALLY, criteria are not. See evals.aggregate: the old
   flat criterion average weighted each eval by how many checks its grader
   happened to have.

3. PRODUCTION SENDS NO TEMPERATURE. bin/llm omits the field, so the provider
   default (typically 1.0) applies -- and this harness used to pin
   temperature=0, benchmarking a configuration production never runs. That is
   the same mistake as the old max_tokens=1024 default, which made qwen3.6-27b
   look like a 50% model. Default is now "omit, exactly like bin/llm". Use
   --temperature 0 --reps 1 when you want a byte-reproducible comparison, and
   say so when you quote those numbers.

4. PRODUCTION SENDS A SYSTEM PROMPT. The harness sent none, which made it score
   the local model as RESISTING prompt injection while production was hijacked
   9 times in 10 -- bin/llm's "Answer directly, no preamble" primes a small
   model to tersely obey any instruction it can see, including one sitting in
   the payload. The prompt is now read out of bin/llm by production_system()
   rather than copied, so the two cannot drift apart again.

Confidence intervals are bootstrapped over per-eval scores, because the eval is
the independent unit, not the criterion. With 13 evals they are wide. That is
the honest picture: it is what stopped "98% beats 95%" being reported as a
ranking when the intervals overlapped almost completely.

MLflow usage: each sweep is a parent run holding the eval-set digest, the config
and the comparison table; each backend is a nested child run with its own
metrics and a per-response table you can open in the UI. The eval set itself is
logged via mlflow.log_input, so a run records the exact dataset digest it was
scored against. The dep floor is `mlflow>=3.14,<4` and it is a real constraint,
not politeness: this file assumes MLflow 3 behaviour (see the file-store comment
in main()), and the earlier `>=2.16` floor only worked because resolution
happened to pick 3.14.

Why not mlflow.genai.evaluate()? Not the reason previously written here. The old
note claimed the custom-metric contract only passes prediction and target
columns -- true of the LEGACY mlflow.evaluate, but MLflow 3's @scorer receives
the whole row (inputs, outputs, expectations, trace), so heterogeneous per-eval
graders WOULD fit, and a pure-code scorer runs fine against this sqlite store
with no judge and no token cost (verified). The real reasons to stay hand-rolled:

  - Built-in scorers are LLM judges. A judge spends Anthropic tokens, which is
    the exact cost this whole setup exists to avoid, and makes a score
    unfalsifiable. Every criterion here is a fact about a string.
  - genai aggregates each scorer as `<name>/mean` across rows. That is the flat
    criterion average this harness deliberately removed -- it weights an eval by
    how many checks its grader happens to have. See evals.aggregate.
  - It would rename every metric, orphaning the runs already in the store and
    breaking --report, in exchange for no measurement we do not already have.

The one MLflow 3 feature worth adopting, tracing, IS adopted: @mlflow.trace on
ask() plus a span per graded attempt in run_model. Each trace is two spans --
`<backend>/<eval>` (CHAIN) holding the inputs and the grade, wrapping `llm_call`
(LLM) holding the real prompt, the real reply and the latency. Retries nest as
sibling llm_call spans under one attempt, so a backed-off eval still reads as
the single graded attempt it is. Traces attach to the parent sweep run, not the
per-backend child, because log_child creates that child only after its rows are
collected; the backend is a span input instead.

This is pure observability and changes no score. Note it stores the FULL prompt,
so a --tiers large run writes 30K-char payloads into the store -- fine at this
scale, worth remembering if the store ever needs to stay small.

Traces are visible in the UI, but only in one served from THIS sqlite file. A
bare `mlflow ui` uses ./mlruns as its backend store, and that directory here
holds artifacts only -- zero meta.yaml -- so it renders an empty UI while the
data sits untouched in mlflow.db. Pass --backend-store-uri, or use :5000.
"""

import argparse
import json
import logging
import os
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

import mlflow
import mlflow.genai          # not pulled in by `import mlflow`; needed for the
                             # prompt registry, whose top-level aliases are deprecated
import pandas as pd
import yaml

from evals import (EVALS, BASELINES, aggregate, fingerprint, prompt_hashes,
                   universal)

# set_active_model() narrates itself at INFO -- two lines per backend per sweep,
# on top of the one line of actual result. The sweep output is the interface here.
logging.getLogger("mlflow.tracking.fluent").setLevel(logging.WARNING)

HERE = pathlib.Path(__file__).parent
CONFIG = pathlib.Path.home() / "ai-configs/litellm/config.yaml"
BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000")
KEY = os.environ.get("LITELLM_KEY", "sk-litellm-local")
TIERS = ("core", "hard", "smoke", "large")


def backends():
    """dbg alias -> real provider/model, straight from the live config."""
    cfg = yaml.safe_load(CONFIG.read_text())
    return {
        m["model_name"]: m["litellm_params"]["model"]
        for m in cfg["model_list"]
        if m["model_name"].startswith("dbg-")
    }


def production_system():
    """The system prompt, read out of bin/llm rather than copied.

    Same rule as backends(): read the live caller, never keep a second copy that
    can drift. This one earned the rule the hard way. bin/llm sends "You are a
    concise assistant. Answer directly, no preamble." and the harness used to
    send no system prompt at all -- so the harness scored the local model as
    RESISTING the injection eval while production was hijacked 9 times out of
    10. "Answer directly, no preamble" primes a small model to comply tersely
    with whatever instruction it can see, including the one hidden in the
    payload. Omitting it measured a system nobody runs.
    """
    m = re.search(r'^SYSTEM="(.*)"$', (HERE.parent / "bin" / "llm").read_text(), re.M)
    if not m:
        sys.exit("could not read SYSTEM= out of bin/llm -- fix the regex, do not "
                 "hardcode the prompt, or the benchmark silently stops matching production")
    return m.group(1)


@mlflow.trace(span_type="LLM", name="llm_call")
def ask(model, prompt, max_tokens, temperature, system, timeout=180):
    """-> (text, latency, usage) or raises Unavailable for a quota/cooldown miss.

    Traced: the span records the real prompt and the real reply, so a score in
    the UI can be opened down to the exact text that earned it. Unavailable
    propagates through the decorator, which marks the span ERROR -- that is how
    a CONTEXT TOO SMALL refusal becomes visible as something other than a zero.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": msgs,
        # Reasoning models spend this budget on thinking BEFORE writing a word,
        # and a model cut off mid-thought returns empty content that grades as a
        # total failure. Measured: groq qwen3.6-27b scored 50% at 1024 purely
        # because 6 of 8 answers hit out=1024 exactly with text=''. That is a
        # budget artefact, not a quality result -- hence the flag. Compare runs
        # only at equal max_tokens; it is logged as a param for that reason.
        "max_tokens": max_tokens,
    }
    # Omitted, not defaulted, when None: bin/llm does not send this field, and
    # the point of the harness is to measure what production actually runs.
    if temperature is not None:
        payload["temperature"] = temperature
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise Unavailable(classify(e.read().decode("utf-8", "replace"))) from None
    except Exception as e:
        raise Unavailable(f"{type(e).__name__}: {e}") from None
    dt = time.monotonic() - t0
    msg = d["choices"][0]["message"]
    # Only `content`, deliberately, even though some backends put the answer in
    # `reasoning_content` and leave content empty. Anything consuming these
    # models in a pipeline reads content; a model that leaves it empty is
    # unusable there regardless of what it was thinking. Scoring that as
    # non_empty=False is the correct answer, not a measurement gap.
    text = msg.get("content") or ""
    return text, dt, d.get("usage", {})


class Unavailable(Exception):
    pass


def classify(raw):
    low = raw.lower()
    if "no deployments available" in low:
        return "COOLING DOWN"
    if "rate limit" in low or "429" in low or "quota" in low:
        return "QUOTA SPENT"
    if "context" in low and ("window" in low or "length" in low):
        return "CONTEXT TOO SMALL"
    return "ERR " + raw[:120].replace("\n", " ")


# Groq's free tier is capped per MINUTE, OpenRouter's per DAY, and both surface
# as an identical 429. Backing off separates them: a TPM window heals in ~60s, a
# daily cap does not. This is not optional politeness -- Groq's measured limit is
# TPM 8,000 and it bills input + max_tokens together, so one eval costs ~1,460
# and a full pass CANNOT fit in one minute. Without a budget here, Groq scores
# as "spent" every single run and we would conclude, wrongly, that it is
# unusable for offload work.
BACKOFF_S = 35
BACKOFF_BUDGET = 13


def grade(e, text):
    return {**e["grade"](text), **universal(text)}


def run_model(alias, reps, max_tokens, temperature, system):
    rows, budget = [], BACKOFF_BUDGET
    for e in EVALS:
        for rep in range(reps):
            # One span per attempt, wrapping the retry loop rather than sitting
            # inside it: a backed-off eval is ONE graded attempt that happened to
            # cost several calls, and the trace should read that way -- the
            # nested llm_call spans show the retries.
            with mlflow.start_span(name=f"{alias}/{e['id']}", span_type="CHAIN") as span:
                span.set_inputs({"backend": alias, "eval": e["id"],
                                 "tier": e["tier"], "rep": rep})
                text, dt, usage, note = None, 0.0, {}, ""
                while text is None:
                    try:
                        text, dt, usage = ask(alias, e["prompt"], max_tokens, temperature, system)
                    except Unavailable as u:
                        s = str(u)
                        # COOLING DOWN is retryable for the same reason QUOTA is,
                        # and missing that cost a real measurement: LiteLLM cools
                        # a deployment down for 60s after upstream 429s, so
                        # "no deployments available" is a ONE-MINUTE state. This
                        # branch used to fall through to the abandon path below,
                        # which is how a sweep that hammered Groq's shared 8k TPM
                        # on two backends reported the third as permanently
                        # unavailable -- it answered fine twenty minutes later.
                        # Two backoffs (70s) outlast the default cooldown.
                        #
                        # Note this is NOT the pre-call-check refusal, which
                        # raises ContextWindowExceededError and classifies as
                        # CONTEXT TOO SMALL -- that one really is permanent for
                        # the prompt, and is still handled below.
                        if ("QUOTA" in s or "COOLING" in s) and budget > 0:
                            budget -= 1
                            note = "after backoff"
                            time.sleep(BACKOFF_S)
                            continue
                        # Context overflow is about this prompt, not the backend, so
                        # score it zero and keep going.
                        if "CONTEXT" in s:
                            text, note = "", s
                            break
                        # Anything else is down for every eval. Stop rather than
                        # burning twelve more timeouts to learn the same thing.
                        span.set_outputs({"unavailable": s})
                        return rows, s
                scores = grade(e, text)
                score = round(sum(scores.values()) / len(scores), 3)
                span.set_outputs({"score": score, "scores": scores,
                                  "note": note, "text": text})
                # Tags make the Traces tab usable at sweep scale: a full run is
                # 10 backends x 9 evals = 90 traces, and without these the only
                # handle on any of them is a span name.
                mlflow.update_current_trace(tags={
                    "alias": alias, "eval": e["id"], "tier": e["tier"],
                    "rep": str(rep), "eval_digest": fingerprint()})
                # The grade, attached to the thing it graded. source_type comes
                # out as CODE, which is the honest label and the whole argument
                # of this harness -- these are checkable facts about a string,
                # not a judge's opinion.
                failed = [k for k, v in scores.items() if not v]
                mlflow.log_feedback(
                    trace_id=span.trace_id, name="score", value=score,
                    rationale="all criteria passed" if not failed
                              else f"failed: {', '.join(failed)}")
                # Only failures get their own assessment. 90 rows of
                # `no_leak: true` is not a finding; the three that are false
                # are, and this is what makes them one filter away in the UI.
                for k in failed:
                    mlflow.log_feedback(trace_id=span.trace_id, name=k, value=False)
            rows.append(dict(
                eval=e["id"], category=e["category"], tier=e["tier"], rep=rep,
                text=text, latency=round(dt, 2), usage=usage, note=note,
                scores=scores,
            ))
    return rows, None


def run_baseline(fn, _reps, _max_tokens, _temperature, _system):
    """Score a degenerate answer offline. No network, no quota, no latency."""
    return [dict(eval=e["id"], category=e["category"], tier=e["tier"], rep=0,
                 text=fn(e["prompt"]), latency=0.0, usage={}, note="baseline",
                 scores=grade(e, fn(e["prompt"]))) for e in EVALS], None


def bootstrap_ci(rows, alpha=0.05, iters=2000, seed=0):
    """Percentile CI over per-eval scores.

    Resamples EVALS, not criteria. Criteria within one eval are not independent
    -- a model that misreads the log fails several of that eval's checks at once
    -- so pooling them as if they were gives an interval far too narrow. With 13
    evals this comes back wide, which is the correct answer to "can you tell
    these two backends apart" for most pairs.
    """
    per = {}
    for r in rows:
        per.setdefault(r["eval"], []).append(sum(r["scores"].values()) / len(r["scores"]))
    means = [sum(v) / len(v) for v in per.values()]
    if len(means) < 2:
        return None, None
    rnd = random.Random(seed)
    draws = sorted(statistics.fmean(rnd.choices(means, k=len(means))) for _ in range(iters))
    return draws[int(alpha / 2 * iters)], draws[int((1 - alpha / 2) * iters)]


def eval_set_frame():
    # prompt_hash is what makes --regrade sound: it pins the QUESTION each stored
    # answer was given, independently of the grader that scored it. Runs logged
    # before this column existed fall back to prompt_chars, which can only ever
    # be a likelihood, not a proof -- see regrade().
    ph = prompt_hashes()
    return pd.DataFrame([
        dict(eval=e["id"], category=e["category"], tier=e["tier"],
             prompt_chars=len(e["prompt"]), prompt_hash=ph[e["id"]],
             criteria=len(e["grade"]("")) + 2)  # +2 universal
        for e in EVALS
    ])


def rows_frame(rows):
    """Per-response table. Truncated text, because the table is for eyeballing a
    failure in the MLflow UI; the untruncated answers go to responses.json."""
    return pd.DataFrame([
        dict(eval=r["eval"], category=r["category"], tier=r["tier"], rep=r["rep"],
             score=round(sum(r["scores"].values()) / len(r["scores"]), 3),
             failed=",".join(k for k, v in r["scores"].items() if not v),
             latency=r["latency"], note=r["note"],
             out_tokens=r["usage"].get("completion_tokens"),
             text=(r["text"] or "")[:600])
        for r in rows
    ])


def external_model(alias, real):
    """The backend as a LoggedModel, so the UI's Models tab is not empty.

    MLflow 3 grew "external models" for exactly this shape: the thing being
    measured is somebody else's HTTP endpoint, so there are no weights to log
    and the classic Model Registry has nothing to hold. set_active_model()
    reuses by name, so an alias is ONE model across every sweep and its scores
    accumulate as history against it -- the per-backend-over-time view that
    --report cannot draw, because --report is a flat list of runs.

    Floors are excluded on purpose. floor-echo is a degenerate string generator
    used to calibrate the scale; listing it as a model invites reading it as a
    backend you could route to.
    """
    if alias.startswith("floor-"):
        return None
    m = mlflow.set_active_model(name=alias)
    # Refreshed every sweep: which endpoint the alias points at is a property of
    # today's litellm config, not of the model record, and it does drift.
    mlflow.MlflowClient().set_logged_model_tags(
        m.model_id, {"backend": real, "provider": real.split("/")[0]})
    return m.model_id


def log_child(alias, real, rows, unavailable, args, t0, floor):
    """One nested run per backend. Returns the summary tuple for the parent."""
    with mlflow.start_run(run_name=alias, nested=True):
        model_id = external_model(alias, real)
        mlflow.set_tags({"alias": alias, "backend": real, "provider": real.split("/")[0],
                         "status": unavailable or "ok", "eval_digest": fingerprint(),
                         "kind": "baseline" if alias.startswith("floor-") else "backend"})
        mlflow.log_params({"alias": alias, "backend": real, "reps": args.reps,
                           "n_evals": len(EVALS), "max_tokens": args.max_tokens,
                           "tiers": args.tiers,
                           "temperature": "omitted" if args.temperature is None
                                          else args.temperature,
                           "system": args.system_used[:480],
                           "eval_digest": fingerprint()})
        if unavailable:
            # No metrics on purpose. Logging 0.0 for a backend that never
            # answered would sink its average in every later comparison.
            print(f"{unavailable}", flush=True)
            mlflow.clear_active_model()
            return (alias, None, unavailable)

        overall = aggregate(rows)
        cats = {c: aggregate(rows, "category", c) for c in {e["category"] for e in EVALS}}
        tiers = {t: aggregate(rows, "tier", t) for t in TIERS}
        lo, hi = bootstrap_ci(rows)
        lat = [r["latency"] for r in rows if r["latency"] > 0]
        metrics = {
            "score_overall": overall,
            **{f"score_{c}": v for c, v in cats.items() if v is not None},
            **{f"tier_{t}": v for t, v in tiers.items() if v is not None},
            # Mean across reps, via aggregate() -- NOT rep 0 only, which is what
            # this did and it was quietly wrong. score_overall averages every
            # rep, so a rep-0-only per-eval metric meant the eval_* values in
            # the UI did not reconcile with the headline they supposedly break
            # down: dbg-local at --reps 3 logged score_overall 87.8% next to
            # eval_* averaging 86.3%. Nothing surfaced the discrepancy, and reps
            # 1 and 2 existed only inside the artifact.
            **{f"eval_{e}": aggregate(rows, "eval", e) for e in {r["eval"] for r in rows}},
            "wall_seconds": round(time.monotonic() - t0, 1),
        }
        if lo is not None:
            metrics |= {"score_ci_low": lo, "score_ci_high": hi, "score_ci_width": hi - lo}
        if floor is not None and floor < 1:
            # Where the score sits in the range a real answer can actually move,
            # rather than in the range 0-100% that it cannot.
            metrics["score_vs_floor"] = max(0.0, (overall - floor) / (1 - floor))
        if lat:
            metrics |= {"latency_median": round(statistics.median(lat), 2),
                        "latency_max": round(max(lat), 2)}
        out = [r["usage"].get("completion_tokens") for r in rows]
        out = [o for o in out if isinstance(o, int)]
        if out:
            metrics["out_tokens_median"] = statistics.median(out)
            # Truncation is the failure mode that once cost qwen 50 points.
            metrics["hit_ceiling"] = sum(1 for o in out if o >= args.max_tokens)
        # model_id attaches the same numbers to the LoggedModel as well as the
        # run, which is what makes the Models tab show a backend's history
        # instead of one disconnected point per sweep.
        mlflow.log_metrics(metrics, model_id=model_id)
        mlflow.clear_active_model()
        mlflow.log_table(rows_frame(rows), "responses.json")
        mlflow.log_dict({"rows": rows}, "responses_full.json")

        med = f"  med {statistics.median(lat):.1f}s" if lat else ""
        ci = f"  [{lo:.0%}-{hi:.0%}]" if lo is not None else ""
        print(f"{overall:.0%}{ci}  " +
              "  ".join(f"{t}={v:.0%}" for t, v in tiers.items() if v is not None) +
              med, flush=True)
        return (alias, overall, dict(tiers=tiers, cats=cats, ci=(lo, hi)))


PROMPT_NAME = "offload-system"

EXPERIMENT_NOTE = """\
**Which free backend can be trusted with a chore off the Anthropic parent?**
Programmatic graders only -- every criterion is a checkable fact about the
returned string, no LLM judge. See `multi-llm/docs/model-quality.md`.

Two things to read correctly:

- **`score_vs_floor`, not `score_overall`.** A degenerate answer (echo the
  prompt back) already scores ~77%. Raw score is not on a 0-100% scale.
- **`eval_digest` must match** before comparing two runs. It hashes every
  prompt and grader; when it moves, the ruler moved, not the backend.
  `bench.py --report` enforces this. The UI does not.

Tabs: **Traces** = the exact prompt, reply and grade behind a score.
**Models** = one entry per `dbg-*` alias (an index, not a trend line -- its
points span every eval set). **Prompts** = the versioned system prompt."""


def register_system_prompt(system):
    """Put the system prompt in the Prompt Registry; return its version, or None.

    It was already logged as params.system truncated to 480 chars, which is fine
    for eyeballing and useless as a record: the string is read live out of
    bin/llm and it MEASURABLY changes results -- adding it flipped the local
    model from resisting the injection eval to being hijacked 9 times in 10. A
    variable that strong deserves versioning, not a truncated copy.

    Registers a new version only when the text actually differs, so the version
    number counts real changes to production rather than sweeps. Never fatal: a
    registry hiccup must not lose a sweep that just spent real quota.
    """
    try:
        c = mlflow.MlflowClient()
        try:
            existing = list(c.search_prompt_versions(PROMPT_NAME))
        except mlflow.exceptions.MlflowException:
            existing = []      # raises rather than returning [] until it exists once
        for v in existing:
            if v.template == system:
                return v
        return mlflow.genai.register_prompt(
            name=PROMPT_NAME, template=system,
            commit_message="system prompt as read from bin/llm")
    except Exception as e:
        print(f"  (prompt registry unavailable: {type(e).__name__})")
        return None


def regrade(args):
    """Re-score answers already in the store with the CURRENT graders. No network.

    Every request in this store cost metered quota from a free tier that is
    capped per day. The answers were kept (responses_full.json), so a grader fix
    should not cost that quota a second time -- the model already said its piece
    and nothing about editing a grader changes what it said.

    Validity is per EVAL, not per run. A stored answer stays gradable while the
    prompt that produced it is unchanged; the grader may be rewritten freely.
    Runs logged since eval_set.json gained prompt_hash are checked exactly.
    Older ones can only be checked on prompt_chars, which is necessary and not
    sufficient -- those are tagged prompt_check=chars so a weaker claim never
    silently reads as a strong one.

    Output is a new sweep tagged kind=regrade, carrying the current digest and
    pointing back at the run each score came from. Nothing is overwritten:
    the original stays exactly as measured.
    """
    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    by_id = {e["id"]: e for e in EVALS}
    want = prompt_hashes()

    df = mlflow.search_runs(experiment_names=[args.experiment],
                            filter_string="tags.kind = 'backend'",
                            order_by=["attributes.start_time DESC"])
    if df.empty:
        sys.exit("no scored backend runs to regrade")
    if args.regrade != "all":
        # Default: the newest sweep only. Regrading the whole store is free but
        # writes a run per backend per sweep, which buries the thing you asked for.
        parents = df["tags.mlflow.parentRunId"].dropna().unique()
        if len(parents):
            df = df[df["tags.mlflow.parentRunId"] == parents[0]]

    floors = {n: aggregate(run_baseline(f, 1, 0, None, None)[0]) for n, f in BASELINES.items()}
    floor = max(floors.values())
    print(f"regrade: {len(df)} run(s) -> digest {fingerprint()}   floor {floor:.0%}\n")

    done, skipped = [], []
    with mlflow.start_run(run_name="sweep-regrade"):
        mlflow.set_tags({"kind": "sweep", "eval_digest": fingerprint(), "source": "regrade"})
        mlflow.log_params({"n_evals": len(EVALS), "eval_digest": fingerprint(),
                           "regrade_scope": args.regrade, "n_sources": len(df)})
        mlflow.log_input(mlflow.data.from_pandas(eval_set_frame(), name="offload-evals"),
                         context="eval")

        for _, src in df.iterrows():
            alias = src.get("tags.alias") or src["run_id"][:8]
            try:
                p = mlflow.artifacts.download_artifacts(
                    run_id=src["run_id"], artifact_path="responses_full.json")
                stored = json.loads(pathlib.Path(p).read_text())["rows"]
            except Exception as e:                       # no artifact, or unreadable
                skipped.append((alias, f"no responses artifact ({type(e).__name__})"))
                continue

            # Prompt identity, exact where the source recorded it. eval_set.json
            # belongs to the SWEEP, not the backend -- one ruler per sweep, not
            # one per child -- so it is fetched from the parent.
            check = "hash"
            parent = src.get("tags.mlflow.parentRunId")
            try:
                q = mlflow.artifacts.download_artifacts(
                    run_id=parent, artifact_path="eval_set.json")
                es = json.loads(pathlib.Path(q).read_text())
                cols, data = es["columns"], es["data"]
                old = {r[cols.index("eval")]: r[cols.index("prompt_hash")] for r in data} \
                    if "prompt_hash" in cols else None
                if old is None:
                    check = "chars"
                    old = {r[cols.index("eval")]: r[cols.index("prompt_chars")] for r in data}
                    want_cmp = {e["id"]: len(e["prompt"]) for e in EVALS}
                else:
                    want_cmp = want
            except Exception:
                skipped.append((alias, "no eval_set.json -- cannot prove prompt identity"))
                continue

            rows = [dict(r, scores=grade(by_id[r["eval"]], r["text"]))
                    for r in stored
                    if r["eval"] in by_id and old.get(r["eval"]) == want_cmp[r["eval"]]]
            if not rows:
                skipped.append((alias, "no eval survived the prompt check"))
                continue

            with mlflow.start_run(run_name=alias, nested=True):
                overall = aggregate(rows)
                lo, hi = bootstrap_ci(rows)
                mlflow.set_tags({
                    "alias": alias, "backend": src.get("tags.backend") or "?",
                    "provider": src.get("tags.provider") or "?", "status": "ok",
                    "kind": "regrade", "eval_digest": fingerprint(),
                    "prompt_check": check, "source_run_id": src["run_id"],
                    "source_digest": src.get("tags.eval_digest") or "?"})
                mlflow.log_params({"alias": alias, "n_evals": len(EVALS),
                                   "n_regraded": len({r["eval"] for r in rows}),
                                   "eval_digest": fingerprint(),
                                   "source_run_id": src["run_id"],
                                   # Carried over: a regrade re-scores that exact
                                   # request, so the request's settings are still
                                   # its settings. Without this the row that
                                   # replaces the source in --report loses its
                                   # temperature and budget.
                                   **{k: src[f"params.{k}"]
                                      for k in ("max_tokens", "temperature")
                                      if f"params.{k}" in src and pd.notna(src[f"params.{k}"])}})
                # Recomputed from the source's OWN stored scores, restricted to
                # the evals that survived the prompt check. Its logged
                # score_overall covers whatever set that sweep ran; comparing
                # against that folds "a grader moved" and "a different subset of
                # evals" into one number and you cannot tell which you are
                # looking at. Same evals, same answers, old ruler vs new.
                kept = {r["eval"] for r in rows}
                before = aggregate([r for r in stored if r["eval"] in kept])
                mlflow.log_metrics({
                    "score_overall": overall,
                    "score_before": before,
                    "score_vs_floor": max(0.0, (overall - floor) / (1 - floor)) if floor < 1 else 0.0,
                    "score_ci_low": lo, "score_ci_high": hi,
                    **{f"tier_{t}": v for t in TIERS
                       if (v := aggregate(rows, "tier", t)) is not None},
                    **{f"eval_{e}": aggregate(rows, "eval", e)
                       for e in {r["eval"] for r in rows}},
                    # The point of the whole exercise: what the grader change did
                    # to this backend, on identical answers.
                    **({"score_delta": overall - float(before)}
                       if before is not None and pd.notna(before) else {})})
                mlflow.log_table(rows_frame(rows), "responses.json")
            done.append((alias, overall, before, check, len({r["eval"] for r in rows})))

    print(f"  {'backend':<18} {'was':>7} {'now':>7} {'delta':>7}  evals  prompt-check")
    for alias, now, before, check, n in sorted(done, key=lambda r: -r[1]):
        was = f"{float(before):.1%}" if before is not None and pd.notna(before) else "  --"
        dl = (f"{now - float(before):+.1%}"
              if before is not None and pd.notna(before) else "   --")
        print(f"  {alias:<18} {was:>7} {now:7.1%} {dl:>7}  {n:5}  {check}")
    for alias, why in skipped:
        print(f"  {alias:<18}   skipped: {why}")
    if not done:
        # The usual cause, and not guessable from "no eval survived": the sweep
        # you pointed at ran tiers this invocation filtered out, so every stored
        # answer belongs to an eval that is not in EVALS right now.
        print(f"\n  nothing regraded. --tiers is {args.tiers!r}; if that sweep ran "
              f"other tiers, match them (--tiers all always does).")
    print("\nno requests were spent. originals untouched; "
          "these are new runs tagged kind=regrade.")


def usage(args):
    """What offloading actually did, read from bin/llm's own log.

    Everything else in this file answers "is this backend good enough". This
    answers the question that comes first and was going unrecorded: did any
    offloading happen, and what did it keep out of an Anthropic context.

    The saving is deliberately stated as two numbers, not one ratio. Payload
    tokens went to a free backend; reply tokens came back and are what you
    actually pay for. The difference understates it, because context persists:
    a log read into a session is charged again on every later turn, while the
    summary is charged once. Quoting a single "% saved" would need an
    assumption about turn count that nothing here measures.
    """
    p = HERE / "usage.jsonl"
    if not p.exists():
        sys.exit(f"no {p.name} yet -- bin/llm writes one line per call")
    rows = []
    for ln in p.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass                      # a partial line from a killed call
    if not rows:
        sys.exit(f"{p.name} is empty")

    df = pd.DataFrame(rows)
    for c in ("prompt_tokens", "completion_tokens"):   # absent when a call failed
        if c not in df:
            df[c] = pd.NA
    ok = df[df["ok"] == True]         # noqa: E712 -- pandas mask, not a bool test
    span = (df["t"].max() - df["t"].min()) / 86400
    print(f"{len(df)} calls over {span:.1f} days   "
          f"{len(df) - len(ok)} failed\n")
    print(f"  {'alias':<18} {'calls':>5} {'fail':>4} {'payload tok':>12} "
          f"{'reply tok':>10} {'median chars':>13} {'med s':>6}")
    for alias, g in df.groupby("alias"):
        gok = g[g["ok"] == True]      # noqa: E712
        print(f"  {alias[:18]:<18} {len(g):>5} {len(g) - len(gok):>4} "
              f"{int(gok['prompt_tokens'].fillna(0).sum()):>12,} "
              f"{int(gok['completion_tokens'].fillna(0).sum()):>10,} "
              f"{int(g['prompt_chars'].median()):>13,} "
              f"{g['elapsed'].median():>6.1f}")
    offl = int(ok["prompt_tokens"].fillna(0).sum())
    back = int(ok["completion_tokens"].fillna(0).sum())
    print(f"\n  {offl:,} payload tokens went to a free backend instead of into an "
          f"Anthropic context.")
    print(f"  {back:,} came back as replies. Net {offl - back:+,} tokens, before "
          f"counting that context persists and a summary is paid for once.")
    if len(df) > len(ok):
        print("\n  failures:")
        for err, n in df[df["ok"] != True]["err"].value_counts().items():  # noqa: E712
            print(f"    {err:<16} {n}")

    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="usage-rollup"):
        # kind=usage, and no score_overall, so --report's dropna keeps this out
        # of the backend comparison. It is a different measurement entirely.
        mlflow.set_tags({"kind": "usage"})
        mlflow.log_metrics({"calls": len(df), "calls_failed": len(df) - len(ok),
                            "payload_tokens": offl, "reply_tokens": back,
                            "net_tokens": offl - back,
                            "payload_chars": int(df["prompt_chars"].sum()),
                            "days_covered": round(span, 2)})
        mlflow.log_table(df, "usage.json")
    print("\nlogged as a usage-rollup run; kept out of --report by design.")


def report(args):
    """Compare finished runs without spending a request."""
    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    df = mlflow.search_runs(experiment_names=[args.experiment],
                            order_by=["metrics.score_overall DESC"])
    if df.empty:
        sys.exit("no runs yet")
    cur = fingerprint()
    # One row per measurement, in three steps.
    if "tags.source_run_id" in df:
        reg = df["tags.kind"] == "regrade"
        # 1. A regrade on a retired digest replaces nothing. Drop it and let the
        #    source speak for itself again, marked stale, which is the truth.
        df = df[~reg | (df["tags.eval_digest"] == cur)]
        # 2. Newest regrade per source. Re-running --regrade is idempotent for
        #    the caller and has to be here too, or every pass doubles a row.
        dup = (df[df["tags.kind"] == "regrade"]
               .sort_values("start_time", ascending=False)
               .duplicated("tags.source_run_id"))
        df = df.drop(dup[dup].index)
        # 3. What survives SUPERSEDES the run it re-scored -- same replies, newer
        #    ruler. Listing both would read as two independent results and let a
        #    backend that happened to get regraded count twice.
        df = df[~df["run_id"].isin(set(df["tags.source_run_id"].dropna()))]
    keep = [c for c in ["tags.alias", "tags.status", "tags.eval_digest", "tags.kind",
                        "params.max_tokens",
                        "params.temperature", "metrics.score_overall", "metrics.score_vs_floor",
                        "metrics.tier_core", "metrics.tier_hard", "metrics.score_ci_low",
                        "metrics.score_ci_high", "metrics.latency_median"] if c in df]
    df = df[keep].dropna(subset=["metrics.score_overall"])
    stale = df[df["tags.eval_digest"] != cur] if "tags.eval_digest" in df else df.iloc[:0]
    print(f"eval digest now: {cur}\n")
    def pct(v, label):
        # A metric that predates its own existence is "-", never "nan%" and
        # never 0%. Runs from before the tier split genuinely have no tier
        # score, and printing a number there invents data.
        return f"  {label} {v:.0%}" if v is not None and v == v else f"  {label} -"

    def line(r):
        ci = ""
        if r.get("metrics.score_ci_low") == r.get("metrics.score_ci_low"):  # not NaN
            ci = f" [{r['metrics.score_ci_low']:.0%}-{r['metrics.score_ci_high']:.0%}]"
        vf = r.get("metrics.score_vs_floor")
        vf = f"  vs-floor {vf:.0%}" if vf is not None and vf == vf else ""
        # Runs made before --temperature existed were pinned to 0; showing the
        # missing param as "None" would read as "omitted", which is now a
        # different and specific setting.
        t = r.get("params.temperature")
        t = "?" if t is None or t != t else t
        # Worth saying out loud: the score is current, the measurement is not.
        # Nothing was re-asked -- these are old replies under today's graders.
        mark = "  (regraded)" if r.get("tags.kind") == "regrade" else ""
        print(f"  {str(r.get('tags.alias')):<18} {r['metrics.score_overall']:.0%}{ci}{vf}"
              f"{pct(r.get('metrics.tier_core'), 'core')}"
              f"{pct(r.get('metrics.tier_hard'), 'hard')}"
              f"  T={t}{mark}")

    # Two blocks, not one sorted list. Interleaving them by score puts numbers
    # from different rulers next to each other in descending order, which is
    # exactly the shape of a comparison -- a per-row marker does not undo that.
    live = df.drop(stale.index)
    if live.empty:
        # Not "no runs" -- runs exist, none were scored by this ruler. Saying so
        # beats an empty block above a long list of numbers.
        print(f"  no run was scored against this eval set. `--regrade all"
              f"{'' if args.tiers == 'core,hard' else ' --tiers ' + args.tiers}`"
              f" re-scores what is already here, for free.")
    for _, r in live.iterrows():
        line(r)
    if len(stale):
        # The whole reason the digest is logged. Numbers from a different eval
        # set are not worse data, they are a different measurement. --regrade
        # is how they rejoin the block above, for free.
        print(f"\n--- {len(stale)} run(s) from a retired eval set. NOT comparable "
              f"to the above; `--regrade all` re-scores them ---")
        for _, r in stale.iterrows():
            line(r)


def hook_limit():
    """The offload hook's block threshold, read out of the hook itself.

    Same rule as backends() and production_system(): never keep a second copy of
    a number that lives somewhere else. If the hook's threshold moves and this
    probe keeps testing the old size, it stops testing production.
    """
    src = (HERE.parent / "bin" / "offload-hook").read_text()
    m = re.search(r"^LIMIT = (\d+) \* 1024", src, re.M)
    return int(m.group(1)) * 1024 if m else 25 * 1024


def routing(args):
    """Which deployment actually serves each production alias, by payload size.

    The scored benchmark cannot answer this. It measures `dbg-*` aliases, which
    map 1:1 to a deployment precisely so a score names one model -- so by
    construction it never exercises the router. Everything real goes through
    `fast`/`smart`/`default`, where a group holds several deployments and
    `enable_pre_call_checks` silently drops any whose declared max_input_tokens
    is smaller than the prompt. The request still succeeds. Nothing warns.

    That mattered the day bin/offload-hook shipped. Measured 2026-07-29: at 2 KB
    `smart` was cerebras/zai-glm-4.7 as advertised; at 25 KB -- the hook's
    threshold at the time -- the same alias was served by
    ollama_chat/vibeforged-14b, a 97% backend silently swapped for the 88% one
    with a 73-98% swing, while gpt-oss-120b sat idle and eligible. Two fixes came
    out of it: context_window_fallbacks now tries the big cloud group before
    local, and the hook dropped to 8 KB, which fits every tier. Neither would
    have been noticed without this probe, because nothing errored.

    Eight tiny requests (max_tokens=16), so it is cheap enough to run whenever
    the config, a provider's limits, or the hook threshold change.
    """
    big = hook_limit()
    cfg = yaml.safe_load(CONFIG.read_text())
    # id -> what it really is, straight from the running proxy.
    try:
        req = urllib.request.Request(f"{BASE}/model/info",
                                     headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.load(r)["data"]
    except Exception as e:
        sys.exit(f"cannot reach the proxy at {BASE}: {e}")
    by_id = {str((m.get("model_info") or {}).get("id")): m["litellm_params"]["model"]
             for m in info}

    groups = [m["model_name"] for m in cfg["model_list"]]
    # The tiers bin/llm actually offers, in its own order, deduped.
    aliases = [a for a in ("default", "fast", "smart", "local")
               if a in groups] or sorted(set(groups))
    pad = "\n".join(f"[{i:05d}] INFO worker: processed batch {i} in {i % 37}ms"
                    for i in range(big // 48 + 1))

    print(f"hook blocks at {big // 1024} KB, so that is the production size.\n"
          f"{'alias':<10}{'at 2 KB (benchmarked, 2 probes)':<46}"
          f"{'at %d KB (what the hook sends)' % (big // 1024)}")
    def serve(alias, text, tools=None):
        body = {"model": alias, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply OK.\n\n" + text}]}
        if tools:
            body["tools"] = tools
        rq = urllib.request.Request(
            f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {KEY}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=300) as r:
                mid = r.headers.get("x-litellm-model-id", "")
                r.read()
            return by_id.get(str(mid), mid[:20] or "?")
        except Exception as e:
            return f"ERROR {classify(str(e))}"

    bad = []
    for alias in aliases:
        # Twice at the small size, deliberately. A group holds several
        # deployments and the router picks by weight, so two calls can land
        # differently for entirely benign reasons -- `fast` alone can answer as
        # groq or as cerebras. Comparing one small call to one large call reports
        # that ordinary variance as a downgrade. What is NOT benign is the large
        # payload reaching a deployment that never serves the small one: that is
        # a capability drop, not a coin flip.
        small = {serve(alias, pad[:2000]) for _ in range(2)}
        large = serve(alias, pad[:big])
        # A group can legitimately answer from several deployments; show them all
        # so "it varied" is visible rather than looking like a single result.
        shown = ", ".join(sorted(small))
        shown = shown if len(shown) <= 44 else shown[:41] + "..."
        print(f"{alias:<10}{shown:<46}{large}")
        if large not in small:
            bad.append(f"  {alias}: serves {shown} at 2 KB but {large} at "
                       f"{big // 1024} KB -- the tier you asked for is not the "
                       f"one serving your real payloads.")
        if alias != "local" and "ollama" in large:
            bad.append(f"  {alias}: a cloud tier is falling back to the local "
                       f"14B at production size. Check context_window_fallbacks; "
                       f"`local` belongs last in that list, not first.")
    # ---------------------------------------------------------------------
    # Second probe: a Claude Code tool array, which the size probe above cannot
    # see. `enable_pre_call_checks` drops deployments whose declared
    # max_input_tokens is under the prompt -- but measured 2026-07-29 it does not
    # count the `tools` array at all. dbg-cerebras-zai declares 4000 and served
    # an 80-tool request of roughly 7.4k tokens; at 108 tools the PROVIDER
    # returned a 400 that no fallback caught, because as far as the router was
    # concerned the prompt was tiny. `ccfree` sends ~9.8k tokens of schemas in
    # every request, so this is the exact shape that breaks it.
    #
    # The tool array is imported from tools.py, not copied -- same rule as
    # backends() and hook_limit(). A second copy would drift and then measure a
    # load nobody sends.
    # ---------------------------------------------------------------------
    try:
        from tools import CORE_TOOLS, FILLER_TOOLS
        cc_tools = CORE_TOOLS + FILLER_TOOLS
    except ImportError:
        cc_tools = None
    if cc_tools:
        est = len(json.dumps(cc_tools)) // 4
        declared = {m["litellm_params"]["model"]:
                    (m.get("model_info") or {}).get("max_input_tokens")
                    for m in cfg["model_list"]}
        print(f"\n{len(cc_tools)} Claude Code tool schemas ~{est:,} tokens, with a "
              f"tiny message.\n{'alias':<10}{'served by':<46}declared ctx")
        for alias in aliases:
            got = serve(alias, "hi", cc_tools)
            lim = declared.get(got)
            print(f"{alias:<10}{got:<46}{lim if lim is not None else '?'}")
            if got.startswith("ERROR"):
                bad.append(f"  {alias}: cannot carry a Claude Code tool array at "
                           f"all ({got}). `ccfree` will not work on this tier.")
            elif lim is not None and lim < est:
                bad.append(
                    f"  {alias}: served by {got}, which declares {lim:,} input "
                    f"tokens for a ~{est:,}-token tool array. The pre-call check "
                    f"DOES count `tools` and still had nowhere better to send "
                    f"this, so the group has no big-context member. Add one; the "
                    f"provider 400s otherwise.")

    if bad:
        print("\nSILENT DOWNGRADE:")
        print("\n".join(dict.fromkeys(bad)))
    else:
        print("\nno tier drops to a deployment it cannot already reach at 2 KB, "
              "and every tier can carry a Claude Code tool array.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", help="comma-separated dbg-* aliases; default all")
    p.add_argument("--evals", help="comma-separated eval ids, within --tiers; "
                                   "default all of them. For re-asking one "
                                   "question without paying for its whole tier")
    # NOT a way to buy confidence in a pass/fail behaviour. Reps inside one run
    # are separate requests but perfectly correlated outcomes: over 20 stored
    # runs, sum_inject scored 1.0 or 0.4 and never once in between, which
    # independent draws would do most of the time. `--reps 10` on an injection
    # eval reads as n=10 and is n=1 -- that is how "10/10 resisted" got written
    # down for a model that a separate run showed being hijacked 3/3. Use reps
    # for latency spread; to test a behaviour, repeat the RUN.
    p.add_argument("--reps", type=int, default=1,
                   help="requests per eval. Latency spread only -- outcomes "
                        "within a run are correlated, so this does NOT make a "
                        "pass/fail result more trustworthy")
    # 4096 because that is what bin/llm sends. Benchmarking at a budget the
    # production caller does not use measures nothing: at 1024, qwen3.6-27b
    # scored 50% purely by being cut off mid-thought, which would have been read
    # as "bad model" when the real answer is "fine model, expensive on a TPM tier".
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="output budget; must match bin/llm (4096) to be meaningful")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: omit the field, exactly as bin/llm does. "
                        "Pass 0 for a reproducible comparison.")
    p.add_argument("--system", default=None,
                   help="system prompt; default is whatever bin/llm currently sends. "
                        "Pass '' to send none (that is NOT what production does).")
    p.add_argument("--baseline", action="store_true",
                   help="score degenerate answers offline to establish the floor")
    p.add_argument("--report", action="store_true", help="compare past runs, no requests")
    p.add_argument("--regrade", nargs="?", const="latest", default=None,
                   choices=["latest", "all"],
                   help="re-score answers already in the store with the current "
                        "graders, spending no quota. 'latest' (default) does the "
                        "newest sweep, 'all' every backend run ever logged. Use "
                        "after changing a grader: the models already answered, and "
                        "editing how you count does not change what they said.")
    p.add_argument("--usage", action="store_true",
                   help="what offloading actually did in production, from the "
                        "usage.jsonl bin/llm writes. Spends nothing and needs "
                        "no proxy; unrelated to --tiers.")
    p.add_argument("--routing", action="store_true",
                   help="which deployment really serves each production tier, at "
                        "benchmark size vs the size the offload hook sends. Catches "
                        "silent downgrades the scored suite cannot see, because that "
                        "one measures 1:1 dbg-* aliases and never exercises the "
                        "router. Also sends one real Claude Code tool array per "
                        "tier. The pre-call check does count `tools`, but it can "
                        "only steer to a big-context member the group actually "
                        "has -- this catches the groups that have none. 12 requests, "
                        "all tiny except the tool arrays; unrelated to --tiers.")
    p.add_argument("--list", action="store_true")
    p.add_argument("--experiment", default="multi-llm-offload")
    p.add_argument("--tiers", default="core,hard",
                   help="which eval tiers to run (default core,hard). 'all' adds "
                        "smoke (four evals no backend has ever failed -- worth one run "
                        "when onboarding a backend) and large (four real tool-output "
                        "payloads, 8k-30k chars, which is where context limits show up "
                        "and where a Groq pass slows to roughly one eval per minute).")
    a = p.parse_args()

    # Before the tier filter, unlike --report/--regrade: these two read the
    # production log and the live router, not EVALS. There is no ruler to pick.
    if a.usage:
        return usage(a)
    if a.routing:
        return routing(a)

    # Filter IN PLACE, before anything reads the set. EVALS is a shared list
    # object, so this reaches every module that imported it -- including
    # fingerprint(), which is the point: a run over a subset is a run against a
    # DIFFERENT RULER and must not share a digest with a full run, or the two
    # sit in the same table looking comparable. Threading a parameter through
    # six call sites would buy nothing except the chance to miss one.
    if a.tiers != "all":
        keep = {t.strip() for t in a.tiers.split(",")}
        unknown = keep - {e["tier"] for e in EVALS}
        if unknown:
            sys.exit(f"unknown tier(s): {sorted(unknown)}")
        EVALS[:] = [e for e in EVALS if e["tier"] in keep]
        if not EVALS:
            sys.exit(f"--tiers {a.tiers} selected no evals")

    # Same in-place filter, one level finer. Without it the smallest runnable
    # unit is a tier, so re-asking ONE question costs its whole tier -- and in
    # the large tier that is 4 requests per backend against a daily free cap.
    # The digest follows automatically, which is the correct answer and not a
    # side effect: a one-eval run IS a different ruler and must not sit in the
    # same table as a full sweep pretending to be comparable.
    if a.evals:
        want = {s.strip() for s in a.evals.split(",")}
        unknown = want - {e["id"] for e in EVALS}
        if unknown:
            sys.exit(f"unknown eval id(s): {sorted(unknown)} "
                     f"(in tier(s) {a.tiers}: {sorted(e['id'] for e in EVALS)})")
        EVALS[:] = [e for e in EVALS if e["id"] in want]

    # Both after the tier filter, so --tiers picks what gets regraded and which
    # ruler --report calls "current" -- a report whose digest matches no sweep
    # anyone runs marks every row stale and says nothing. Both before
    # backends(), which needs the proxy up; neither touches the network.
    if a.report:
        return report(a)
    if a.regrade:
        return regrade(a)

    known = backends()
    chosen = a.models.split(",") if a.models else list(known)
    unknown = [m for m in chosen if m not in known]
    if unknown:
        sys.exit(f"unknown alias(es): {unknown}\nknown: {list(known)}")

    if a.list:
        for m in chosen:
            print(f"  {m:<18} {known[m]}")
        print(f"\n{len(chosen)} backends x {len(EVALS)} evals x {a.reps} reps "
              f"= {len(chosen) * len(EVALS) * a.reps} requests")
        return

    # sqlite, not the file store: MLflow 3 refuses `file:./mlruns` outright
    # ("in maintenance mode"). Artifacts still land on disk beside the db.
    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment(a.experiment)
    # Renders as the experiment description in the UI. Someone opening :5000
    # cold sees a wall of percentages with no statement of what they mean; the
    # two traps below are the ones that have actually misled a reader here.
    mlflow.set_experiment_tag("mlflow.note.content", EXPERIMENT_NOTE)

    # The floor is computed first and for free, so every backend in this sweep
    # can be scored against it in the same run rather than against a number
    # remembered from some other day.
    system = production_system() if a.system is None else a.system
    a.system_used = system
    floors = {name: aggregate(run_baseline(fn, 1, 0, None, None)[0])
              for name, fn in BASELINES.items()}
    floor = max(floors.values())

    jobs = ([(n, f"baseline/{n}", run_baseline, BASELINES[n]) for n in BASELINES]
            if a.baseline else
            [(m, known[m], run_model, None) for m in chosen])

    label = "baselines" if a.baseline else "backends"
    print(f"{len(jobs)} {label}, {len(EVALS)} evals, {a.reps} rep(s), "
          f"max_tokens={a.max_tokens}, "
          f"temperature={'omitted (production)' if a.temperature is None else a.temperature}")
    print(f"eval digest {fingerprint()}   floor {floor:.0%} "
          f"({max(floors, key=floors.get)})")
    # Printed, not silent: the system prompt measurably changes injection
    # resistance, so a run whose header does not match bin/llm is not a
    # production measurement.
    print(f"system: {system[:70]!r}{' ...' if len(system) > 70 else ''}\n")

    prompt_v = register_system_prompt(system)

    summary = []
    with mlflow.start_run(run_name=f"sweep-{'baseline' if a.baseline else 'backends'}") as run:
        mlflow.set_tags({"kind": "sweep", "eval_digest": fingerprint()})
        if prompt_v is not None:
            # Links the run to the exact prompt text, so "which system prompt was
            # this scored under" is answerable from the UI instead of from a
            # truncated param and a guess.
            mlflow.MlflowClient().link_prompt_version_to_run(run.info.run_id, prompt_v)
            mlflow.set_tags({"system_prompt_version": str(prompt_v.version)})
        mlflow.log_params({"n_evals": len(EVALS), "reps": a.reps,
                           "max_tokens": a.max_tokens, "eval_digest": fingerprint(),
                           "temperature": "omitted" if a.temperature is None else a.temperature,
                           "system": system[:480],
                           "n_targets": len(jobs)})
        mlflow.log_metrics({"floor_score": floor,
                            **{f"floor_{k.replace('floor-', '')}": v for k, v in floors.items()}})
        # Dataset lineage: the run records the digest of the exact set it scored
        # against, so MLflow itself can tell you two runs used different rulers.
        mlflow.log_input(mlflow.data.from_pandas(eval_set_frame(), name="offload-evals"),
                         context="eval")
        mlflow.log_table(eval_set_frame(), "eval_set.json")

        for alias, real, runner, extra in jobs:
            print(f"  {alias:<18} ", end="", flush=True)
            t0 = time.monotonic()
            rows, unavailable = (runner(extra, a.reps, a.max_tokens, a.temperature, system)
                                 if extra is not None else
                                 runner(alias, a.reps, a.max_tokens, a.temperature, system))
            summary.append(log_child(alias, real, rows, unavailable, a, t0,
                                     None if a.baseline else floor))

        done = [s for s in summary if s[1] is not None]
        if done:
            mlflow.log_table(pd.DataFrame([
                dict(alias=al, overall=round(ov, 3),
                     core=round(d["tiers"].get("core") or 0, 3),
                     hard=round(d["tiers"].get("hard") or 0, 3),
                     ci_low=round(d["ci"][0], 3) if d["ci"][0] is not None else None,
                     ci_high=round(d["ci"][1], 3) if d["ci"][1] is not None else None)
                for al, ov, d in done]), "comparison.json")
            mlflow.log_metric("best_score", max(s[1] for s in done))

    print("\n" + "-" * 72)
    for alias, overall, d in sorted(done, key=lambda s: -s[1]):
        vf = ""
        if not a.baseline and floor < 1:
            vf = f"  vs-floor {max(0.0, (overall - floor) / (1 - floor)):.0%}"
        print(f"  {alias:<18} {overall:.0%}{vf}   " +
              "  ".join(f"{t} {v:.0%}" for t, v in d["tiers"].items() if v is not None))
    for alias, overall, why in [s for s in summary if s[1] is None]:
        print(f"  {alias:<18}  --   {why}")
    if not a.baseline:
        # Two backends whose intervals overlap are not ranked, they are tied.
        print(f"\n  floor {floor:.0%}: a degenerate answer already scores this. "
              f"Read vs-floor, not the raw score.")
    # Both UI units are disabled by default, so naming the start command matters
    # more than naming the URL -- an unstarted service is the likeliest reason
    # the link looks dead.
    print(f"\nresults: {HERE / 'mlflow.db'}"
          f"\n         srv start mlflow-server  ->  http://127.0.0.1:5000"
          f"\n         srv start mlflow-ui      ->  http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
