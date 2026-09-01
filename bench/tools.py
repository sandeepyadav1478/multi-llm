#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""How many tools can each backend actually handle, and could it drive Claude Code?

Separate from bench.py on purpose. That suite grades PROSE -- did the summary keep
the fact that mattered -- and every question here is pass/fail on a structured
field instead: did the right function name come back, did its arguments parse.
Bolting a JSON-schema grader onto a text harness would have meant threading a
second scoring path through nine call sites to answer a question that fits in one
file.

The question this exists for. `ccfree` and `cclocal` point Claude Code itself at
these backends, and Claude Code does not send one tool -- this session carries 11
resident tools and ~90 deferred ones behind ToolSearch, so the real load is close
to 100 schemas in every request. A backend that answers beautifully with one tool
and silently stops calling tools at forty is useless for that and looks fine on
every other measurement in this directory. Nothing else here would catch it.

What is measured, per (backend, tool-count) cell:

  schema_ok    the provider accepted the tool array at all -- some reject size
               or specific JSON Schema features, and that is a hard ceiling
  called       it emitted a tool call instead of prose describing one
  right_tool   it picked the tool the prompt unambiguously required
  valid_args   arguments parse as JSON and carry every required property
  no_halluc    the name it returned actually exists in the array it was sent
  types_ok     arguments validate against that tool's OWN schema -- types,
               enums and nested required fields, not just key presence. This is
               the one that separates "named the right tool" from "produced a
               call the client would actually accept"

Usage:
  uv run tools.py                      # sweep, every dbg-* alias in the config
  uv run tools.py --models dbg-groq
  uv run tools.py --counts 1,10,50
  uv run tools.py --list               # show the probes and exit, no requests
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent.parent / "litellm" / "config.yaml"
BASE = "http://localhost:4000"
KEY = "sk-litellm-local"

# ---------------------------------------------------------------------------
# The tool set. REAL Claude Code tools, transcribed from a live session's own
# tool list rather than invented, because the point is to measure the actual
# load. Descriptions are trimmed to one line -- Claude Code's real ones run to
# paragraphs, which would make this a context-length test instead of a
# tool-count test. That is a different question and bench.py's `large` tier
# already asks it.
# ---------------------------------------------------------------------------
def _fn(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


_STR = lambda d: {"type": "string", "description": d}
_INT = lambda d: {"type": "integer", "description": d}
_BOOL = lambda d: {"type": "boolean", "description": d}
_ENUM = lambda d, vals: {"type": "string", "description": d, "enum": vals}


def _ARR(d, props, required):
    """An array of objects -- the shape a flat {string, integer} pool never tests.

    TaskCreate/TodoWrite really takes this, and it is the schema feature most
    likely to defeat a small model: it has to nest, and it has to hit an enum
    inside the nesting. A pool made entirely of flat scalars measures none of
    that and reports 100%.
    """
    return {"type": "array", "description": d,
            "items": {"type": "object", "properties": props, "required": required}}

# The eleven that are always resident, plus the file/search tools every probe
# needs to be able to confuse with each other. Order here is NOT the order sent.
CORE_TOOLS = [
    _fn("Read", "Reads a file from the local filesystem.",
        {"file_path": _STR("The absolute path to the file to read"),
         "offset": _INT("The line number to start reading from"),
         "limit": _INT("The number of lines to read")}, ["file_path"]),
    _fn("Write", "Writes a file to the local filesystem, overwriting if one exists.",
        {"file_path": _STR("The absolute path to the file to write"),
         "content": _STR("The content to write to the file")}, ["file_path", "content"]),
    _fn("Edit", "Performs exact string replacement in a file.",
        {"file_path": _STR("The absolute path to the file to modify"),
         "old_string": _STR("The text to replace"),
         "new_string": _STR("The text to replace it with"),
         "replace_all": _BOOL("Replace every occurrence")},
        ["file_path", "old_string", "new_string"]),
    _fn("Bash", "Executes a bash command and returns its output.",
        {"command": _STR("The command to execute"),
         "description": _STR("Clear, concise description of what this command does"),
         "timeout": _INT("Optional timeout in milliseconds")}, ["command"]),
    _fn("Glob", "Fast file pattern matching that works with any codebase size.",
        {"pattern": _STR("The glob pattern to match files against"),
         "path": _STR("The directory to search in")}, ["pattern"]),
    _fn("Grep", "A powerful search tool built on ripgrep. Searches file contents.",
        {"pattern": _STR("The regular expression pattern to search for"),
         "path": _STR("File or directory to search in"),
         "glob": _STR("Glob pattern to filter files")}, ["pattern"]),
    _fn("WebFetch", "Fetches content from a specified URL and processes it.",
        {"url": _STR("The URL to fetch content from"),
         "prompt": _STR("The prompt to run on the fetched content")}, ["url", "prompt"]),
    _fn("WebSearch", "Search the web and use the results to inform responses.",
        {"query": _STR("The search query to use")}, ["query"]),
    # Nested array-of-objects with an enum inside it. This is the real shape, and
    # it is deliberately the hardest schema in the pool -- see _ARR.
    _fn("TaskCreate", "Create tasks to track progress on multi-step work.",
        {"tasks": _ARR("The tasks to create",
                       {"description": _STR("Short description of the task"),
                        "status": _ENUM("Current state",
                                        ["pending", "in_progress", "completed"])},
                       ["description", "status"])}, ["tasks"]),
    _fn("NotebookEdit", "Replaces the contents of a specific cell in a Jupyter notebook.",
        {"notebook_path": _STR("The absolute path to the notebook"),
         "new_source": _STR("The new source for the cell"),
         "cell_type": _ENUM("The type of the cell", ["code", "markdown"]),
         "cell_id": _STR("The id of the cell to edit")}, ["notebook_path", "new_source"]),
    _fn("Skill", "Invoke a packaged skill by name.",
        {"skill": _STR("The name of a skill from the available-skills list"),
         "args": _STR("Optional arguments for the skill")}, ["skill"]),
]

# Filler. These are the REAL deferred names from a tokensave-enabled session --
# the actual thing that takes a Claude Code request from 11 tools to ~100 -- with
# schemas approximated to a common shape. Real names matter more than real
# schemas here: name collision is what makes a big array hard to choose from,
# and `tokensave_read` sitting next to `Read` is exactly the confusion under test.
FILLER_NAMES = [
    "tokensave_context", "tokensave_search", "tokensave_status", "tokensave_body",
    "tokensave_node", "tokensave_callers", "tokensave_callees", "tokensave_call_chain",
    "tokensave_impact", "tokensave_affected", "tokensave_test_map", "tokensave_read",
    "tokensave_test_coverage", "tokensave_run_affected_tests", "tokensave_implementations",
    "tokensave_type_hierarchy", "tokensave_module_api", "tokensave_dead_code",
    "tokensave_unused_imports", "tokensave_simplify_scan", "tokensave_redundancy",
    "tokensave_complexity", "tokensave_hotspots", "tokensave_god_class",
    "tokensave_circular", "tokensave_blame", "tokensave_log", "tokensave_diff_context",
    "tokensave_pr_context", "tokensave_replace_symbol", "tokensave_str_replace",
    "tokensave_insert_at_symbol", "tokensave_annotations", "tokensave_branch_diff",
    "tokensave_branch_list", "tokensave_branch_search", "tokensave_by_qualified_name",
    "tokensave_callers_for", "tokensave_changelog", "tokensave_commit_context",
    "tokensave_config", "tokensave_constructors", "tokensave_coupling",
    "tokensave_dependencies", "tokensave_dependency_depth", "tokensave_derives",
    "tokensave_diagnose", "tokensave_diagnostics", "tokensave_diff",
    "tokensave_distribution", "tokensave_doc", "tokensave_doc_coverage",
    "tokensave_dsm", "tokensave_entities", "tokensave_field_sites",
    "tokensave_file_dependents", "tokensave_files", "tokensave_find_exact_symbol",
    "tokensave_gini", "tokensave_health", "tokensave_impls",
    "tokensave_inheritance_depth", "tokensave_insert_at", "tokensave_largest",
    "tokensave_multi_str_replace", "tokensave_port_order", "tokensave_port_status",
    "tokensave_rank", "tokensave_recursion", "tokensave_record_code_area",
    "tokensave_record_decision", "tokensave_rename_preview", "tokensave_runtime",
    "tokensave_session_end", "tokensave_session_recall", "tokensave_session_start",
    "tokensave_signature", "tokensave_signature_search", "tokensave_similar",
    "tokensave_test_risk", "tokensave_todos", "tokensave_unsafe_patterns",
    "CronCreate", "CronDelete", "CronList", "Monitor", "PushNotification",
    "SendMessage", "TaskGet", "TaskList", "TaskOutput", "TaskStop", "TaskUpdate",
    "ExitPlanMode", "EnterPlanMode", "ListMcpResourcesTool", "ReadMcpResourceTool",
]

FILLER_TOOLS = [
    _fn(n, f"Code-graph operation: {n.replace('_', ' ')}.",
        {"query": _STR("Search query or symbol name"),
         "limit": _INT("Maximum number of results to return")}, ["query"])
    for n in FILLER_NAMES
]

# ---------------------------------------------------------------------------
# Probes. Each names exactly one correct tool, and says so without using the
# tool's own name -- "read the file" would just be string matching. The point is
# whether the model can still SELECT under load, so ambiguity is the enemy: a
# probe two tools could satisfy measures nothing when it picks the other one.
# ---------------------------------------------------------------------------
PROBES = [
    # "Use a tool" was not enough here: `cat /etc/hostname` is a correct answer,
    # and cerebras gave it. That scored as a wrong-tool failure when the probe
    # was the thing at fault -- so the shell is excluded explicitly, without
    # naming the tool that is wanted.
    dict(id="read", tool="Read", args=["file_path"],
         prompt="What are the contents of the file /etc/hostname? Use the "
                "dedicated file-reading tool, not a shell command."),
    dict(id="glob", tool="Glob", args=["pattern"],
         prompt="List every file in this project whose name ends in .test.js. "
                "Match by filename pattern only -- do not search inside files."),
    dict(id="bash", tool="Bash", args=["command"],
         prompt="Execute the shell command `npm run build` and show me its output."),
    dict(id="grep", tool="Grep", args=["pattern"],
         prompt="Search the text INSIDE the project's files for the string "
                "TODO_URGENT and tell me which files contain it."),
    dict(id="write", tool="Write", args=["file_path", "content"],
         prompt="Create a brand new file at /tmp/notes.md containing exactly the "
                "text 'hello'. The file does not exist yet."),

    # ---- added 2026-07-29 -------------------------------------------------
    # The five above cover five tools. Claude Code resident set is ELEVEN, and
    # "108 tools in the array" was never the same claim as "can invoke all of
    # them" -- the other 103 were decoys nobody ever asked it to pick. These
    # six close that gap, and they are deliberately the awkward ones: two take
    # a constrained vocabulary, one takes a nested array, one is trivially
    # confusable with Write.
    # The file content is inlined for a reason, and it is the same reason the
    # `read` probe has to exclude the shell. Without it, `local` called Read
    # first in 4 runs out of 4 and scored a clean zero -- while being right:
    # Claude Code's own Edit tool refuses to run until the file has been read in
    # this conversation, so Read-then-Edit is the correct agentic pattern and
    # this was measuring obedience to a rule the probe never mentioned. Handed
    # the content, the same model emits a perfect Edit 3 times out of 3.
    # A single-shot probe may only ask for something answerable in a single shot.
    dict(id="edit", tool="Edit", args=["file_path", "old_string", "new_string"],
         prompt="The file /srv/app/config.py contains exactly:\n"
                "PORT = 8080\nDEBUG=True\nHOST = '0.0.0.0'\n\n"
                "You have already read it, so do not read it again. Change "
                "DEBUG=True to DEBUG=False by replacing that exact string in "
                "place. Do not rewrite the whole file, do not run a shell "
                "command."),
    dict(id="webfetch", tool="WebFetch", args=["url", "prompt"],
         prompt="Fetch the page at https://example.com/pricing and tell me what "
                "the monthly price is. Retrieve that specific URL directly."),
    dict(id="websearch", tool="WebSearch", args=["query"],
         prompt="Find out what the current stable version of PostgreSQL is by "
                "searching the web. Do not fetch one specific URL -- search."),
    dict(id="task", tool="TaskCreate", args=["tasks"],
         prompt="Track this multi-step work as a task list: first migrate the "
                "database, then update the client. Mark the first one as "
                "already in progress and the second as still pending."),
    dict(id="notebook", tool="NotebookEdit", args=["notebook_path", "new_source"],
         prompt="In the Jupyter notebook /work/analysis.ipynb, replace the "
                "contents of cell c4 with the code `df.head()`."),
    dict(id="skill", tool="Skill", args=["skill"],
         prompt="Run the packaged skill named 'graphify' to turn my notes into "
                "a knowledge graph. Invoke it by name."),

    # The 97 filler tools are the REAL reason a request carries 108 schemas, and
    # until now not one probe ever asked for any of them -- they were scenery.
    # That left the discrimination that actually matters here untested: this
    # user's sessions always load tokensave, so the model must pick
    # `tokensave_callers` out of 97 near-identical `tokensave_*` names while
    # `Grep` and `Read` sit in the same array being plausible. If it cannot, MCP
    # is unusable under cclocal and every measurement above would still say 100%.
    # Target chosen for having no prefix twin. The first version aimed at
    # `tokensave_callers` and scored a miss when the model answered
    # `tokensave_callers_for` -- but the filler descriptions are generated stubs,
    # so those two read as "Code-graph operation: tokensave callers." and
    # "...tokensave callers for." Nothing could tell them apart, and grading a
    # model on an unanswerable question measures the question. Ten of the 97
    # names are prefix-twinned like that; _selftest keeps the target out of them.
    dict(id="mcp", tool="tokensave_dead_code", args=["query"],
         prompt="Using the code-graph tool, find the functions in this project "
                "that are never reached from anywhere -- dead, unreachable code. "
                "Query the graph; do not grep the files and do not read one."),
]

# Names where another tool is a strict prefix of this one, or vice versa. With
# generated one-line descriptions these are genuinely indistinguishable, so a
# probe may not target one. Asserted, not just documented.
PREFIX_TWINS = {n for n in FILLER_NAMES for m in FILLER_NAMES
                if m != n and (m.startswith(n) or n.startswith(m))}

# Claude Code batches independent work: "If you intend to call multiple tools and
# there are no dependencies between the calls, make all of the independent calls
# in the same block." A model that can only ever emit one call per turn still
# works -- it just costs a round trip per tool, which on a 3s local backend is
# the difference between one turn and four. Graded separately because it is a
# throughput property, not a correctness one.
PARALLEL_PROBE = dict(
    id="parallel", tools=["Read", "Read"], args=["file_path"],
    prompt="Read the contents of BOTH /etc/hostname and /etc/timezone. They are "
           "independent, so request both at once in a single response rather "
           "than one after the other.")


def build_tools(n, target, seed):
    """n tools including `target`, placed away from both ends.

    Position is the trap this guards against. A target always sent first scores a
    model that simply picks tools[0] as perfect, and always-last measures recency
    instead. It goes near the middle, shifted per probe by `seed` so no single
    index is ever the answer twice.
    """
    others = [t for t in CORE_TOOLS + FILLER_TOOLS
              if t["function"]["name"] != target]
    if n < 1:
        raise ValueError("need at least the target tool")
    picked = others[:n - 1]
    tgt = next(t for t in CORE_TOOLS + FILLER_TOOLS
               if t["function"]["name"] == target)
    at = (len(picked) // 2 + seed) % (len(picked) + 1)
    return picked[:at] + [tgt] + picked[at:], at


def ask(model, prompt, tools, system=None, timeout=180, msgs=None):
    """-> (message dict, latency, error string or None).

    `msgs` overrides prompt/system entirely, for the multi-turn loop -- which
    needs to send back an assistant turn carrying tool_calls plus a `tool` role
    reply, a shape no single prompt can express.
    """
    if msgs is None:
        msgs = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model,
                       "messages": msgs,
                       "tools": tools,
                       "max_tokens": 1024}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}",
                                  "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"], time.time() - t0, None
    except urllib.error.HTTPError as e:
        detail = e.read()[:400].decode(errors="replace")
        # The interesting failure. A provider that refuses the array outright is
        # a hard ceiling, and it reads completely differently from a model that
        # accepted the tools and then chose badly -- so never collapse the two.
        short = re.sub(r"\s+", " ", detail)[:150]
        return None, time.time() - t0, f"HTTP {e.code}: {short}"
    except Exception as e:
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


# A quota miss is NOT a capability result, and the first version of this file
# scored it as one: groq came back all-zeros across four tool counts and the
# table read as "cannot handle 5 tools" when every one of those cells was a 429.
# bench.py has a whole section on this exact trap. Anything matching here is
# reported as no-data and excluded from the percentages, never as a failure.
UNAVAILABLE = re.compile(
    r"429|rate.?limit|no deployments available|quota|cooldown|too many requests",
    re.I)


def classify(err):
    """-> 'ok' | 'unavailable' | 'refused'."""
    if not err:
        return "ok"
    return "unavailable" if UNAVAILABLE.search(err) else "refused"


# Some 429s are transient and some are arithmetic. Groq states its own numbers:
# "TPM: Limit 12000, Used 9547, Requested 11747" -- a single Claude Code request
# asking for 98% of a whole minute's allowance. Waiting cannot fix that, because
# the next attempt requests the same 11,747 against the same 12,000. Measured
# cost of not knowing the difference: ~8 minutes of sleep per backend to arrive
# at the same 429. Detecting it also turns a shrug ("rate-limited, no data")
# into the actual finding, which is that the budget is smaller than one turn.
#
# Groq sends TWO shapes and the `Used` field is the difference, so it has to be
# optional. The first cut of this required it and therefore missed the shape
# that matters MORE:
#   "Limit 12000, Used 9547, Requested 11747"  -- window partly spent
#   "Limit 8000, Requested 12993, please reduce your message size"
# The second has no `Used` because none is needed: the request alone is 162% of
# the entire per-minute cap, so it can never succeed at any time on an idle
# account. Requiring `Used` meant the permanent failure was retried and the
# transient one was not, which is exactly backwards.
TPM = re.compile(r"Limit (\d+),(?: Used (\d+),)? Requested (\d+)")

# Not "exceeds the cap" -- 11,747 of 12,000 does not exceed it, and a strict
# check would let the futile retry through. The request only has to be big
# enough that it needs a near-empty window, which our own previous attempt just
# drained. Above this share the retry needs the bucket to refill completely, so
# the sweep is waiting a full window to re-ask a question with a known answer.
FUTILE_SHARE = 0.8


def over_budget(err):
    """Request too big for the per-minute allowance itself -> retrying is futile.

    Returns a human-readable reason, or None if a retry could plausibly help.
    """
    m = TPM.search(err or "")
    if not m:
        return None
    limit, want = int(m.group(1)), int(m.group(3))
    if not limit or want < FUTILE_SHARE * limit:
        return None                     # small ask: someone else's traffic, so wait
    how = ("exceeds the whole cap, so it fails on a completely idle account"
           if want > limit else
           "needs the window near-empty, which the failed attempt itself prevents")
    return (f"one request needs {want:,} tokens against a {limit:,}/min cap "
            f"({want / limit:.0%} of the budget) -- {how}")


# The other half of the Claude Code load. A big tool array is only half of what
# a real request carries -- the system prompt runs to thousands of tokens of
# behavioural rules, and the failure mode worth catching is a model that selects
# tools fine on a bare prompt and stops once it has instructions competing for
# attention. Deliberately generic instructions: this measures whether tool
# selection survives a long system prompt, not whether the model obeys Anthropic's.
CC_SYSTEM = ("You are an interactive CLI tool that helps users with software "
             "engineering tasks. Use the instructions below and the tools "
             "available to you to assist the user.\n\n") + "\n".join(
    f"- {line}" for line in [
        "IMPORTANT: Assist with defensive security tasks only.",
        "Refuse to write code that may be used maliciously.",
        "IMPORTANT: You must NEVER generate or guess URLs unless confident.",
        "Only use emojis if the user explicitly requests it.",
        "You should be concise, direct, and to the point.",
        "Answer concisely with fewer than 4 lines unless asked for detail.",
        "Avoid introductions, conclusions, and explanations unless asked.",
        "Do not add code comments unless the user asks you to.",
        "Follow existing code conventions in the repository.",
        "NEVER assume a given library is available, even if well known.",
        "When making changes, first understand the file's conventions.",
        "Always follow security best practices.",
        "Never introduce code that exposes or logs secrets and keys.",
        "Never commit changes unless the user explicitly asks you to.",
        "Use the available search tools to understand the codebase.",
        "Prefer the dedicated file tools over shell commands when one fits.",
        "Do not use bash commands to read files when a read tool exists.",
        "Verify solutions with tests if possible, never assume a test framework.",
        "Check the README or search the codebase to determine the test command.",
        "Do not commit to the main or master branch.",
        "Report outcomes faithfully; if tests fail, say so with the output.",
        "When the user asks you to do something, do it rather than describing it.",
    ] * 6)  # ~10k chars, the order of a real Claude Code system prompt


def ask_retry(model, prompt, tools, system=None, tries=3):
    """ask(), but wait out a cooldown instead of scoring it as no-data.

    Groq's free tier rate-limits within a sweep, which left three of five tool
    counts unmeasurable -- a gap, not a result, and a gap is what makes a table
    unusable. The router says how long it wants ("Try again in 7 seconds"), so
    honour that rather than guessing; only a 'refused' or a real answer is worth
    returning early, since retrying either would just spend quota to learn the
    same thing twice.
    """
    for i in range(tries):
        msg, dt, err = ask(model, prompt, tools, system)
        if classify(err) != "unavailable":
            return msg, dt, err
        why = over_budget(err)
        if why:                         # arithmetic, not congestion -- stop now
            return msg, dt, f"{err.split(':')[0]}: over budget -- {why}"
        if i == tries - 1:
            break
        m = re.search(r"try again in (\d+)", err, re.I)
        time.sleep(min(int(m.group(1)) + 2 if m else 10 * (i + 1), 60))
    return msg, dt, err


_JTYPE = {"string": str, "integer": int, "boolean": bool,
          "array": list, "object": dict, "number": (int, float)}


def schema_errors(value, schema, path="args"):
    """Validate against the tool's OWN JSON Schema. -> list of human errors.

    `valid_args` only ever asked "is the required key present", which passes
    `{"tasks": "migrate the database"}` against a schema demanding an array of
    objects carrying an enum. Every backend scored 100% on argument quality
    partly because nothing in the pool had a shape that could be got wrong --
    the 97 filler tools are all {string, integer}. Both halves of that are now
    fixed, so this is deliberately strict: a wrong type is a call the client
    would reject, which is indistinguishable from no call at all.

    Not a general JSON Schema implementation -- type, enum, required and one
    level of nesting is the whole surface these tool definitions use.
    """
    errs = []
    t = schema.get("type")
    if t and t in _JTYPE and not isinstance(value, _JTYPE[t]):
        # bool is an int in Python and "true" is not a boolean anywhere.
        if not (t == "integer" and isinstance(value, bool)):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if t == "integer" and isinstance(value, bool):
        return [f"{path}: expected integer, got boolean"]
    if (vals := schema.get("enum")) and value not in vals:
        return [f"{path}: {value!r} not in {vals}"]
    if t == "object":
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in value:
                errs.append(f"{path}.{k}: required, missing")
        for k, v in value.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
    if t == "array" and (items := schema.get("items")):
        if not value:
            errs.append(f"{path}: empty array")
        for i, v in enumerate(value):
            errs += schema_errors(v, items, f"{path}[{i}]")
    return errs


def grade(msg, probe, tools):
    names = {t["function"]["name"] for t in tools}
    calls = (msg or {}).get("tool_calls") or []
    out = dict(schema_ok=True, called=bool(calls), right_tool=False,
               valid_args=False, no_halluc=False, types_ok=False)
    if not calls:
        return out
    f = calls[0]["function"]
    out["no_halluc"] = f["name"] in names
    out["right_tool"] = f["name"] == probe["tool"]
    try:
        args = json.loads(f.get("arguments") or "{}")
        # Required-and-present, not merely parseable. `{}` is valid JSON and a
        # completely useless tool call, and it would otherwise score a point.
        out["valid_args"] = (isinstance(args, dict)
                             and all(a in args and args[a] not in (None, "")
                                     for a in probe["args"]))
        # Only meaningful against the schema it was actually aiming at: a call to
        # the wrong tool validated against the right tool's schema measures
        # nothing, so a wrong tool is a types_ok failure by definition.
        if out["right_tool"]:
            tgt = next(t for t in tools if t["function"]["name"] == probe["tool"])
            errs = schema_errors(args, tgt["function"]["parameters"])
            out["types_ok"] = not errs
            if errs:
                out["type_err"] = errs[0]
    except (ValueError, TypeError):
        pass
    return out


def grade_parallel(msg, tools):
    """Did it batch two independent calls into one response?"""
    calls = (msg or {}).get("tool_calls") or []
    paths = []
    for c in calls:
        try:
            paths.append(json.loads(c["function"].get("arguments") or "{}")
                         .get("file_path"))
        except (ValueError, TypeError):
            pass
    return dict(n_calls=len(calls),
                batched=len(calls) >= 2,
                both_files=len({p for p in paths if p}) >= 2,
                names_ok=all(c["function"]["name"] == "Read" for c in calls) and bool(calls))


# A value the model cannot possibly know, so quoting it back proves the tool
# result was actually read rather than a plausible hostname being invented.
LOOP_MARKER = "zurith-a4f1c9"


def loop_probe(model, n, system=None):
    """The turn everything above skips: does a tool RESULT come back in?

    Every other measurement in this file is single-shot -- prompt in, tool call
    out, done. Claude Code is a loop: it sends the call's output back as a
    `tool` role message and expects the model to carry on. A backend that picks
    tools perfectly and cannot consume its own tool result is useless for it,
    and scores 5/5 on every other column here. That gap is the reason this
    exists.

    Three things can break, and they are graded apart because the fixes differ:
      accepted    the provider took an assistant-with-tool_calls turn plus a
                  `tool` role reply at all -- some 400 on the message shape
      answered    it produced prose instead of re-emitting the same call, i.e.
                  it understood the task was now finished
      used_result it quoted the marker, so it actually read the result rather
                  than answering from the prompt
    """
    probe = PROBES[0]                       # read /etc/hostname
    tools, _ = build_tools(n, probe["tool"], 0)
    m1, dt1, e1 = ask(model, probe["prompt"], tools, system)
    calls = (m1 or {}).get("tool_calls") or []
    if e1 or not calls:
        return dict(turn1=False, accepted=False, answered=False,
                    used_result=False, note=e1 or "no tool call on turn 1"), dt1

    call = calls[0]
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": probe["prompt"]},
        {"role": "assistant", "content": m1.get("content") or None,
         "tool_calls": [{"id": call.get("id") or "call_0", "type": "function",
                         "function": call["function"]}]},
        {"role": "tool", "tool_call_id": call.get("id") or "call_0",
         "content": LOOP_MARKER},
    ]
    m2, dt2, e2 = ask(model, None, tools, msgs=msgs)
    if e2:
        return dict(turn1=True, accepted=False, answered=False,
                    used_result=False, note=e2[:120]), dt1 + dt2
    text = (m2 or {}).get("content") or ""
    return dict(turn1=True, accepted=True,
                answered=bool(text.strip()) and not (m2 or {}).get("tool_calls"),
                used_result=LOOP_MARKER in text,
                note=re.sub(r"\s+", " ", text)[:80]), dt1 + dt2


def debug_aliases():
    """Every dbg-* alias in the config -- NOT a liveness check.

    Named `reachable()` at first, which cost an hour: it made a 10-alias sweep
    get written down as "8 backends" because a separate liveness probe had
    found 8 up, and nothing in the name said the two were different questions.
    A dead alias still gets swept, and its cells come back as no-data.
    """
    cfg = yaml.safe_load(CONFIG.read_text())
    return [m["model_name"] for m in cfg["model_list"]
            if m["model_name"].startswith("dbg-")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", help="comma-separated dbg-* aliases; default all")
    p.add_argument("--counts", default="1,5,15,40,80",
                   help="tool-array sizes to sweep (default 1,5,15,40,80)")
    p.add_argument("--probes", help="comma-separated probe ids; default all")
    p.add_argument("--cc-system", action="store_true",
                   help="also send a ~10k-char Claude-Code-shaped system prompt, "
                        "so tool selection is measured under the real load")
    p.add_argument("--list", action="store_true", help="show probes and exit")
    p.add_argument("--loop", action="store_true",
                   help="run the two-turn loop instead of the sweep: tool call, "
                        "tool result back, continuation. This is what Claude "
                        "Code actually does and what the sweep never tests")
    p.add_argument("--parallel", action="store_true",
                   help="ask for two independent calls in one response. Claude "
                        "Code batches; a model that cannot costs a round trip "
                        "per tool. Throughput, not correctness")
    p.add_argument("--json", help="also write raw results here")
    a = p.parse_args()

    probes = PROBES
    if a.probes:
        want = {s.strip() for s in a.probes.split(",")}
        unknown = want - {x["id"] for x in probes}
        if unknown:
            sys.exit(f"unknown probe(s): {sorted(unknown)}")
        probes = [x for x in probes if x["id"] in want]

    total_pool = len(CORE_TOOLS) + len(FILLER_TOOLS)
    if a.list:
        print(f"{total_pool} tools available ({len(CORE_TOOLS)} real Claude Code "
              f"+ {len(FILLER_TOOLS)} real deferred/MCP names)\n")
        for x in probes:
            print(f"  {x['id']:<8} -> {x['tool']:<10} needs {x['args']}")
            print(f"           {x['prompt'][:96]}")
        return 0

    counts = [int(c) for c in a.counts.split(",")]
    if max(counts) > total_pool:
        sys.exit(f"--counts max {max(counts)} exceeds the {total_pool} tools defined")
    models = [m.strip() for m in a.models.split(",")] if a.models else debug_aliases()

    if a.parallel:
        sysp = CC_SYSTEM if a.cc_system else None
        print(f"parallel tool calls, system prompt: {'yes' if sysp else 'no'}\n"
              f"{'backend':<18}{'tools':>6}{'calls':>7}{'batched':>9}"
              f"{'2 files':>9}{'names':>7}{'secs':>7}")
        rows = []
        for model in models:
            for n in counts:
                tools, _ = build_tools(n, "Read", 0)
                msg, dt, err = ask_retry(model, PARALLEL_PROBE["prompt"], tools, sysp)
                if err:
                    print(f"{model:<18}{n:>6}  [{classify(err)}] {err[:60]}")
                    continue
                g = grade_parallel(msg, tools)
                rows.append(dict(model=model, n=n, secs=round(dt, 2), **g))
                tick = lambda k: "yes" if g[k] else "NO"
                print(f"{model:<18}{n:>6}{g['n_calls']:>7}{tick('batched'):>9}"
                      f"{tick('both_files'):>9}{tick('names_ok'):>7}{dt:>7.1f}")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
            print(f"\nraw -> {a.json}")
        good = [r for r in rows if r["batched"] and r["both_files"]]
        print(f"\n{len(good)}/{len(rows)} batched two independent calls into "
              f"one response.")
        return 0

    if a.loop:
        sysp = CC_SYSTEM if a.cc_system else None
        print(f"two-turn loop, {len(counts)} size(s), system prompt: "
              f"{'yes' if sysp else 'no'}\n"
              f"{'backend':<18}{'tools':>6}  {'turn1':>6}{'accepted':>9}"
              f"{'answered':>9}{'used':>6}{'secs':>7}  reply")
        rows = []
        for model in models:
            for n in counts:
                g, dt = loop_probe(model, n, sysp)
                rows.append(dict(model=model, n=n, secs=round(dt, 2), **g))
                tick = lambda k: "yes" if g[k] else "NO"
                print(f"{model:<18}{n:>6}  {tick('turn1'):>6}"
                      f"{tick('accepted'):>9}{tick('answered'):>9}"
                      f"{tick('used_result'):>6}{dt:>7.1f}  {g['note']}")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
            print(f"\nraw -> {a.json}")
        broken = [r for r in rows if not (r["accepted"] and r["used_result"])]
        print(f"\n{len(rows) - len(broken)}/{len(rows)} completed a real "
              f"tool-call/tool-result/continue cycle.")
        return 1 if broken else 0

    print(f"{len(models)} backend(s), {len(probes)} probes, counts {counts} "
          f"-> {len(models) * len(probes) * len(counts)} requests")
    print(f"tool pool: {total_pool} ({len(CORE_TOOLS)} real Claude Code, "
          f"{len(FILLER_TOOLS)} real deferred names)\n")

    raw, table = [], {}
    for model in models:
        for n in counts:
            cells = []
            for i, probe in enumerate(probes):
                tools, at = build_tools(n, probe["tool"], i)
                msg, dt, err = ask_retry(model, probe["prompt"], tools,
                                         CC_SYSTEM if a.cc_system else None)
                kind = classify(err)
                if kind == "refused":
                    g = dict(schema_ok=False, called=False, right_tool=False,
                             valid_args=False, no_halluc=False, types_ok=False)
                elif kind == "unavailable":
                    g = None            # no data; must not become a zero
                else:
                    g = grade(msg, probe, tools)
                got = ((msg or {}).get("tool_calls") or [{}])[0].get(
                    "function", {}).get("name", "")
                cells.append(g)
                raw.append(dict(model=model, n=n, probe=probe["id"], pos=at,
                                got=got, latency=round(dt, 2), error=err,
                                status=kind, **(g or {})))
                if err:
                    print(f"  {model:<18} n={n:<4} {probe['id']:<6} "
                          f"[{kind}] {err[:100]}")
                # A wrong tool or a malformed argument is the finding, so say it
                # while the sweep runs rather than only in the raw JSON.
                elif g and not (g["right_tool"] and g["types_ok"]):
                    why = g.get("type_err") or (f"called {got or 'nothing'}")
                    print(f"  {model:<18} n={n:<4} {probe['id']:<9} {why[:90]}")
            table[(model, n)] = cells
        # One line per backend as it finishes, so a long sweep is readable live.
        print(f"  {model} done")

    print(f"\n{'backend':<18}{'tools':>6}{'n':>4}  {'schema':>7}{'called':>8}"
          f"{'right':>7}{'args':>7}{'real':>7}{'types':>7}")
    print("-" * 73)
    for model in models:
        for n in counts:
            scored = [x for x in table[(model, n)] if x is not None]
            k = len(scored)
            if not k:
                print(f"{model:<18}{n:>6}{0:>4}  {'-':>7}{'-':>8}{'-':>7}"
                      f"{'-':>7}{'-':>7}{'-':>7}   all unavailable")
                continue
            def pc(key):
                return f"{sum(1 for x in scored if x[key]) / k:.0%}"
            print(f"{model:<18}{n:>6}{k:>4}  {pc('schema_ok'):>7}{pc('called'):>8}"
                  f"{pc('right_tool'):>7}{pc('valid_args'):>7}{pc('no_halluc'):>7}"
                  f"{pc('types_ok'):>7}")
        print()

    # The headline: the largest swept size where every probe still picked the
    # right tool AND filled its required arguments. Reported as "at least" --
    # this sweeps a grid, so the true ceiling sits somewhere in the gap above.
    # A size where nothing was scored is not evidence either way and cannot be
    # allowed to end the ceiling; it is reported as a gap instead.
    print("ceiling (all probes right_tool AND valid_args; n = probes scored):")
    for model in models:
        ok, gaps = [], []
        for n in counts:
            scored = [x for x in table[(model, n)] if x is not None]
            if not scored:
                gaps.append(n)
            elif all(x["right_tool"] and x["valid_args"] for x in scored):
                ok.append(n)
        note = f"   (no data at {gaps})" if gaps else ""
        if not ok:
            print(f"  {model:<18} no size fully passed{note}")
        else:
            print(f"  {model:<18} >={max(ok)}{note}")

    if a.json:
        Path(a.json).write_text(json.dumps(raw, indent=1))
        print(f"\nraw -> {a.json}")
    return 0


def _selftest():
    # build_tools is the only real logic here, and every trap it exists to avoid
    # is a silent one: a target that vanished, a duplicate, or a fixed position
    # that turns "picks tools[0]" into a perfect score.
    for n in (1, 2, 5, 40):
        for seed in range(4):
            tools, at = build_tools(n, "Grep", seed)
            names = [t["function"]["name"] for t in tools]
            assert len(tools) == n, f"asked for {n}, got {len(tools)}"
            assert names.count("Grep") == 1, f"target must appear exactly once: {names}"
            assert len(set(names)) == n, "no duplicate tool names"
            assert names[at] == "Grep", "reported position must be the real one"
    # Position must actually move, or the sweep measures index 0 forever.
    assert len({build_tools(9, "Grep", s)[1] for s in range(4)}) > 1, \
        "target position must vary with seed"
    # The whole pool, not just the core set: the `mcp` probe deliberately targets
    # a filler, because "can it pick one MCP tool out of 97" is a real question.
    assert {p["tool"] for p in PROBES} <= ({t["function"]["name"] for t in CORE_TOOLS}
                                           | set(FILLER_NAMES)), \
        "every probe's answer must exist somewhere in the tool pool"
    assert len(set(FILLER_NAMES)) == len(FILLER_NAMES), "duplicate filler name"
    assert not ({t["function"]["name"] for t in CORE_TOOLS} & set(FILLER_NAMES)), \
        "a name in both pools would let build_tools drop the target"
    # An empty args dict is valid JSON and a useless call; it must not score.
    g = grade({"tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]},
              PROBES[0], CORE_TOOLS)
    assert g["right_tool"] and not g["valid_args"], "empty args must fail valid_args"
    g = grade({"content": "I would use the Read tool."}, PROBES[0], CORE_TOOLS)
    assert not g["called"] and not g["right_tool"], "prose about a tool is not a call"

    # ---- schema validation ------------------------------------------------
    # The case that motivated types_ok: required key present, shape entirely
    # wrong. valid_args says yes, and the real client would reject the call.
    _task = next(p for p in PROBES if p["tool"] == "TaskCreate")
    _flat = grade({"tool_calls": [{"function": {
        "name": "TaskCreate",
        "arguments": '{"tasks": "migrate the database"}'}}]}, _task, CORE_TOOLS)
    assert _flat["valid_args"], "key IS present -- this is why valid_args is not enough"
    assert not _flat["types_ok"], "a string where an array belongs must fail types_ok"

    _good = grade({"tool_calls": [{"function": {
        "name": "TaskCreate",
        "arguments": '{"tasks": [{"description": "migrate", "status": "pending"}]}'
    }}]}, _task, CORE_TOOLS)
    assert _good["types_ok"], f"a correct nested call must pass: {_good.get('type_err')}"

    _badenum = grade({"tool_calls": [{"function": {
        "name": "TaskCreate",
        "arguments": '{"tasks": [{"description": "migrate", "status": "doing"}]}'
    }}]}, _task, CORE_TOOLS)
    assert not _badenum["types_ok"], "a value outside the enum must fail"
    assert "doing" in _badenum["type_err"], "the error should name the bad value"

    # A wrong tool cannot be validated against the right tool's schema.
    _wrong = grade({"tool_calls": [{"function": {
        "name": "Write", "arguments": '{"file_path": "/a", "content": "b"}'}}]},
        _task, CORE_TOOLS)
    assert not _wrong["types_ok"], "wrong tool is never types_ok"

    # Nested required, and the bool/int confusion Python invites.
    assert schema_errors({"tasks": [{"description": "x"}]},
                         CORE_TOOLS[8]["function"]["parameters"]), \
        "a missing nested required field must be caught"
    assert schema_errors(True, {"type": "integer"}), "True is not an integer"
    assert not schema_errors(3, {"type": "number"}), "an int satisfies number"
    assert schema_errors([], {"type": "array", "items": {"type": "string"}}), \
        "an empty array is not a filled-in argument"

    # A count would drift silently; what matters is that no resident tool is
    # unprobed, which is exactly the gap that made "108 tools" mean less than
    # it sounded -- 103 of them were decoys nobody ever asked for.
    _probed = {p["tool"] for p in PROBES}
    assert _probed >= {t["function"]["name"] for t in CORE_TOOLS}, \
        f"unprobed resident tool(s): {{t['function']['name'] for t in CORE_TOOLS}} - {_probed}"
    assert len(_probed) == len(PROBES), "one probe per tool"
    assert _probed & set(FILLER_NAMES), "at least one MCP/deferred tool must be probed"
    # An unanswerable probe measures the probe. `tokensave_callers` vs
    # `tokensave_callers_for` cost one wrong "the model cannot pick MCP tools".
    assert not (_probed & PREFIX_TWINS), \
        f"probe target has an indistinguishable twin: {_probed & PREFIX_TWINS}"

    # The parallel grader: two calls to the same tool with different arguments.
    _par = grade_parallel({"tool_calls": [
        {"function": {"name": "Read", "arguments": '{"file_path": "/etc/hostname"}'}},
        {"function": {"name": "Read", "arguments": '{"file_path": "/etc/timezone"}'}},
    ]}, CORE_TOOLS)
    assert _par["batched"] and _par["both_files"] and _par["names_ok"], _par
    _one = grade_parallel({"tool_calls": [
        {"function": {"name": "Read", "arguments": '{"file_path": "/etc/hostname"}'}}]},
        CORE_TOOLS)
    assert not _one["batched"], "one call is not a batch"
    # Same file twice is a batch of one useful call, and must not score both_files.
    _dup = grade_parallel({"tool_calls": [
        {"function": {"name": "Read", "arguments": '{"file_path": "/etc/hostname"}'}},
        {"function": {"name": "Read", "arguments": '{"file_path": "/etc/hostname"}'}},
    ]}, CORE_TOOLS)
    assert _dup["batched"] and not _dup["both_files"], "duplicate paths are not two files"
    # Real groq 429, verbatim -- the request alone exceeds the whole minute.
    _hopeless = ("RateLimitError: GroqException - Rate limit reached for model "
                 "`llama-3.3-70b-versatile` on tokens per minute (TPM): "
                 "Limit 12000, Used 9547, Requested 11747. Try again in 46.47s.")
    assert over_budget(_hopeless), "a request bigger than the cap must not retry"
    assert classify(_hopeless) == "unavailable", "still no-data, never a zero"
    # The other real shape: no `Used`, and the request alone busts the cap.
    # Requiring `Used` made this permanent failure the one thing that DID retry.
    _worse = ("RateLimitError: GroqException - Request too large for model "
              "`qwen/qwen3.6-27b` on tokens per minute (TPM): Limit 8000, "
              "Requested 12993, please reduce your message size and try again.")
    assert over_budget(_worse), "the 'no Used' shape must be caught too"
    assert "idle account" in over_budget(_worse), \
        "over-cap is permanent, and the message should say so"
    # Same shape, but the request fits -- that one IS worth waiting out.
    assert over_budget(_hopeless.replace("Requested 11747", "Requested 900")) is None, \
        "a small request behind someone else's traffic is transient, so retry"
    assert over_budget("HTTP 500: boom") is None, "no TPM numbers -> no verdict"
    print("tools.py selftest: ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
