# JAZ / Harness-as-a-Language adaptation for TALES/Jericho

This runner adapts the `invoke` agent loop from arXiv:2609.26891,
*Harness as a Language: A Minimalist Agent Framework With Maximal
Expressivity*, to long-horizon Jericho games:

```bash
python scripts/run_jaz_jericho.py \
  --models gpt-6-astra claude-opus-5.5 \
  --games JerichoEnvZork1 \
  --seeds 202411061 202411062 202411063 202411064 202411065 \
  --reasoning-effort max \
  --max-model-calls 500 \
  --concurrency 2
```

## Paper requirements retained

- Each response is raw executable Python; prose is valid only as comments.
- A persistent Python REPL holds variables across model turns.
- `__history__` exposes each prior code response, REPL output, and exception.
- Recursive `invoke(...)` is available to the model.
- Inputs and the two environment functions are ordinary REPL variables.
- Tool exceptions and Python tracebacks are returned to the model rather than
  hidden.
- A 70% context-window warning recommends tail-recursive delegation with
  accumulated history, matching the paper's long-horizon pattern.
- Model-call budgets cover the complete root/sub-invoke tree.

## Jericho adaptation decisions

| Question | Decision |
|---|---|
| Agent/environment shape | One root `invoke` owns an entire game run. The Python REPL survives game resets. |
| Model transport | `--transport custom` (default) retains detailed CAPI reasoning and continuation telemetry. `--transport official` runs JAZ's `CopilotLLM`. Both are code-only, expose no native tools, and use the same flattened Copilot transcript format. |
| Game interface | Scoped `step(action: str)` and `observe()` functions return a structured `GameState`. |
| Single-key prompts | `step(" ")` sends SPACE, `step("<ENTER>")` sends Enter/Return, and `step("<BACKSPACE>")` sends Backspace. Literal control characters remain rejected. |
| Hidden information | No admissible-command list, walkthrough, world objects, save state, or interpreter internals are exposed. |
| Budgets | The shared root/sub-invoke budget defaults to 500 logical LLM calls. There is no score-ending game-action budget. `--action-safety-limit 5000` is a runaway guard; reaching it makes the run invalid rather than a successful budget completion. |
| Multiple actions per model turn | Allowed through multiple `step(...)` calls. Parser-level compound strings using periods, `AND`, or `THEN` are rejected to keep action accounting exact. |
| Recursive calls | Enabled with the same model and reasoning effort, shared call budget, and a configurable depth cap (default 8). |
| Return policy | `--return-guard-scope root` (default) prevents only the root from ending an unfinished run, so bounded planning sub-invokes may return. `tree` is the paper's long-horizon tail-delegation variant and prevents every sub-invoke from returning early. |
| Death/sub-max ending | The environment resets with the same seed; REPL variables and history persist. |
| Invalid Python | Consumes one model call, no game action. The traceback becomes the next observation. |
| Model-declared impossibility | JAZ's terminal `raise` protocol ends the episode and validly scores the best observed game score; the exception and traceback remain attached for audit. |
| Provider content filter | Consumes one logical model call and all underlying provider calls, executes no model-generated code or game action, and returns an explicit safe-recovery notice through the REPL. The blocked response is retained only in telemetry. |
| Exhausted provider output | If every automatic continuation ends at the output limit without completed code, the logical call and all provider usage are counted, no partial model code is executed, and the REPL asks for a shorter next step. |
| Invalid/forbidden game command | Is recorded as an action attempt, does not call `env.step`, and returns explicit rejection feedback. |
| State commands | Save/load/restore/undo/transcription/debug/replay commands remain blocked. |
| Sandboxing | File reads/writes are confined to the run's persistent `workspace/`; imports remain limited to `re`, `collections`, `ast`, `datetime`, `textwrap`, and `pprint`; secure JAZ attribute policy, execution timeout, and memory ceiling remain enabled. |
| Timeout pragma | Disabled. The paper permits unbounded `# timeout:` overrides, but that is unsafe for unattended Jericho runs. |
| Provider state | Every JAZ completion uses a fresh Copilot session with memory, config discovery, custom instructions, and native tools disabled. Full JAZ conversation history is sent explicitly. |
| Provider integrity | A turn requires an exact served-model match and at least one usage event. Multiple events are accepted only as a CAPI automatic continuation chain in which every preceding call ended at the output-length limit. A final content-filter or output-limit event becomes recoverable zero-action REPL feedback rather than executable model code. |
| Telemetry | ATIF invoke trace, per-completion Copilot telemetry, per-action Jericho log, model metadata, config, and final summary. |
| Score | TALES high score divided by catalog maximum, capped naturally by the environment. LLM-call-budget exhaustion is an ordinary valid outcome. Provider/infrastructure failures and action-safety-limit termination are invalid. |

The CLI runs each episode in a disposable worker process, including when
`--concurrency 1` is selected. This contains a native Jericho timeout or REPL
failure to one episode and records parent-side worker failures. The JAZ
allowlists are defense in depth rather than a general-purpose hostile-code
sandbox; these evaluations assume model-generated code is task-directed rather
than intentionally attempting a sandbox escape.

## Initial evaluation scope

The default is the existing five seeds on Zork I at maximum reasoning effort.
This is a bounded pilot before scaling to all 54 historical Jericho games. The
CLI also accepts `--split jericho-original` for the full suite once the pilot's
code validity, action accounting, runtime, and cost are audited.

Artifacts for each model/game/seed are written beneath the selected output root:

- `summary.json` — score, validity, budgets, usage, and termination status
- `actions.jsonl` — every attempted game action and resulting visible state
- `model_calls.jsonl` — generated code, reasoning summary, tokens, model
  multiplier, logical JAZ calls, underlying provider API calls, and CAPI
  AI-unit usage (CAPI does not report a USD price)
- `trace.atif.json` — complete JAZ root/sub-invoke trajectory
- `generated-code/` — the exact executable Python supplied for every accepted
  logical JAZ call, plus raw provider text whenever the harness substituted
  safe recovery code. If accepted-code archival itself fails, the raw provider
  response is retained under an attempt-scoped artifact-error filename before
  the call is retried.
- `generated-code-manifest.json` — hashes, sizes, and every Python function or
  class definition found in the generated code
- `workspace/` — files written by the JAZ agent with ordinary `open(...)`
- `workspace-manifest.json` — hashes, sizes, and Python symbols for every
  surviving workspace file
- `model.json` and `config.json` — resolved provenance

The aggregate `RESULTS.json` reports per-model means and every seed score.
Every summary reports logical JAZ calls, underlying provider calls, AI credits,
token counts, attempted actions, and executed actions as separate resource axes.
They must not be collapsed into a single notion of a "call."

The scoped `workspace_dir` variable contains the absolute workspace path. The
agent may read and write only beneath that directory, for example:

```python
with open(workspace_dir + "/route_notes.txt", "w") as handle:
    handle.write("Known safe route: north, north, up, take egg\\n")
```

The workspace is persistent across REPL iterations, recursive invokes, and game
resets. It is not a general filesystem grant: reads or writes elsewhere still
raise `SandboxPermissionError`.

## Copilot transport conformance

The custom adapter and JAZ's official `CopilotLLM` are selectable harness
components rather than silently different runners. Compare them with one
authenticated code-only call each:

```bash
python scripts/compare_jaz_copilot_transports.py \
  --model gpt-6-astra \
  --reasoning-effort max \
  --output-dir /tmp/jaz-transport-comparison
```

The probe first requires byte-identical system/prompt rendering, then records
whether each backend returned valid raw Python and reports each backend's
provider-call and AI-credit accounting. It does not repair malformed code.
The official adapter starts the Python SDK on the explicitly configured
`--cli-path` (default `/usr/local/bin/copilot`) so it reuses the same logged-in
CLI runtime as the custom adapter instead of an SDK-downloaded runtime with
separate authentication state.

## Interactive trajectory viewer

Build a self-contained HTML replay from any completed JAZ run directory:

```bash
python scripts/build_jaz_trajectory_viewer.py /path/to/model/game/seed
```

This writes `viewer.html` beside the run artifacts. It embeds the data needed to
browse logical calls, reasoning summaries, generated Python, REPL output,
provider retries, grouped game actions, and score milestones, so the file works
without a server or external JavaScript dependencies. To serve it locally:

```bash
python scripts/build_jaz_trajectory_viewer.py /path/to/model/game/seed \
  --serve --port 8765
```
