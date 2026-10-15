#!/usr/bin/env python3
"""Build a self-contained interactive viewer for a JAZ trajectory."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote


REQUIRED_ARTIFACTS = (
    "summary.json",
    "actions.jsonl",
    "model_calls.jsonl",
    "trace.atif.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read JSONL artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
        rows.append(value)
    return rows


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _call_title(code: str, call_number: int) -> str:
    for line in code.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("In ["):
            continue
        if candidate.startswith("#"):
            candidate = candidate.lstrip("#").strip()
            if candidate.lower().startswith("plan:"):
                candidate = candidate[5:].strip()
            if candidate:
                return candidate[:100]
        break
    return f"Logical call {call_number}"


def _trace_observations(trace: dict[str, Any]) -> dict[int, str]:
    observations: dict[int, str] = {}
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return observations
    for step in steps:
        if not isinstance(step, dict) or step.get("source") != "user":
            continue
        extra = step.get("extra")
        iteration = extra.get("iteration") if isinstance(extra, dict) else None
        message = step.get("message")
        if isinstance(iteration, int) and isinstance(message, str):
            observations[iteration + 1] = message
    return observations


def _action_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": _integer(row.get("action_index")),
        "command": row.get("submitted_action"),
        "executed_command": row.get("executed_action"),
        "executed": bool(row.get("action_was_executed")),
        "invalid_reason": row.get("invalid_reason"),
        "blocked_command": row.get("blocked_command"),
        "observation": row.get("observation") or row.get("feedback") or "",
        "reward": _integer(row.get("reward")),
        "score": _integer(row.get("score")),
        "highscore": _integer(row.get("highscore")),
        "moves": _integer(row.get("moves")),
        "episode_done": bool(row.get("episode_done")),
        "episode_won": bool(row.get("episode_won")),
        "episode_lost": bool(row.get("episode_lost")),
        "restarted": bool(row.get("restarted")),
        "run_complete": bool(row.get("run_complete")),
        "completion_reason": row.get("completion_reason"),
    }


def _retry_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": _integer(row.get("attempt")),
        "error_type": row.get("error_type") or "ProviderError",
        "error": row.get("error") or "Unknown provider error",
        "duration_seconds": _number(row.get("duration")),
        "response": row.get("response"),
    }


def build_viewer_payload(run_dir: Path) -> dict[str, Any]:
    """Load one run directory and produce the compact browser data model."""
    run_dir = Path(run_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"Run directory {run_dir} is missing required artifacts: "
            f"{', '.join(missing)}"
        )

    summary = _read_json(run_dir / "summary.json")
    trace = _read_json(run_dir / "trace.atif.json")
    model_rows = _read_jsonl(run_dir / "model_calls.jsonl")
    action_rows = _read_jsonl(run_dir / "actions.jsonl")
    observations = _trace_observations(trace)

    actions_by_call: dict[int, list[dict[str, Any]]] = defaultdict(list)
    chart: list[dict[str, Any]] = [{"action": 0, "score": 0, "call": 0}]
    milestones: list[dict[str, Any]] = []
    previous_highscore = 0
    for row in action_rows:
        call_number = _integer(row.get("model_calls_completed"))
        action = _action_payload(row)
        actions_by_call[call_number].append(action)
        chart.append(
            {
                "action": action["index"],
                "score": action["highscore"],
                "call": call_number,
            }
        )
        if action["highscore"] > previous_highscore:
            milestones.append(
                {
                    "action": action["index"],
                    "call": call_number,
                    "score": action["highscore"],
                    "delta": action["highscore"] - previous_highscore,
                    "command": action["command"],
                    "observation": action["observation"],
                }
            )
            previous_highscore = action["highscore"]

    calls: list[dict[str, Any]] = []
    pending_retries: list[dict[str, Any]] = []
    highscore = 0
    actions_seen = 0
    for row in model_rows:
        if row.get("status") != "ok":
            pending_retries.append(_retry_payload(row))
            continue

        call_number = _integer(row.get("call"))
        call_actions = actions_by_call.get(call_number, [])
        score_before = highscore
        actions_before = actions_seen
        if call_actions:
            highscore = call_actions[-1]["highscore"]
            actions_seen = call_actions[-1]["index"]
        totals = row.get("totals") if isinstance(row.get("totals"), dict) else {}
        code = str(row.get("response") or "")
        calls.append(
            {
                "number": call_number,
                "attempt": _integer(row.get("attempt")),
                "title": _call_title(code, call_number),
                "timestamp": _number(row.get("timestamp")),
                "duration_seconds": _number(row.get("duration")),
                "reasoning": row.get("reasoning_summary") or "",
                "reasoning_is_summary": bool(row.get("reasoning_is_summary")),
                "code": code,
                "repl_output": observations.get(call_number, ""),
                "response_transformed": bool(row.get("response_transformed")),
                "response_transform_reason": row.get("response_transform_reason"),
                "retry_errors": pending_retries,
                "actions": call_actions,
                "action_count": len(call_actions),
                "actions_before": actions_before,
                "actions_after": actions_seen,
                "score_before": score_before,
                "score_after": highscore,
                "score_delta": highscore - score_before,
                "provider_calls": _integer(totals.get("api_calls"), 1),
                "prompt_tokens": _integer(totals.get("input_tokens")),
                "completion_tokens": _integer(totals.get("output_tokens")),
                "reasoning_tokens": _integer(totals.get("reasoning_tokens")),
                "cached_tokens": _integer(totals.get("cache_read_tokens")),
                "cache_write_tokens": _integer(totals.get("cache_write_tokens")),
                "ai_credits": _number(row.get("ai_credits")),
            }
        )
        pending_retries = []

    game_result = (
        summary.get("game_result")
        if isinstance(summary.get("game_result"), dict)
        else {}
    )
    model_usage = (
        summary.get("model_usage")
        if isinstance(summary.get("model_usage"), dict)
        else {}
    )
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    trace_agent = trace.get("agent") if isinstance(trace.get("agent"), dict) else {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run": {
            "id": summary.get("run_id") or run_dir.name,
            "status": summary.get("status"),
            "score_valid": bool(summary.get("score_valid")),
            "model": summary.get("model") or trace_agent.get("model_name"),
            "reasoning_effort": summary.get("reasoning_effort"),
            "game": game_result.get("game") or config.get("game"),
            "seed": game_result.get("seed") or config.get("seed"),
            "score": _integer(game_result.get("score")),
            "max_score": _integer(game_result.get("max_score")),
            "moves": _integer(game_result.get("moves")),
            "actions": _integer(game_result.get("actions_used"), len(action_rows)),
            "action_limit": config.get("max_actions"),
            "action_safety_limit": config.get("action_safety_limit"),
            "return_guard_scope": summary.get("return_guard_scope")
            or config.get("return_guard_scope"),
            "transport": summary.get("transport") or config.get("transport"),
            "logical_calls": _integer(model_usage.get("completed_calls"), len(calls)),
            "attempts": _integer(
                model_usage.get("completion_attempts"), len(model_rows)
            ),
            "provider_calls": _integer(model_usage.get("provider_api_calls")),
            "prompt_tokens": _integer(model_usage.get("prompt_tokens")),
            "completion_tokens": _integer(model_usage.get("completion_tokens")),
            "reasoning_tokens": _integer(model_usage.get("reasoning_tokens")),
            "ai_credits": _number(model_usage.get("ai_credits")),
            "duration_seconds": _number(summary.get("duration_seconds")),
            "completion_reason": game_result.get("completion_reason"),
            "initial_observation": game_result.get("initial_observation") or "",
            "final_observation": game_result.get("final_observation") or "",
            "result": summary.get("result"),
            "provider_retry_count": sum(len(call["retry_errors"]) for call in calls)
            + len(pending_retries),
        },
        "trace": {
            "schema_version": trace.get("schema_version"),
            "trajectory_id": trace.get("trajectory_id"),
            "notes": trace.get("notes"),
        },
        "calls": calls,
        "trailing_retry_errors": pending_retries,
        "chart": chart,
        "milestones": milestones,
        "artifact_links": {},
    }


def _browser_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_viewer(run_dir: Path, output_path: Path | None = None) -> Path:
    """Build the viewer and return its absolute output path."""
    run_dir = Path(run_dir).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else run_dir / "viewer.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_payload(run_dir)
    payload["artifact_links"] = {
        name: Path(os.path.relpath(run_dir / name, output_path.parent)).as_posix()
        for name in REQUIRED_ARTIFACTS
    }
    document = HTML_TEMPLATE.replace("__TRAJECTORY_DATA__", _browser_json(payload))
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_viewer(output_path: Path, host: str, port: int) -> None:
    handler = partial(_QuietHandler, directory=str(output_path.parent))
    server = ThreadingHTTPServer((host, port), handler)
    display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    url = f"http://{display_host}:{port}/{quote(output_path.name)}"
    print(f"Serving JAZ trajectory viewer at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML viewer for a JAZ run directory."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = build_viewer(args.run_dir, args.output)
    print(f"Wrote {output_path}", flush=True)
    if args.serve:
        serve_viewer(output_path, args.host, args.port)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JAZ trajectory viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d14;
      --panel: #101722;
      --panel-2: #151e2c;
      --border: #273246;
      --text: #e7edf7;
      --muted: #93a2b8;
      --accent: #8b5cf6;
      --accent-2: #22d3ee;
      --green: #34d399;
      --yellow: #fbbf24;
      --red: #fb7185;
      --code: #0a1019;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    button, input { font: inherit; }
    button { color: inherit; }
    a { color: var(--accent-2); }
    .shell { min-height: 100vh; }
    .hero {
      padding: 24px 28px 18px;
      border-bottom: 1px solid var(--border);
      background:
        radial-gradient(circle at 10% -40%, rgba(139, 92, 246, .38), transparent 45%),
        radial-gradient(circle at 90% -20%, rgba(34, 211, 238, .22), transparent 36%),
        var(--panel);
    }
    .eyebrow {
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 { margin: 6px 0 4px; font-size: clamp(24px, 4vw, 42px); line-height: 1.05; }
    .subtitle { color: var(--muted); font-size: 14px; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-top: 20px;
    }
    .stat {
      background: rgba(9, 13, 20, .62);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      min-width: 0;
    }
    .stat-label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .stat-value { margin-top: 3px; font-size: 20px; font-weight: 750; }
    .chart-panel { padding: 16px 28px; border-bottom: 1px solid var(--border); }
    .section-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .section-heading h2, .section-heading h3 { margin: 0; }
    #score-chart {
      display: block;
      width: 100%;
      height: 170px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(139,92,246,.08), transparent);
    }
    .milestones { display: flex; gap: 7px; overflow-x: auto; padding: 10px 0 2px; }
    .milestone {
      white-space: nowrap;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--panel);
      padding: 6px 9px;
      cursor: pointer;
      font-size: 12px;
    }
    .milestone:hover { border-color: var(--green); }
    .layout { display: grid; grid-template-columns: 330px minmax(0, 1fr); }
    .sidebar {
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      border-right: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      flex-direction: column;
    }
    .filters { padding: 14px; border-bottom: 1px solid var(--border); }
    .search {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: var(--bg);
      color: var(--text);
      padding: 9px 10px;
      outline: none;
    }
    .search:focus { border-color: var(--accent); }
    .checks { display: flex; flex-wrap: wrap; gap: 8px 12px; margin-top: 10px; }
    .checks label { color: var(--muted); font-size: 12px; }
    .call-list { overflow: auto; padding: 8px; }
    .call-button {
      display: block;
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 9px;
      background: transparent;
      padding: 9px;
      cursor: pointer;
      margin-bottom: 3px;
    }
    .call-button:hover { background: var(--panel-2); }
    .call-button.active {
      background: rgba(139,92,246,.14);
      border-color: var(--accent);
    }
    .call-number { color: var(--accent-2); font-size: 11px; font-weight: 800; }
    .call-title { font-size: 13px; margin: 3px 0; line-height: 1.3; }
    .call-meta { color: var(--muted); font-size: 11px; }
    .score-up { color: var(--green); font-weight: 800; }
    .retry-dot { color: var(--red); }
    .main { min-width: 0; padding: 22px 26px 60px; }
    .call-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 14px;
    }
    .call-header h2 { margin: 3px 0 7px; font-size: 25px; }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; }
    .badge {
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      padding: 4px 8px;
      font-size: 11px;
    }
    .badge.good { color: var(--green); border-color: rgba(52,211,153,.45); }
    .badge.warn { color: var(--yellow); border-color: rgba(251,191,36,.45); }
    .nav-buttons { display: flex; gap: 7px; }
    .nav-buttons button, .copy-button {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 7px 10px;
      cursor: pointer;
    }
    .nav-buttons button:hover, .copy-button:hover { border-color: var(--accent); }
    .nav-buttons button:disabled { opacity: .4; cursor: default; }
    .call-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .mini-stat {
      padding: 9px 10px;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: var(--panel);
    }
    .mini-stat span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; }
    .mini-stat strong { display: block; margin-top: 2px; font-size: 14px; }
    .retry-box {
      border: 1px solid rgba(251,113,133,.45);
      background: rgba(251,113,133,.08);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }
    .retry-box h3 { color: var(--red); margin: 0 0 8px; font-size: 14px; }
    .retry-item { color: #fecdd3; font-size: 12px; margin-top: 5px; }
    .two-column {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }
    details.card, .card {
      border: 1px solid var(--border);
      border-radius: 11px;
      background: var(--panel);
      overflow: hidden;
      margin-bottom: 12px;
    }
    details.card summary {
      cursor: pointer;
      font-weight: 700;
      padding: 11px 13px;
      border-bottom: 1px solid transparent;
    }
    details.card[open] summary { border-bottom-color: var(--border); }
    .card-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 11px 13px;
      border-bottom: 1px solid var(--border);
      font-weight: 700;
    }
    pre {
      margin: 0;
      padding: 14px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: var(--code);
      max-height: 560px;
      overflow: auto;
    }
    .empty { color: var(--muted); padding: 14px; font-style: italic; }
    .actions-heading { margin: 20px 0 9px; }
    .action {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      margin-bottom: 8px;
      overflow: hidden;
    }
    .action summary {
      cursor: pointer;
      list-style: none;
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
    }
    .action summary::-webkit-details-marker { display: none; }
    .action-index { color: var(--muted); font-size: 11px; }
    .command { font: 600 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .action-score { font-size: 12px; white-space: nowrap; }
    .action-observation { border-top: 1px solid var(--border); }
    .footer {
      color: var(--muted);
      font-size: 12px;
      margin-top: 24px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }
    .raw-links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .hidden { display: none !important; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; height: 420px; border-right: 0; border-bottom: 1px solid var(--border); }
      .two-column { grid-template-columns: 1fr; }
      .main { padding: 18px 15px 50px; }
      .hero, .chart-panel { padding-left: 15px; padding-right: 15px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <div class="eyebrow">Harness-as-a-Language replay</div>
      <h1 id="title">JAZ trajectory</h1>
      <div class="subtitle" id="subtitle"></div>
      <div class="summary-grid" id="summary-grid"></div>
    </header>
    <section class="chart-panel">
      <div class="section-heading">
        <h2>Score progression</h2>
        <span class="subtitle">Milestones jump to the responsible logical call</span>
      </div>
      <svg id="score-chart" role="img" aria-label="Score by game action"></svg>
      <div class="milestones" id="milestones"></div>
    </section>
    <div class="layout">
      <aside class="sidebar">
        <div class="filters">
          <input class="search" id="search" type="search"
            placeholder="Search reasoning, code, output, actions…">
          <div class="checks">
            <label><input id="score-filter" type="checkbox"> score changes</label>
            <label><input id="action-filter" type="checkbox"> has actions</label>
            <label><input id="retry-filter" type="checkbox"> provider retries</label>
          </div>
        </div>
        <div class="call-list" id="call-list"></div>
      </aside>
      <main class="main" id="main"></main>
    </div>
  </div>
  <script id="trajectory-data" type="application/json">__TRAJECTORY_DATA__</script>
  <script>
    "use strict";
    const data = JSON.parse(document.getElementById("trajectory-data").textContent);
    const state = { selected: 1, visible: data.calls.map(call => call.number) };
    const number = new Intl.NumberFormat("en-US");

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }

    function formatCount(value) {
      return number.format(Number(value || 0));
    }

    function formatDuration(seconds) {
      const value = Number(seconds || 0);
      if (value < 60) return `${value.toFixed(1)}s`;
      const hours = Math.floor(value / 3600);
      const minutes = Math.floor((value % 3600) / 60);
      const remainder = Math.round(value % 60);
      return hours ? `${hours}h ${minutes}m ${remainder}s` : `${minutes}m ${remainder}s`;
    }

    function summaryStat(label, value) {
      const card = el("div", "stat");
      card.append(el("div", "stat-label", label), el("div", "stat-value", value));
      return card;
    }

    function miniStat(label, value) {
      const card = el("div", "mini-stat");
      card.append(el("span", "", label), el("strong", "", value));
      return card;
    }

    function renderHeader() {
      const run = data.run;
      document.title = `${run.model} · ${run.game} · JAZ trajectory`;
      document.getElementById("title").textContent =
        `${run.model} won ${run.game} — ${run.score}/${run.max_score}`;
      document.getElementById("subtitle").textContent =
        `Seed ${run.seed} · ${run.reasoning_effort} reasoning · ${run.transport || "custom"} transport · ${run.return_guard_scope || "tree"} guard · ${run.status} · ${data.run_dir}`;
      const grid = document.getElementById("summary-grid");
      const actionLimit = run.action_limit == null ? "unlimited" : formatCount(run.action_limit);
      const actionSafety = run.action_safety_limit == null
        ? "disabled"
        : formatCount(run.action_safety_limit);
      [
        ["Score", `${run.score}/${run.max_score}`],
        ["Game actions", `${formatCount(run.actions)} / ${actionLimit}`],
        ["Action safety guard", actionSafety],
        ["Logical JAZ calls", formatCount(run.logical_calls)],
        ["Provider calls", formatCount(run.provider_calls)],
        ["Retried failures", formatCount(run.provider_retry_count)],
        ["Game moves", formatCount(run.moves)],
        ["Reasoning tokens", formatCount(run.reasoning_tokens)],
        ["Wall time", formatDuration(run.duration_seconds)],
      ].forEach(([label, value]) => grid.append(summaryStat(label, value)));
    }

    function svgNode(tag, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      return node;
    }

    function renderChart() {
      const svg = document.getElementById("score-chart");
      const width = 1000, height = 170, left = 42, right = 16, top = 14, bottom = 26;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const maxAction = Math.max(...data.chart.map(point => point.action), 1);
      const maxScore = Math.max(data.run.max_score, 1);
      const x = action => left + (Number(action) / maxAction) * (width - left - right);
      const y = score => top + (1 - Number(score) / maxScore) * (height - top - bottom);

      [0, .25, .5, .75, 1].forEach(fraction => {
        const score = Math.round(maxScore * fraction);
        const line = svgNode("line", {
          x1: left, x2: width - right, y1: y(score), y2: y(score),
          stroke: "#273246", "stroke-width": "1"
        });
        const label = svgNode("text", {
          x: left - 8, y: y(score) + 4, fill: "#93a2b8",
          "font-size": "10", "text-anchor": "end"
        });
        label.textContent = score;
        svg.append(line, label);
      });

      const points = data.chart.map(point => `${x(point.action)},${y(point.score)}`).join(" ");
      svg.append(svgNode("polyline", {
        points, fill: "none", stroke: "#8b5cf6", "stroke-width": "3",
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
      data.milestones.forEach(item => {
        const dot = svgNode("circle", {
          cx: x(item.action), cy: y(item.score), r: "4",
          fill: "#34d399", stroke: "#090d14", "stroke-width": "2",
          tabindex: "0", role: "button"
        });
        dot.addEventListener("click", () => selectCall(item.call));
        const title = svgNode("title");
        title.textContent = `Action ${item.action}: +${item.delta} via ${item.command}`;
        dot.append(title);
        svg.append(dot);
      });
      const axis = svgNode("text", {
        x: width - right, y: height - 7, fill: "#93a2b8",
        "font-size": "10", "text-anchor": "end"
      });
      axis.textContent = `${maxAction} game actions`;
      svg.append(axis);

      const milestones = document.getElementById("milestones");
      data.milestones.forEach(item => {
        const button = el(
          "button", "milestone",
          `+${item.delta} → ${item.score} · #${item.action} ${item.command}`
        );
        button.title = item.observation;
        button.addEventListener("click", () => selectCall(item.call));
        milestones.append(button);
      });
    }

    function searchable(call) {
      return [
        call.title, call.reasoning, call.code, call.repl_output,
        ...call.actions.flatMap(action => [action.command, action.observation])
      ].join("\n").toLowerCase();
    }

    function filterCalls() {
      const query = document.getElementById("search").value.trim().toLowerCase();
      const scoreOnly = document.getElementById("score-filter").checked;
      const actionsOnly = document.getElementById("action-filter").checked;
      const retriesOnly = document.getElementById("retry-filter").checked;
      state.visible = data.calls
        .filter(call => !query || searchable(call).includes(query))
        .filter(call => !scoreOnly || call.score_delta > 0)
        .filter(call => !actionsOnly || call.action_count > 0)
        .filter(call => !retriesOnly || call.retry_errors.length > 0)
        .map(call => call.number);
      if (!state.visible.includes(state.selected) && state.visible.length) {
        state.selected = state.visible[0];
      }
      renderCallList();
      renderSelected();
    }

    function renderCallList() {
      const list = document.getElementById("call-list");
      list.replaceChildren();
      if (!state.visible.length) {
        list.append(el("div", "empty", "No logical calls match these filters."));
        return;
      }
      state.visible.forEach(numberValue => {
        const call = data.calls.find(item => item.number === numberValue);
        const button = el(
          "button",
          `call-button${call.number === state.selected ? " active" : ""}`
        );
        const numberLine = el("div", "call-number", `CALL ${call.number} · ATTEMPT ${call.attempt}`);
        if (call.retry_errors.length) {
          numberLine.append(el("span", "retry-dot", ` · ${call.retry_errors.length} RETRY`));
        }
        const title = el("div", "call-title", call.title);
        const scoreText = call.score_delta > 0
          ? ` · +${call.score_delta} → ${call.score_after}`
          : ` · score ${call.score_after}`;
        const meta = el("div", "call-meta");
        meta.append(
          document.createTextNode(`${call.action_count} actions`),
          el("span", call.score_delta > 0 ? "score-up" : "", scoreText)
        );
        button.append(numberLine, title, meta);
        button.addEventListener("click", () => selectCall(call.number));
        list.append(button);
      });
      list.querySelector(".call-button.active")?.scrollIntoView({ block: "nearest" });
    }

    function addTextCard(parent, titleText, content, open = true, copy = false) {
      const card = el("details", "card");
      card.open = open;
      const summary = el("summary", "", titleText);
      card.append(summary);
      if (!content) {
        card.append(el("div", "empty", "No content recorded."));
      } else {
        if (copy) {
          const titleRow = el("div", "card-title");
          titleRow.append(el("span", "", "Executable response"));
          const copyButton = el("button", "copy-button", "Copy");
          copyButton.addEventListener("click", async event => {
            event.preventDefault();
            try {
              await navigator.clipboard.writeText(content);
              copyButton.textContent = "Copied";
              setTimeout(() => { copyButton.textContent = "Copy"; }, 1200);
            } catch {
              copyButton.textContent = "Unavailable";
            }
          });
          card.append(titleRow);
        }
        card.append(el("pre", "", content));
      }
      parent.append(card);
    }

    function renderRetries(parent, call) {
      if (!call.retry_errors.length) return;
      const box = el("div", "retry-box");
      box.append(el("h3", "", "Provider attempt failed and was retried"));
      call.retry_errors.forEach(error => {
        box.append(el(
          "div", "retry-item",
          `Attempt ${error.attempt} · ${error.error_type} · ${formatDuration(error.duration_seconds)} · ${error.error}`
        ));
      });
      parent.append(box);
    }

    function renderActions(parent, call) {
      parent.append(el("h3", "actions-heading", `Game actions (${call.action_count})`));
      if (!call.actions.length) {
        parent.append(el("div", "empty", "This logical call executed no game actions."));
        return;
      }
      call.actions.forEach(action => {
        const details = el("details", "action");
        if (action.reward !== 0 || action.run_complete) details.open = true;
        const summary = el("summary");
        summary.append(
          el("span", "action-index", `ACTION ${action.index}`),
          el("span", "command", action.command || "(no command)")
        );
        const score = el(
          "span",
          `action-score${action.reward > 0 ? " score-up" : ""}`,
          `${action.reward > 0 ? `+${action.reward} · ` : ""}${action.highscore}/${data.run.max_score}`
        );
        summary.append(score);
        details.append(summary);
        const output = action.invalid_reason
          ? `[rejected: ${action.invalid_reason}]\n${action.observation}`
          : action.observation;
        details.append(el("pre", "action-observation", output));
        parent.append(details);
      });
    }

    function renderSelected() {
      const main = document.getElementById("main");
      main.replaceChildren();
      const call = data.calls.find(item => item.number === state.selected);
      if (!call) {
        main.append(el("div", "empty", "Select a logical call."));
        return;
      }
      const visibleIndex = state.visible.indexOf(call.number);
      const header = el("div", "call-header");
      const heading = el("div");
      heading.append(
        el("div", "eyebrow", `Logical call ${call.number} · provider attempt ${call.attempt}`),
        el("h2", "", call.title)
      );
      const badges = el("div", "badges");
      badges.append(
        el("span", "badge", `${call.action_count} game actions`),
        el("span", call.score_delta > 0 ? "badge good" : "badge", `score ${call.score_before} → ${call.score_after}`),
        el("span", "badge", `${call.provider_calls} provider calls`)
      );
      if (call.response_transformed) {
        badges.append(el("span", "badge warn", `safe ${call.response_transform_reason} recovery`));
      }
      heading.append(badges);
      const navigation = el("div", "nav-buttons");
      const previous = el("button", "", "← Prev");
      const next = el("button", "", "Next →");
      previous.disabled = visibleIndex <= 0;
      next.disabled = visibleIndex < 0 || visibleIndex >= state.visible.length - 1;
      previous.addEventListener("click", () => selectCall(state.visible[visibleIndex - 1]));
      next.addEventListener("click", () => selectCall(state.visible[visibleIndex + 1]));
      navigation.append(previous, next);
      header.append(heading, navigation);
      main.append(header);

      const stats = el("div", "call-stats");
      [
        ["Wall time", formatDuration(call.duration_seconds)],
        ["Actions", call.action_count],
        ["Provider calls", call.provider_calls],
        ["Prompt tokens", formatCount(call.prompt_tokens)],
        ["Completion tokens", formatCount(call.completion_tokens)],
        ["Reasoning tokens", formatCount(call.reasoning_tokens)],
        ["Cached tokens", formatCount(call.cached_tokens)],
        ["AI credits", Number(call.ai_credits).toFixed(3)],
      ].forEach(([label, value]) => stats.append(miniStat(label, value)));
      main.append(stats);
      renderRetries(main, call);

      const columns = el("div", "two-column");
      addTextCard(columns, "Reasoning summary", call.reasoning, true, false);
      addTextCard(columns, "Generated Python", call.code, true, true);
      main.append(columns);
      addTextCard(main, "REPL output", call.repl_output, call.action_count > 0, false);
      renderActions(main, call);

      const footer = el("div", "footer");
      footer.append(el(
        "div", "",
        `${data.trace.schema_version || "ATIF"} · trajectory ${data.trace.trajectory_id || "unknown"} · generated ${data.generated_at}`
      ));
      const links = el("div", "raw-links");
      Object.entries(data.artifact_links).forEach(([name, href]) => {
        const anchor = el("a", "", name);
        anchor.href = href;
        anchor.target = "_blank";
        links.append(anchor);
      });
      footer.append(links);
      main.append(footer);
      history.replaceState(null, "", `#call-${call.number}`);
    }

    function selectCall(callNumber) {
      if (!data.calls.some(call => call.number === callNumber)) return;
      state.selected = callNumber;
      if (!state.visible.includes(callNumber)) {
        document.getElementById("search").value = "";
        document.getElementById("score-filter").checked = false;
        document.getElementById("action-filter").checked = false;
        document.getElementById("retry-filter").checked = false;
        state.visible = data.calls.map(call => call.number);
      }
      renderCallList();
      renderSelected();
      document.getElementById("main").scrollIntoView({ block: "start" });
    }

    function initialize() {
      renderHeader();
      renderChart();
      const hashCall = Number((location.hash.match(/^#call-(\d+)$/) || [])[1]);
      if (data.calls.some(call => call.number === hashCall)) state.selected = hashCall;
      document.getElementById("search").addEventListener("input", filterCalls);
      ["score-filter", "action-filter", "retry-filter"].forEach(id => {
        document.getElementById(id).addEventListener("change", filterCalls);
      });
      document.addEventListener("keydown", event => {
        const tag = document.activeElement?.tagName;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
          event.preventDefault();
          document.getElementById("search").focus();
          return;
        }
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        const index = state.visible.indexOf(state.selected);
        if (event.key === "ArrowLeft" && index > 0) selectCall(state.visible[index - 1]);
        if (event.key === "ArrowRight" && index < state.visible.length - 1) selectCall(state.visible[index + 1]);
      });
      renderCallList();
      renderSelected();
    }

    initialize();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
