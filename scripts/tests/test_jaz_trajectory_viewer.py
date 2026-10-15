import json
from pathlib import Path

from build_jaz_trajectory_viewer import build_viewer, build_viewer_payload


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": "test-run",
            "status": "completed_max_score",
            "score_valid": True,
            "model": "test-model",
            "reasoning_effort": "max",
            "duration_seconds": 12.5,
            "config": {"game": "TestGame", "seed": 7, "max_actions": None},
            "game_result": {
                "game": "TestGame",
                "seed": 7,
                "score": 10,
                "max_score": 10,
                "moves": 1,
                "actions_used": 1,
                "completion_reason": "max_score",
                "initial_observation": "Start",
                "final_observation": "Won",
            },
            "model_usage": {
                "completion_attempts": 3,
                "completed_calls": 2,
                "provider_api_calls": 4,
                "reasoning_tokens": 100,
            },
        },
    )
    _write_jsonl(
        run_dir / "model_calls.jsonl",
        [
            {
                "attempt": 1,
                "call": 1,
                "status": "ok",
                "timestamp": 1,
                "duration": 2,
                "response": "# Plan: inspect\nprint(observe())",
                "reasoning_summary": "Inspect first.",
                "reasoning_is_summary": True,
                "totals": {"api_calls": 1, "reasoning_tokens": 40},
            },
            {
                "attempt": 2,
                "status": "error",
                "duration": 3,
                "error_type": "BridgeError",
                "error": "timed out",
            },
            {
                "attempt": 3,
                "call": 2,
                "status": "ok",
                "timestamp": 6,
                "duration": 4,
                "response": "# Win safely\nprint(step('win'))\n# </script>",
                "reasoning_summary": "Use the winning action.",
                "reasoning_is_summary": True,
                "totals": {"api_calls": 3, "reasoning_tokens": 60},
            },
        ],
    )
    _write_jsonl(
        run_dir / "actions.jsonl",
        [
            {
                "action_index": 1,
                "model_calls_completed": 2,
                "submitted_action": "win",
                "executed_action": "win",
                "action_was_executed": True,
                "observation": "Victory",
                "reward": 10,
                "score": 10,
                "highscore": 10,
                "moves": 1,
                "run_complete": True,
                "completion_reason": "max_score",
            }
        ],
    )
    _write_json(
        run_dir / "trace.atif.json",
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "trajectory-1",
            "agent": {"model_name": "test-model"},
            "steps": [
                {
                    "source": "user",
                    "message": "Observed start",
                    "extra": {"iteration": 0},
                },
                {
                    "source": "user",
                    "message": "Observed victory",
                    "extra": {"iteration": 1},
                },
            ],
        },
    )
    return run_dir


def test_build_viewer_payload_joins_calls_retries_actions_and_output(tmp_path):
    run_dir = _fixture_run(tmp_path)

    payload = build_viewer_payload(run_dir)

    assert payload["run"]["score"] == 10
    assert payload["run"]["action_limit"] is None
    assert payload["run"]["provider_retry_count"] == 1
    assert payload["calls"][0]["repl_output"] == "Observed start"
    assert payload["calls"][1]["repl_output"] == "Observed victory"
    assert payload["calls"][1]["retry_errors"][0]["error"] == "timed out"
    assert payload["calls"][1]["actions"][0]["command"] == "win"
    assert payload["calls"][1]["score_delta"] == 10
    assert payload["milestones"] == [
        {
            "action": 1,
            "call": 2,
            "score": 10,
            "delta": 10,
            "command": "win",
            "observation": "Victory",
        }
    ]


def test_build_viewer_writes_self_contained_safe_html(tmp_path):
    run_dir = _fixture_run(tmp_path)

    output = build_viewer(run_dir)
    document = output.read_text(encoding="utf-8")

    assert output == run_dir / "viewer.html"
    assert "__TRAJECTORY_DATA__" not in document
    assert "\\u003c/script\\u003e" in document
    assert '"model":"test-model"' in document
    assert "https://" not in document
