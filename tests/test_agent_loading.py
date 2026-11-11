import subprocess
import sys
import textwrap
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark.py"


def _write_agent(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(f"""
            import pathlib

            import tales

            pathlib.Path({str(marker)!r}).write_text("executed")


            class CustomAgent(tales.Agent):
                pass


            tales.agent.register(
                name="custom-agent",
                desc="Custom agent.",
                klass=CustomAgent,
                add_arguments=lambda parser: parser,
            )
            """))


def _run_benchmark(cwd, *args):
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_startup_and_help_do_not_execute_cwd_agents(tmp_path):
    marker = tmp_path / "executed.txt"
    _write_agent(tmp_path / "agents" / "evil.py", marker)

    help_result = _run_benchmark(tmp_path, "--help")
    startup_result = _run_benchmark(tmp_path)

    assert help_result.returncode == 0, help_result.stderr
    assert not marker.exists()
    assert "custom-agent" not in help_result.stdout

    # Without a subcommand, only the bundled agents are listed.
    assert startup_result.returncode == 1, startup_result.stderr
    assert not marker.exists()
    assert "reasoning" in startup_result.stdout
    assert "custom-agent" not in startup_result.stdout


def test_explicit_agent_path_loads_custom_agent(tmp_path):
    marker = tmp_path / "executed.txt"
    agent_file = tmp_path / "my_agents" / "custom.py"
    _write_agent(agent_file, marker)

    result = _run_benchmark(tmp_path, "--agent", str(agent_file))

    assert marker.exists()
    assert "custom-agent" in result.stdout
    assert "reasoning" in result.stdout


def test_explicit_agent_pattern_without_match_fails(tmp_path):
    result = _run_benchmark(tmp_path, "--agent", "missing/*.py")

    assert result.returncode != 0
    assert "did not match any file" in result.stderr
