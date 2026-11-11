from pathlib import Path

import pytest

import benchmark


class FakeEnvironment:
    def __init__(self):
        self.calls = []
        self.closed = False

    def _record(self, name):
        cwd = Path.cwd()
        self.calls.append((name, cwd))
        return cwd

    def reset(self, seed=None):
        self._record("reset")
        return "start", {"seed": seed}

    def step(self, action):
        cwd = self._record(f"step:{action}")
        save_file = cwd / "game.qzl"
        if action == "save":
            save_file.write_text("saved")
            return "saved"
        if action == "restore":
            return "restored" if save_file.exists() else "failed"
        return "ok"

    def close(self):
        self._record("close")
        self.closed = True


def test_environment_save_files_are_isolated_and_cleaned(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    created = []

    def fake_make(*args, **kwargs):
        env = FakeEnvironment()
        env.calls.append(("make", Path.cwd()))
        created.append(env)
        return env

    monkeypatch.setattr(benchmark.gym, "make", fake_make)

    with benchmark._isolated_environment("JerichoEnvZork1", False) as env:
        first_workdir = Path(env.workdir)
        assert Path.cwd() == tmp_path
        env.reset(seed=42)
        assert env.step("save") == "saved"
        assert env.step("restore") == "restored"
        assert (first_workdir / "game.qzl").exists()

    assert created[0].closed
    assert not first_workdir.exists()
    assert not (tmp_path / "game.qzl").exists()

    with benchmark._isolated_environment("JerichoEnvZork1", False) as env:
        second_workdir = Path(env.workdir)
        env.reset(seed=42)
        assert env.step("restore") == "failed"

    assert first_workdir != second_workdir
    assert not second_workdir.exists()
    for fake_env in created:
        assert all(cwd != tmp_path for _, cwd in fake_env.calls)


def test_environment_closes_and_cleans_up_after_exception(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake_env = FakeEnvironment()
    monkeypatch.setattr(benchmark.gym, "make", lambda *args, **kwargs: fake_env)

    with pytest.raises(RuntimeError, match="boom"):
        with benchmark._isolated_environment("JerichoEnvZork1", False) as env:
            workdir = Path(env.workdir)
            env.reset()
            raise RuntimeError("boom")

    assert fake_env.closed
    assert not workdir.exists()
    assert Path.cwd() == tmp_path


def test_evaluation_paths_are_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = benchmark._evaluation_paths(
        "relative-logs", "JerichoEnvZork1", "a0_sNone_steps2"
    )

    assert all(Path(path).is_absolute() for path in paths)
    assert Path(paths[0]) == tmp_path / "relative-logs" / "JerichoEnvZork1"
