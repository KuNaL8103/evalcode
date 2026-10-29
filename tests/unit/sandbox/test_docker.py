"""Docker sandbox: command hardening, JUnit parsing, and safe fallback."""

from __future__ import annotations

import subprocess
from pathlib import Path

from evalcode.core.config import Settings
from evalcode.sandbox.docker import DEFAULT_DOCKER_IMAGE, DockerSandbox, build_sandbox
from evalcode.sandbox.runner import run_in_sandbox

PASS_RESULT = {
    "passed": True,
    "category": "pass",
    "exit_code": 0,
    "timed_out": False,
    "duration_s": 0.01,
    "tests_total": 1,
    "tests_failed": 0,
    "failures": [],
    "stdout": "",
    "stderr": "",
}


def test_build_command_has_isolation_flags(tmp_path: Path) -> None:
    sandbox = DockerSandbox(image="img:test", memory_mb=256, cpus=0.5)
    cmd = sandbox._build_command(tmp_path, "evalcode-test")
    joined = " ".join(cmd)

    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "evalcode-test" in cmd
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--memory=256m" in cmd
    assert "--cpus=0.5" in cmd
    assert "--pids-limit=128" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--tmpfs=/tmp:rw,noexec,nosuid,size=64m" in cmd
    assert f"{tmp_path.resolve()}:/work:rw" in joined
    assert "-w" in cmd and cmd[cmd.index("-w") + 1] == "/work"
    assert cmd[-9:] == [
        "img:test",
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml=/work/report.xml",
        "test_solution.py",
    ]


def test_falls_back_when_daemon_unavailable() -> None:
    calls: list[tuple[str, str, float, int]] = []

    def fallback(code: str, tests: str, *, timeout_s: float, mem_mb: int) -> dict:
        calls.append((code, tests, timeout_s, mem_mb))
        return dict(PASS_RESULT)

    sandbox = DockerSandbox(fallback=fallback, probe=lambda: False)
    result = sandbox.run("def f(): pass", "def test_f(): pass", timeout_s=3.0, mem_mb=64)

    assert result["passed"] is True
    assert calls == [("def f(): pass", "def test_f(): pass", 3.0, 64)]
    assert sandbox._available is False


def test_falls_back_on_missing_docker_binary() -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("docker not installed")

    calls: list[str] = []

    def fallback(code: str, tests: str, *, timeout_s: float, mem_mb: int) -> dict:
        calls.append(code)
        return dict(PASS_RESULT)

    sandbox = DockerSandbox(
        fallback=fallback,
        probe=lambda: True,
        command_runner=missing,
    )
    result = sandbox.run("def f(): pass", "def test_f(): pass", timeout_s=3.0, mem_mb=64)
    assert result["passed"] is True
    assert calls == ["def f(): pass"]


def test_falls_back_on_docker_infra_exit_code() -> None:
    def runner(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 125, stdout="", stderr="No such image")

    calls: list[str] = []

    def fallback(code: str, tests: str, *, timeout_s: float, mem_mb: int) -> dict:
        calls.append(code)
        return dict(PASS_RESULT)

    sandbox = DockerSandbox(fallback=fallback, probe=lambda: True, command_runner=runner)
    result = sandbox.run("def f(): pass", "def test_f(): pass", timeout_s=3.0, mem_mb=64)
    assert result["passed"] is True
    assert calls == ["def f(): pass"]


def test_successful_container_run_parses_junit() -> None:
    def runner(cmd, **_kwargs):
        mount = cmd[cmd.index("-v") + 1]
        workdir = Path(mount.rsplit(":/work:rw", 1)[0])
        (workdir / "report.xml").write_text(
            '<testsuite tests="1" failures="0" errors="0">'
            '<testcase name="test_f" time="0.01"/></testsuite>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")

    sandbox = DockerSandbox(probe=lambda: True, command_runner=runner)
    result = sandbox.run(
        "def f(): pass",
        "def test_f(): pass",
        timeout_s=3.0,
        mem_mb=64,
    )

    assert result["category"] == "pass"
    assert result["passed"] is True
    assert result["tests_total"] == 1
    assert result["exit_code"] == 0
    assert "1 passed" in result["stdout"]


def test_timeout_returns_timeout_category() -> None:
    def runner(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1)

    sandbox = DockerSandbox(probe=lambda: True, command_runner=runner)
    result = sandbox.run("def f(): pass", "def test_f(): pass", timeout_s=1.0, mem_mb=64)

    assert result["category"] == "timeout"
    assert result["passed"] is False
    assert result["timed_out"] is True


def test_preflight_still_short_circuits_before_docker() -> None:
    def runner(cmd, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("docker should not be invoked for a SyntaxError")

    sandbox = DockerSandbox(probe=lambda: True, command_runner=runner)
    result = sandbox.run("def broken(", "def test_x(): pass", timeout_s=1.0, mem_mb=64)
    assert result["category"] == "syntax_error"


def test_build_sandbox_selection() -> None:
    assert build_sandbox(Settings()) is run_in_sandbox

    docker = build_sandbox(Settings(sandbox_backend="docker"))
    assert isinstance(docker, DockerSandbox)
    assert docker.image == DEFAULT_DOCKER_IMAGE
    assert docker.memory_mb == 256
    assert docker.cpus == 0.5


def test_dockerfile_exists_and_uses_slim_python() -> None:
    dockerfile = Path(__file__).resolve().parents[3] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert text.startswith("# evalcode sandbox image")
    assert "FROM python:3.12-slim" in text
    assert "USER sandbox" in text
    assert "pytest" in text
