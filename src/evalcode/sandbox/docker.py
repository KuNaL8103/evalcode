"""Docker sandbox backend with an automatic subprocess fallback.

The generated ``solution.py`` + tests are written to a temporary directory,
bind-mounted into a locked-down container, and executed with:

    docker run --rm --network none --memory=256m --cpus=0.5 ...

The container gets no network, a read-only root filesystem, a writable
``/work`` mount, a small ``/tmp`` tmpfs, and dropped capabilities. Results are
normalized through the same :func:`evalcode.sandbox.runner.finalize_result`
routine as the subprocess backend, so downstream nodes cannot tell the
difference.

If the Docker daemon is not running, the ``docker`` CLI is missing, or the
image cannot be started (exit code 125), the runner logs a warning and calls
the existing subprocess sandbox instead. A Docker outage never fails a task.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evalcode.core.config import Settings
from evalcode.sandbox.errors import C_TIMEOUT, truncate_output
from evalcode.sandbox.runner import finalize_result, new_result, preflight, run_in_sandbox

__all__ = ["DEFAULT_DOCKER_IMAGE", "DockerSandbox", "build_sandbox"]

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_IMAGE = "evalcode-sandbox:latest"
_CONTAINER_WORKDIR = "/work"
# Docker's own CLI/runtime startup is addded to the pytest wall-clock budget.
_CONTAINER_STARTUP_GRACE_S = 5.0


class DockerUnavailable(RuntimeError):
    """Docker could not start the container (daemon/image/CLI problem)."""


class DockerSandbox:
    """Run pytest inside a throwaway Docker container.

    Callable with the same signature as :func:`run_in_sandbox`, so it can be
    injected as ``Dependencies.sandbox`` without changing the graph.
    """

    def __init__(
        self,
        image: str = DEFAULT_DOCKER_IMAGE,
        *,
        fallback: Callable[..., dict] = run_in_sandbox,
        memory_mb: int = 256,
        cpus: float = 0.5,
        pids_limit: int = 128,
        docker_bin: str = "docker",
        startup_grace_s: float = _CONTAINER_STARTUP_GRACE_S,
        probe: Callable[[], bool] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.image = image
        self.fallback = fallback
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker_bin = docker_bin
        self.startup_grace_s = startup_grace_s
        self._probe = probe
        self._command_runner = command_runner
        self._available: bool | None = None

    # --- availability ------------------------------------------------------

    def is_available(self, *, refresh: bool = False) -> bool:
        """True when the Docker daemon answers; cached after the first probe."""
        if self._available is not None and not refresh:
            return self._available
        if self._probe is not None:
            available = bool(self._probe())
        else:
            try:
                completed = self._command_runner(
                    [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                available = completed.returncode == 0
            except (OSError, subprocess.SubprocessError):
                available = False
        if not available:
            logger.warning(
                "Docker daemon unavailable (%s); falling back to the subprocess sandbox",
                self.docker_bin,
            )
        self._available = available
        return available

    def _default_fallback(self, code: str, tests: str, *, timeout_s: float, mem_mb: int) -> dict:
        return self.fallback(code, tests, timeout_s=timeout_s, mem_mb=mem_mb)

    # --- public runner -----------------------------------------------------

    def __call__(
        self, code: str, tests: str, *, timeout_s: float = 30.0, mem_mb: int = 128
    ) -> dict:
        return self.run(code, tests, timeout_s=timeout_s, mem_mb=mem_mb)

    def run(self, code: str, tests: str, *, timeout_s: float = 30.0, mem_mb: int = 128) -> dict:
        """Run in Docker; fall back to the subprocess sandbox on infra errors."""
        if not self.is_available():
            return self._default_fallback(code, tests, timeout_s=timeout_s, mem_mb=mem_mb)
        try:
            return self._run_container(code, tests, timeout_s=timeout_s)
        except (DockerUnavailable, OSError, subprocess.SubprocessError) as exc:
            logger.warning("Docker sandbox unavailable (%s); using subprocess fallback", exc)
            return self._default_fallback(code, tests, timeout_s=timeout_s, mem_mb=mem_mb)

    def _run_container(self, code: str, tests: str, *, timeout_s: float) -> dict:
        failed_preflight = preflight(code)
        if failed_preflight is not None:
            return failed_preflight

        result = new_result()
        with tempfile.TemporaryDirectory(prefix="evalcode-docker-") as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "solution.py").write_text(code, encoding="utf-8")
            (workdir / "test_solution.py").write_text(tests + "\n", encoding="utf-8")

            name = f"evalcode-{uuid.uuid4().hex[:12]}"
            cmd = self._build_command(workdir, name)
            start = time.monotonic()
            try:
                completed = self._command_runner(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s + self.startup_grace_s,
                )
            except subprocess.TimeoutExpired:
                self._remove_container(name)
                result["duration_s"] = time.monotonic() - start
                result["category"] = C_TIMEOUT
                result["timed_out"] = True
                result["failures"] = [
                    {
                        "test_name": "timeout",
                        "error_type": "TimeoutExpired",
                        "message": f"exceeded {timeout_s}s",
                        "traceback": "",
                    }
                ]
                return result

            result["duration_s"] = time.monotonic() - start
            result["stdout"] = truncate_output(_as_text(completed.stdout))
            result["stderr"] = truncate_output(_as_text(completed.stderr))
            exit_code = int(completed.returncode or 0)

            # 125/126/127 are docker CLI/runtime errors, not test failures.
            if exit_code in (125, 126, 127):
                raise DockerUnavailable(f"docker run exited {exit_code}: {result['stderr'][:200]}")

            return finalize_result(
                result,
                exit_code=exit_code,
                report_path=workdir / "report.xml",
            )

    # --- command construction ---------------------------------------------

    def _build_command(self, workdir: Path, name: str) -> list[str]:
        """Build the hardened ``docker run`` command (pure; unit-tested)."""
        mount = f"{Path(workdir).resolve()}:{_CONTAINER_WORKDIR}:rw"
        cmd = [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            f"--memory={self.memory_mb}m",
            f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HOME=/tmp",
            "-v",
            mount,
            "-w",
            _CONTAINER_WORKDIR,
        ]
        # On POSIX, run as the host user so the bind-mounted workdir is writable.
        if os.name != "nt" and hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        cmd += [
            self.image,
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml=/work/report.xml",
            "test_solution.py",
        ]
        return cmd

    def _remove_container(self, name: str) -> None:
        """Best-effort cleanup after a client-side timeout."""
        for args in (["kill", name], ["rm", "-f", name]):
            try:
                self._command_runner([self.docker_bin, *args], capture_output=True, timeout=5)
            except Exception:
                pass


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_sandbox(settings: Settings) -> Callable[..., dict]:
    """Select the configured backend.

    ``SANDBOX_BACKEND=docker`` returns a :class:`DockerSandbox` whose fallback
    is the existing subprocess runner. The default stays subprocess so local
    behavior and the 139-test suite are unchanged unless you opt in.
    """
    if settings.sandbox_backend == "docker":
        return DockerSandbox(
            image=settings.docker_image,
            memory_mb=settings.docker_memory_mb,
            cpus=settings.docker_cpus,
            fallback=run_in_sandbox,
        )
    return run_in_sandbox
