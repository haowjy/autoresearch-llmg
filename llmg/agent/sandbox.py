"""Sandboxed shell for agentic search (allowlisted commands only)."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from llmg.agent.trace import TRACE_SHELL_STDOUT_MAX, truncate_text, write_trace

log = logging.getLogger(__name__)

ALLOWED_BINARIES = frozenset(
    {"rg", "grep", "find", "head", "cat", "wc", "ls", "sqlite3", "echo", "pwd"}
)
DEFAULT_MAX_BYTES = 256_000
DEFAULT_TIMEOUT_S = 30.0


@dataclass
class SandboxStats:
    cmd_count: int = 0
    bytes_read: int = 0


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed_s: float


class AgentSandbox:
    def __init__(
        self,
        workspace: Path,
        *,
        trace_path: Path | None = None,
        max_output_bytes: int = DEFAULT_MAX_BYTES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_path
        self.max_output_bytes = max_output_bytes
        self.timeout_s = timeout_s
        self.stats = SandboxStats()

    def close(self) -> None:
        return

    def _log_trace(self, **fields) -> None:
        if self.trace_path is None:
            return
        stdout = fields.pop("stdout", "") or ""
        stderr = fields.pop("stderr", "") or ""
        write_trace(
            self.trace_path,
            "sandbox",
            stdout=truncate_text(stdout, TRACE_SHELL_STDOUT_MAX) if stdout else "",
            stderr=truncate_text(stderr, TRACE_SHELL_STDOUT_MAX) if stderr else "",
            **fields,
        )

    def _validate_command(self, cmd: str) -> list[str]:
        cmd = cmd.strip()
        if not cmd:
            raise ValueError("empty command")
        if ";" in cmd or "&&" in cmd or "||" in cmd or "|" in cmd or "`" in cmd:
            raise ValueError("chained or piped commands not allowed")
        parts = shlex.split(cmd)
        if not parts:
            raise ValueError("empty command")
        binary = Path(parts[0]).name
        if binary not in ALLOWED_BINARIES:
            raise ValueError(f"command {binary!r} not in allowlist")
        return parts

    def run(self, cmd: str) -> CommandResult:
        parts = self._validate_command(cmd)
        self.stats.cmd_count += 1
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                parts,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            result = CommandResult(
                stdout="",
                stderr=f"executable not found: {exc}",
                returncode=127,
                elapsed_s=time.monotonic() - t0,
            )
            self._log_trace(cmd=cmd, **result.__dict__)
            return result
        except subprocess.TimeoutExpired:
            result = CommandResult(
                stdout="",
                stderr=f"timeout after {self.timeout_s}s",
                returncode=-1,
                elapsed_s=time.monotonic() - t0,
            )
            self._log_trace(cmd=cmd, **result.__dict__)
            return result

        out = (proc.stdout or "")[: self.max_output_bytes]
        err = (proc.stderr or "")[: self.max_output_bytes]
        self.stats.bytes_read += len(out) + len(err)
        result = CommandResult(
            stdout=out,
            stderr=err,
            returncode=proc.returncode,
            elapsed_s=time.monotonic() - t0,
        )
        self._log_trace(cmd=cmd, **result.__dict__)
        return result
