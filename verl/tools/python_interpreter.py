"""PythonInterpreterTool – executes Python code snippets in a sandbox.

Reward:
    • stderr empty   → 0.0 (neutral)
    • stderr present → –0.1 (penalty)

Features:
    • Auto‑wrap *pure single expressions* with ``print()`` so stdout captures the value.
      – The wrapper is **not** applied if the code already contains ``print(``, an
        assignment (``=``), semicolons that split multiple statements, or newlines
        followed by an assignment.
    • Returns output wrapped in ``<tool_output>{...}</tool_output>``.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import tempfile
from textwrap import dedent
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["PythonInterpreterTool"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _maybe_wrap_print(code: str) -> str:
    """Append ``print(<expr>)`` if the snippet ends with a pure expression.

    Heuristics:
    1. If the snippet already contains ``print(``, leave it unchanged.
    2. Split the snippet into logical lines (``;`` → ``\n``).
    3. Inspect the *last non‑empty* chunk.
       • If that chunk has ``=`` (assignment) → bail (not a pure expr).
       • Else, try ``compile(chunk, '', 'eval')``. If this succeeds, it is a
         valid expression, so we append ``print(<chunk>)``.
    """
    if "print(" in code:
        return code

    # Treat semicolons like newlines for easier parsing
    code_nl = code.replace(";", "\n")
    lines = [ln.strip() for ln in code_nl.split("\n") if ln.strip()]
    if not lines:
        return code

    last = lines[-1]
    if "=" in last:
        return code

    try:
        compile(last, "<expr>", "eval")
    except SyntaxError:
        return code

    dedented = dedent(code).rstrip()
    return f"{dedented}\nprint({last})"


def _run_snippet(code: str, timeout: int = 5) -> Tuple[str, str]:
    """Execute *code* in a temporary file and return (stdout, stderr)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(dedent(code))
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class PythonInterpreterTool(BaseTool):
    """A multi‑turn tool allowing the LM to run Python snippets."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._timeout = int(config.get("timeout", 5))
        self._state: dict[str, dict[str, Any]] = {}

    # OpenAI function‑tool schema ------------------------------------------------
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:  # noqa: D401
        return self.tool_schema

    # Lifecycle methods ---------------------------------------------------------
    async def create(self, instance_id: Optional[str] = None, **_) -> str:
        iid = instance_id or str(uuid4())
        self._state[iid] = {}
        return iid

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **_,
    ) -> Tuple[str, float, dict]:
        code = str(parameters.get("code", ""))
        stdout, stderr = _run_snippet(_maybe_wrap_print(code), timeout=self._timeout)
        reward = -0.1 if stderr else 0.0
        payload = json.dumps({"stdout": stdout, "stderr": stderr})
        return f"<tool_output>{payload}</tool_output>"[:3000], reward, {}

    async def calc_reward(self, instance_id: str, **_) -> float:  # noqa: D401
        return 0.0

    async def release(self, instance_id: str, **_):
        self._state.pop(instance_id, None)
