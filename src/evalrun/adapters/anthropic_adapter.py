import json
import time

import anthropic

from .base import GenerationResult, ModelAdapter

_RETRY_DELAYS = (1.0, 2.0, 4.0)
_RETRYABLE = (anthropic.APIError, anthropic.APITimeoutError, anthropic.RateLimitError)
_DEFAULT_API_TIMEOUT = 120  # seconds per API call
_DEFAULT_CASE_TIMEOUT = 120  # seconds total wallclock per generation
_DEFAULT_MAX_TOKENS = 1024  # Anthropic requires max_tokens

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code and return stdout/stderr. "
        "Use this to write and run code that helps you solve the task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    },
}


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        self._model = model
        kwargs: dict = {"timeout": _DEFAULT_API_TIMEOUT}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)

    def _call_with_retries(self, call_params: dict) -> tuple:
        """Make an API call with retries. Returns (response, elapsed_ms)."""
        last_err: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                start = time.perf_counter()
                response = self._client.messages.create(**call_params)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return response, elapsed_ms
            except anthropic.BadRequestError:
                raise  # Don't retry client errors (e.g. safety blocks)
            except _RETRYABLE as exc:
                last_err = exc
                if attempt < len(_RETRY_DELAYS) - 1:
                    time.sleep(delay)

        raise RuntimeError(
            f"Anthropic call failed after {len(_RETRY_DELAYS)} retries"
        ) from last_err

    def _build_call_params(self, messages: list[dict], system: str = "", **params) -> dict:
        call_params: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": params.get("max_tokens", _DEFAULT_MAX_TOKENS),
        }
        if system:
            call_params["system"] = system
        if "temperature" in params:
            call_params["temperature"] = params["temperature"]
        return call_params

    def _extract_text(self, response) -> str:
        """Extract text from response content blocks."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _generate_single(self, prompt: str, system: str = "", **params) -> GenerationResult:
        messages: list[dict] = [{"role": "user", "content": prompt}]

        call_params = self._build_call_params(messages, system, **params)

        try:
            response, elapsed_ms = self._call_with_retries(call_params)
        except anthropic.BadRequestError as exc:
            # Distinguish billing/auth errors from safety blocks
            msg = str(exc).lower()
            if "credit" in msg or "balance" in msg or "billing" in msg:
                raise RuntimeError(f"Anthropic billing error: {exc}") from exc
            return GenerationResult(
                text="SAFETY_BLOCKED",
                latency_ms=0.0,
                tokens_in=None,
                tokens_out=None,
            )

        text = self._extract_text(response)
        usage = response.usage

        return GenerationResult(
            text=text,
            latency_ms=elapsed_ms,
            tokens_in=usage.input_tokens if usage else None,
            tokens_out=usage.output_tokens if usage else None,
        )

    def _generate_with_tools(self, prompt: str, system: str = "", **params) -> GenerationResult:
        from ..tools import run_python

        max_turns = params.pop("max_tool_turns", 10)
        case_timeout = params.pop("case_timeout", _DEFAULT_CASE_TIMEOUT)
        deadline = time.perf_counter() + case_timeout

        messages: list[dict] = [{"role": "user", "content": prompt}]

        total_tokens_in = 0
        total_tokens_out = 0
        total_latency_ms = 0.0
        tool_calls_made = 0
        last_text = ""

        for _ in range(max_turns):
            # Check wallclock budget before each API call
            if time.perf_counter() >= deadline:
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made, "timed_out": True},
                )

            call_params = self._build_call_params(messages, system, **params)
            call_params["tools"] = [RUN_PYTHON_TOOL]

            try:
                response, elapsed_ms = self._call_with_retries(call_params)
            except anthropic.BadRequestError as exc:
                # Distinguish billing/auth errors from safety blocks
                msg = str(exc).lower()
                if "credit" in msg or "balance" in msg or "billing" in msg:
                    raise RuntimeError(f"Anthropic billing error: {exc}") from exc
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made, "error": True} if tool_calls_made > 0 else None,
                )
            except RuntimeError:
                raise  # Propagate retry exhaustion and billing errors
            total_latency_ms += elapsed_ms

            usage = response.usage
            if usage:
                total_tokens_in += usage.input_tokens or 0
                total_tokens_out += usage.output_tokens or 0

            # Extract text from response content blocks
            last_text = self._extract_text(response) or last_text

            # If the model ended its turn without requesting tools, we're done
            if response.stop_reason == "end_turn":
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made} if tool_calls_made > 0 else None,
                )

            # Collect tool_use blocks from the response
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                # No tool calls and not end_turn — return what we have
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made} if tool_calls_made > 0 else None,
                )

            # Append the full assistant message with all content blocks
            # Convert content blocks to dicts for the messages list
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call and build tool_result blocks
            tool_results = []
            for block in tool_use_blocks:
                tool_calls_made += 1
                if block.name == "run_python":
                    try:
                        code = block.input.get("code", "")
                        result = run_python(code)
                    except Exception as e:
                        result = {"stdout": "", "stderr": str(e), "exit_code": -1}
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        # Max turns exhausted — return whatever we have
        return GenerationResult(
            text=last_text,
            latency_ms=total_latency_ms,
            tokens_in=total_tokens_in or None,
            tokens_out=total_tokens_out or None,
            tool_meta={"tool_calls": tool_calls_made, "max_turns_reached": True},
        )

    def generate(self, prompt: str, system: str = "", **params) -> GenerationResult:
        tools_enabled = params.pop("tools_enabled", False)
        if tools_enabled:
            return self._generate_with_tools(prompt, system, **params)
        return self._generate_single(prompt, system, **params)

    @property
    def model_id(self) -> str:
        return f"anthropic:{self._model}"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"
