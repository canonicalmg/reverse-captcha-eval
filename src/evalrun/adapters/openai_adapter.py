import json
import time

from openai import OpenAI, APIError, APITimeoutError, BadRequestError, RateLimitError

from .base import GenerationResult, ModelAdapter

_RETRY_DELAYS = (1.0, 2.0, 4.0)
_RETRYABLE = (APIError, APITimeoutError, RateLimitError)
_DEFAULT_API_TIMEOUT = 120  # seconds per API call
_DEFAULT_CASE_TIMEOUT = 120  # seconds total wallclock per generation

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code and return stdout/stderr. Use this to write and run code that helps you solve the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        },
    },
}


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model
        self._base_url = base_url
        kwargs: dict = {"timeout": _DEFAULT_API_TIMEOUT}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _call_with_retries(self, call_params: dict) -> tuple:
        """Make an API call with retries. Returns (response, elapsed_ms)."""
        last_err: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**call_params)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return response, elapsed_ms
            except _RETRYABLE as exc:
                last_err = exc
                if attempt < len(_RETRY_DELAYS) - 1:
                    time.sleep(delay)

        raise RuntimeError(
            f"OpenAI call failed after {len(_RETRY_DELAYS)} retries"
        ) from last_err

    def _build_call_params(self, messages: list[dict], **params) -> dict:
        call_params: dict = {
            "model": self._model,
            "messages": messages,
        }
        if "temperature" in params:
            call_params["temperature"] = params["temperature"]
        if "max_tokens" in params:
            call_params["max_tokens"] = params["max_tokens"]
        return call_params

    def _generate_single(self, prompt: str, system: str = "", **params) -> GenerationResult:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_params = self._build_call_params(messages, **params)
        response, elapsed_ms = self._call_with_retries(call_params)

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            text=choice.message.content or "",
            latency_ms=elapsed_ms,
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
        )

    def _generate_with_tools(self, prompt: str, system: str = "", **params) -> GenerationResult:
        from ..tools import run_python

        max_turns = params.pop("max_tool_turns", 10)
        case_timeout = params.pop("case_timeout", _DEFAULT_CASE_TIMEOUT)
        deadline = time.perf_counter() + case_timeout

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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

            call_params = self._build_call_params(messages, **params)
            call_params["tools"] = [RUN_PYTHON_TOOL]

            try:
                response, elapsed_ms = self._call_with_retries(call_params)
            except (RuntimeError, BadRequestError):
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made, "error": True} if tool_calls_made > 0 else None,
                )
            total_latency_ms += elapsed_ms

            usage = response.usage
            if usage:
                total_tokens_in += usage.prompt_tokens or 0
                total_tokens_out += usage.completion_tokens or 0

            choice = response.choices[0]
            message = choice.message
            last_text = message.content or ""

            # If the model gave a text response (no tool calls), we're done
            if choice.finish_reason == "stop" or not message.tool_calls:
                return GenerationResult(
                    text=last_text,
                    latency_ms=total_latency_ms,
                    tokens_in=total_tokens_in or None,
                    tokens_out=total_tokens_out or None,
                    tool_meta={"tool_calls": tool_calls_made} if tool_calls_made > 0 else None,
                )

            # Append assistant message with tool calls
            messages.append(message.to_dict() if hasattr(message, "to_dict") else {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool call
            for tc in message.tool_calls:
                tool_calls_made += 1
                if tc.function.name == "run_python":
                    try:
                        args = json.loads(tc.function.arguments)
                        result = run_python(args.get("code", ""))
                    except (json.JSONDecodeError, Exception) as e:
                        result = {"stdout": "", "stderr": str(e), "exit_code": -1}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {tc.function.name}"}),
                    })

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
        prefix = "ollama" if self._base_url and "11434" in self._base_url else "openai"
        return f"{prefix}:{self._model}"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama" if self._base_url and "11434" in self._base_url else "openai"
