"""Tests for the Anthropic model adapter."""

import time
from unittest.mock import MagicMock, patch

import pytest

import anthropic

from evalrun.adapters.anthropic_adapter import AnthropicAdapter, _RETRY_DELAYS


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, input_data: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(content_blocks, stop_reason="end_turn", input_tokens=10, output_tokens=20):
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnthropicAdapterProperties:
    def test_model_id(self):
        with patch("anthropic.Anthropic"):
            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        assert adapter.model_id == "anthropic:claude-sonnet-4-20250514"

    def test_model_name(self):
        with patch("anthropic.Anthropic"):
            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        assert adapter.model_name == "claude-sonnet-4-20250514"

    def test_provider(self):
        with patch("anthropic.Anthropic"):
            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        assert adapter.provider == "anthropic"

    def test_default_model(self):
        with patch("anthropic.Anthropic"):
            adapter = AnthropicAdapter()
        assert adapter.model_name == "claude-sonnet-4-20250514"


class TestGenerateSingle:
    def test_basic_generation(self):
        """Basic single-message generation returns correct result."""
        mock_response = _make_response(
            [_make_text_block("Hello, world!")],
            stop_reason="end_turn",
            input_tokens=5,
            output_tokens=3,
        )

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = mock_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            result = adapter.generate("Say hello")

        assert result.text == "Hello, world!"
        assert result.tokens_in == 5
        assert result.tokens_out == 3
        assert result.latency_ms > 0
        assert result.tool_meta is None

    def test_system_prompt_passed_as_top_level_param(self):
        """System prompt is sent as a top-level param, not as a message."""
        mock_response = _make_response([_make_text_block("I am helpful.")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = mock_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            adapter.generate("Who are you?", system="You are a helpful assistant.")

        call_kwargs = client_instance.messages.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        # System should be a top-level kwarg, not in messages
        assert all_kwargs.get("system") == "You are a helpful assistant."
        messages = all_kwargs.get("messages")
        # Messages should only contain the user message, not a system message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_empty_system_not_included(self):
        """When system is empty string, system param should not be set."""
        mock_response = _make_response([_make_text_block("response")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = mock_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            adapter.generate("test prompt")

        call_kwargs = client_instance.messages.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "system" not in all_kwargs

    def test_temperature_and_max_tokens_forwarded(self):
        """Extra params like temperature and max_tokens are forwarded."""
        mock_response = _make_response([_make_text_block("ok")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = mock_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            adapter.generate("test", temperature=0.5, max_tokens=2048)

        call_kwargs = client_instance.messages.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs["temperature"] == 0.5
        assert all_kwargs["max_tokens"] == 2048

    def test_default_max_tokens(self):
        """Default max_tokens is 1024 when not specified."""
        mock_response = _make_response([_make_text_block("ok")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = mock_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            adapter.generate("test")

        call_kwargs = client_instance.messages.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs["max_tokens"] == 1024


class TestRetryLogic:
    def test_retries_on_api_error(self):
        """Retries on APIError and succeeds on final attempt."""
        mock_response = _make_response([_make_text_block("recovered")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = [
                anthropic.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                anthropic.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                mock_response,
            ]

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            with patch("time.sleep"):  # Skip actual sleep
                result = adapter.generate("test")

        assert result.text == "recovered"
        assert client_instance.messages.create.call_count == 3

    def test_retries_on_rate_limit(self):
        """Retries on RateLimitError."""
        mock_response = _make_response([_make_text_block("ok")])

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = [
                anthropic.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                mock_response,
            ]

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            with patch("time.sleep"):
                result = adapter.generate("test")

        assert result.text == "ok"
        assert client_instance.messages.create.call_count == 2

    def test_raises_after_all_retries_exhausted(self):
        """Raises RuntimeError after all retry attempts fail."""
        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = anthropic.APIError(
                message="persistent error",
                request=MagicMock(),
                body=None,
            )

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="Anthropic call failed after"):
                    adapter.generate("test")

        assert client_instance.messages.create.call_count == len(_RETRY_DELAYS)


class TestSafetyBlock:
    def test_bad_request_returns_safety_blocked(self):
        """BadRequestError in single generation returns SAFETY_BLOCKED."""
        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = anthropic.BadRequestError(
                message="content blocked",
                response=MagicMock(status_code=400, headers={}),
                body=None,
            )

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            result = adapter.generate("bad prompt")

        assert result.text == "SAFETY_BLOCKED"
        assert result.latency_ms == 0.0
        assert result.tokens_in is None
        assert result.tokens_out is None

    def test_bad_request_not_retried(self):
        """BadRequestError should not be retried (only called once)."""
        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = anthropic.BadRequestError(
                message="content blocked",
                response=MagicMock(status_code=400, headers={}),
                body=None,
            )

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            adapter.generate("bad prompt")

        # Should be called only once -- no retries for BadRequestError
        assert client_instance.messages.create.call_count == 1


class TestToolUse:
    def test_tool_use_loop(self):
        """Tool-use loop: model requests tool, gets result, returns final text."""
        # First response: model requests a tool call
        tool_block = _make_tool_use_block(
            tool_id="toolu_01abc",
            name="run_python",
            input_data={"code": "print(2 + 2)"},
        )
        first_response = _make_response(
            [_make_text_block("Let me calculate that."), tool_block],
            stop_reason="tool_use",
            input_tokens=15,
            output_tokens=25,
        )

        # Second response: model returns final text
        second_response = _make_response(
            [_make_text_block("The answer is 4.")],
            stop_reason="end_turn",
            input_tokens=30,
            output_tokens=10,
        )

        mock_tool_result = {
            "stdout": "4\n",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 50.0,
            "truncated": False,
        }

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = [first_response, second_response]

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")

            with patch("evalrun.tools.run_python", return_value=mock_tool_result):
                result = adapter.generate("What is 2+2?", tools_enabled=True)

        assert result.text == "The answer is 4."
        assert result.tokens_in == 45  # 15 + 30
        assert result.tokens_out == 35  # 25 + 10
        assert result.tool_meta == {"tool_calls": 1}
        assert result.latency_ms > 0

    def test_tool_use_no_tool_calls_returns_text(self):
        """When tools are enabled but model doesn't call any, returns text directly."""
        response = _make_response(
            [_make_text_block("No tools needed.")],
            stop_reason="end_turn",
            input_tokens=5,
            output_tokens=5,
        )

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            result = adapter.generate("Simple question", tools_enabled=True)

        assert result.text == "No tools needed."
        assert result.tool_meta is None

    def test_tool_use_max_turns_exhausted(self):
        """When max_tool_turns is reached, returns with max_turns_reached meta."""
        # Every response requests a tool call
        tool_block = _make_tool_use_block(
            tool_id="toolu_loop",
            name="run_python",
            input_data={"code": "print('looping')"},
        )
        tool_response = _make_response(
            [_make_text_block("Trying again..."), tool_block],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=10,
        )

        mock_tool_result = {
            "stdout": "looping\n",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 10.0,
            "truncated": False,
        }

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            # Return tool_response for every call (more than max_turns)
            client_instance.messages.create.return_value = tool_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")

            with patch("evalrun.tools.run_python", return_value=mock_tool_result):
                result = adapter.generate(
                    "Loop forever",
                    tools_enabled=True,
                    max_tool_turns=3,
                )

        assert result.tool_meta["tool_calls"] == 3
        assert result.tool_meta["max_turns_reached"] is True

    def test_tool_use_timeout(self):
        """When case_timeout is exceeded, returns with timed_out meta."""
        tool_block = _make_tool_use_block(
            tool_id="toolu_timeout",
            name="run_python",
            input_data={"code": "import time; time.sleep(100)"},
        )
        tool_response = _make_response(
            [_make_text_block("Working..."), tool_block],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=10,
        )

        mock_tool_result = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 1.0,
            "truncated": False,
        }

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.return_value = tool_response

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")

            # Use a very short timeout so the second iteration hits the deadline
            with patch("evalrun.tools.run_python", return_value=mock_tool_result):
                result = adapter.generate(
                    "Slow task",
                    tools_enabled=True,
                    case_timeout=0,  # Immediately times out on second iteration
                )

        # First iteration succeeds, second hits deadline
        assert result.tool_meta["timed_out"] is True

    def test_tool_use_unknown_tool(self):
        """Unknown tool names return an error result and loop continues."""
        unknown_block = _make_tool_use_block(
            tool_id="toolu_unknown",
            name="unknown_tool",
            input_data={"arg": "value"},
        )
        tool_response = _make_response(
            [unknown_block],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=10,
        )
        final_response = _make_response(
            [_make_text_block("Done.")],
            stop_reason="end_turn",
            input_tokens=20,
            output_tokens=5,
        )

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = [tool_response, final_response]

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")

            with patch("evalrun.tools.run_python"):
                result = adapter.generate("Use unknown tool", tools_enabled=True)

        assert result.text == "Done."
        assert result.tool_meta["tool_calls"] == 1

    def test_tool_use_bad_request_returns_gracefully(self):
        """BadRequestError during tool loop returns gracefully."""
        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = anthropic.BadRequestError(
                message="content blocked",
                response=MagicMock(status_code=400, headers={}),
                body=None,
            )

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
            result = adapter.generate("bad prompt", tools_enabled=True)

        assert result.text == ""
        assert result.tool_meta is None


class TestToolUseMessageFormat:
    def test_tool_result_message_format(self):
        """Verify the messages sent to the API have correct Anthropic format."""
        tool_block = _make_tool_use_block(
            tool_id="toolu_fmt",
            name="run_python",
            input_data={"code": "print(1)"},
        )
        first_response = _make_response(
            [tool_block],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=10,
        )
        second_response = _make_response(
            [_make_text_block("Result: 1")],
            stop_reason="end_turn",
            input_tokens=20,
            output_tokens=5,
        )

        mock_tool_result = {
            "stdout": "1\n",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5.0,
            "truncated": False,
        }

        with patch("anthropic.Anthropic") as MockClient:
            client_instance = MockClient.return_value
            client_instance.messages.create.side_effect = [first_response, second_response]

            adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")

            with patch("evalrun.tools.run_python", return_value=mock_tool_result):
                adapter.generate("Run code", tools_enabled=True)

        # Check the second API call's messages
        second_call_kwargs = client_instance.messages.create.call_args_list[1]
        all_kwargs = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        messages = all_kwargs["messages"]

        # Should be: user, assistant (with tool_use), user (with tool_result)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # The assistant message should contain the tool_use block
        assistant_content = messages[1]["content"]
        tool_use_items = [b for b in assistant_content if b.get("type") == "tool_use"]
        assert len(tool_use_items) == 1
        assert tool_use_items[0]["id"] == "toolu_fmt"
        assert tool_use_items[0]["name"] == "run_python"

        # The user message should contain a tool_result block
        user_content = messages[2]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "tool_result"
        assert user_content[0]["tool_use_id"] == "toolu_fmt"
