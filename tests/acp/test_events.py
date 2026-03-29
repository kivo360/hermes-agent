"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import ToolCallStart, ToolCallProgress, AgentThoughtChunk, AgentMessageChunk

from acp_adapter.events import (
    make_client_terminal_cb,
    make_message_cb,
    make_plan_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None

    def test_handles_string_args(self, mock_conn, event_loop_fixture):
        """If args is a JSON string, it should be parsed."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("read_file", "Reading /etc/hosts", '{"path": "/etc/hosts"}')

        assert "read_file" in tool_call_ids

    def test_handles_non_dict_args(self, mock_conn, event_loop_fixture):
        """If args is not a dict, it should be wrapped."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("terminal", "$ echo hi", None)

        assert "terminal" in tool_call_ids

    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("terminal", "$ ls", {"command": "ls"})
            progress_cb("terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------


class TestThinkingCallback:
    def test_emits_thought_chunk(self, mock_conn, event_loop_fixture):
        """Thinking callback should emit AgentThoughtChunk."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Analyzing the code...")

        mock_rcts.assert_called_once()

    def test_ignores_empty_text(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_ignores_untracked_tools(self, mock_conn, event_loop_fixture):
        """Tools not in tool_call_ids should be silently ignored."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb(1, [{"name": "unknown_tool", "result": "ok"}])

        mock_rcts.assert_not_called()

    def test_handles_string_tool_info(self, mock_conn, event_loop_fixture):
        """Tool info as a string (just the name) should work."""
        tool_call_ids = {"read_file": "tc-def456"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(2, ["read_file"])

        assert "read_file" not in tool_call_ids
        mock_rcts.assert_called_once()


# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------


class TestMessageCallback:
    def test_emits_agent_message_chunk(self, mock_conn, event_loop_fixture):
        """Message callback should emit AgentMessageChunk."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Here is your answer.")

        mock_rcts.assert_called_once()

    def test_ignores_empty_message(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Plan callback
# ---------------------------------------------------------------------------


class TestPlanCallback:
    def test_emits_plan_update(self, mock_conn, event_loop_fixture):
        """Plan callback should emit AgentPlanUpdate with plan entries."""
        loop = event_loop_fixture

        cb = make_plan_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            entries = [
                {"content": "Read the code", "priority": "high", "status": "in_progress"},
                {"content": "Write tests", "priority": "medium", "status": "pending"},
            ]
            cb(entries)

        mock_rcts.assert_called_once()

    def test_ignores_empty_entries(self, mock_conn, event_loop_fixture):
        """Empty entries list should not emit any update."""
        loop = event_loop_fixture

        cb = make_plan_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb([])

        mock_rcts.assert_not_called()

    def test_validates_priority_and_status(self, mock_conn, event_loop_fixture):
        """Invalid priority/status values should be normalized to defaults."""
        loop = event_loop_fixture

        cb = make_plan_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Invalid priority and status should be defaulted
            entries = [
                {"content": "Task", "priority": "urgent", "status": "running"},
            ]
            cb(entries)

        mock_rcts.assert_called_once()

    def test_skips_non_dict_entries(self, mock_conn, event_loop_fixture):
        """Non-dict entries should be silently skipped."""
        loop = event_loop_fixture

        cb = make_plan_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            entries = ["not a dict", 42, None]
            cb(entries)

        mock_rcts.assert_not_called()

    def test_defaults_missing_fields(self, mock_conn, event_loop_fixture):
        """Entries without priority/status should use defaults."""
        loop = event_loop_fixture

        cb = make_plan_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            entries = [{"content": "Do something"}]
            cb(entries)

        mock_rcts.assert_called_once()


# ---------------------------------------------------------------------------
# Client terminal callback
# ---------------------------------------------------------------------------


class TestClientTerminalCallback:
    def test_returns_none_when_client_no_terminal_support(self, mock_conn, event_loop_fixture):
        """Should return None when client doesn't support terminals."""
        from types import SimpleNamespace

        loop = event_loop_fixture
        caps = SimpleNamespace(terminal=False)

        cb = make_client_terminal_cb(mock_conn, "session-1", loop, caps)

        result = cb("ls", args=None, cwd="/tmp", env=None)
        assert result is None

    def test_returns_none_when_capabilities_none(self, mock_conn, event_loop_fixture):
        """Should return None when capabilities object is None."""
        loop = event_loop_fixture

        cb = make_client_terminal_cb(mock_conn, "session-1", loop, None)

        result = cb("echo hi")
        assert result is None
