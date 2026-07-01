"""Unit tests for agents/tool_gap_analyzer.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAnalyzeForToolGap:
    @pytest.mark.asyncio
    async def test_no_gap_detected_does_not_write_file(self, tmp_path):
        from agents.tool_gap_analyzer import analyze_for_tool_gap
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"gap_detected": false}'
        client.chat.completions.create = AsyncMock(return_value=resp)
        suggestions_file = tmp_path / "tool_suggestions.jsonl"
        with patch("agents.tool_gap_analyzer._SUGGESTIONS_FILE", suggestions_file):
            await analyze_for_tool_gap(
                client=client,
                model="test-model",
                query="hello",
                response="Hi there",
                existing_tool_names=["get_orders"],
            )
        assert not suggestions_file.exists()

    @pytest.mark.asyncio
    async def test_gap_detected_writes_to_file(self, tmp_path):
        from agents.tool_gap_analyzer import analyze_for_tool_gap
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({
            "gap_detected": True,
            "tool_name": "get_returns",
            "description": "Fetch return requests",
            "example_queries": ["zwroty"],
        })
        client.chat.completions.create = AsyncMock(return_value=resp)
        suggestions_file = tmp_path / "tool_suggestions.jsonl"
        with patch("agents.tool_gap_analyzer._SUGGESTIONS_FILE", suggestions_file):
            await analyze_for_tool_gap(
                client=client,
                model="test-model",
                query="pokaż zwroty",
                response="I cannot fetch return requests.",
                existing_tool_names=["get_orders"],
            )
        assert suggestions_file.exists()
        line = json.loads(suggestions_file.read_text().strip())
        assert line["suggested_tool"] == "get_returns"

    @pytest.mark.asyncio
    async def test_exception_is_swallowed(self):
        from agents.tool_gap_analyzer import analyze_for_tool_gap
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        # Should not raise
        await analyze_for_tool_gap(
            client=client,
            model="test-model",
            query="hello",
            response="hello",
            existing_tool_names=[],
        )

    @pytest.mark.asyncio
    async def test_invalid_json_is_swallowed(self):
        from agents.tool_gap_analyzer import analyze_for_tool_gap
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "not valid json"
        client.chat.completions.create = AsyncMock(return_value=resp)
        await analyze_for_tool_gap(
            client=client, model="m", query="q", response="r",
            existing_tool_names=[],
        )

    @pytest.mark.asyncio
    async def test_markdown_json_parsed(self, tmp_path):
        from agents.tool_gap_analyzer import analyze_for_tool_gap
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = (
            '```json\n{"gap_detected": true, "tool_name": "foo",'
            ' "description": "bar", "example_queries": []}\n```'
        )
        client.chat.completions.create = AsyncMock(return_value=resp)
        suggestions_file = tmp_path / "tool_suggestions.jsonl"
        with patch("agents.tool_gap_analyzer._SUGGESTIONS_FILE", suggestions_file):
            await analyze_for_tool_gap(
                client=client, model="m", query="q", response="r",
                existing_tool_names=[],
            )
        assert suggestions_file.exists()
