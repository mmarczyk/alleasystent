"""Unit tests for agents/allegro/allegro_tools.py."""
from __future__ import annotations

import pytest


class TestAllegroTools:
    @pytest.fixture(autouse=True)
    def load_tools(self):
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS
        self.tools = ALLEGRO_TOOLS

    def test_is_list(self):
        assert isinstance(self.tools, list)

    def test_has_at_least_10_tools(self):
        assert len(self.tools) >= 10

    def test_all_have_type_function(self):
        for tool in self.tools:
            assert tool.get("type") == "function", f"Tool missing type=function: {tool}"

    def test_all_have_name(self):
        for tool in self.tools:
            assert "name" in tool["function"], f"Tool missing name: {tool}"
            assert isinstance(tool["function"]["name"], str)
            assert tool["function"]["name"] != ""

    def test_all_have_description(self):
        for tool in self.tools:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 10

    def test_all_have_parameters(self):
        for tool in self.tools:
            assert "parameters" in tool["function"]
            params = tool["function"]["parameters"]
            assert params.get("type") == "object"
            assert "properties" in params

    def test_known_tool_names_present(self):
        names = {t["function"]["name"] for t in self.tools}
        expected = {
            "get_new_orders",
            "get_orders",
            "get_order_details",
            "get_active_offers",
            "update_offer_price",
            "update_offer_stock",
            "send_message_to_buyer",
            "get_message_threads",
            "get_account_info",
        }
        for name in expected:
            assert name in names, f"Expected tool '{name}' not found"

    def test_names_are_unique(self):
        names = [t["function"]["name"] for t in self.tools]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_get_orders_has_status_enum(self):
        tool = next(t for t in self.tools if t["function"]["name"] == "get_orders")
        props = tool["function"]["parameters"]["properties"]
        assert "status" in props
        assert "enum" in props["status"]
