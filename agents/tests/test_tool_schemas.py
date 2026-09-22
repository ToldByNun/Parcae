from __future__ import annotations

import json

import pytest

from parcae_agent.allowlist import ALLOWED_TOOLS, BRIDGE_OWNED_KEYS, TOOL_ARG_KEYS
from parcae_agent.tool_schemas import (
    TOOL_SCHEMA_ORDER,
    assert_schemas_match_allowlist,
    openai_tools,
    tool_names,
    tool_schema,
)


def test_assert_schemas_match_allowlist_ok() -> None:
    assert_schemas_match_allowlist()


def test_openai_tools_default_covers_allowlist() -> None:
    tools = openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert names == list(TOOL_SCHEMA_ORDER)
    assert set(names) == ALLOWED_TOOLS
    for entry in tools:
        assert entry["type"] == "function"
        params = entry["function"]["parameters"]
        assert params["type"] == "object"
        assert params.get("additionalProperties") is False


def test_openai_tools_subset_and_rejects_unknown() -> None:
    tools = openai_tools(["catalog", "rank"])
    assert [t["function"]["name"] for t in tools] == ["catalog", "rank"]
    with pytest.raises(KeyError, match="allow-list"):
        openai_tools(["catalog", "blind_crack"])


def test_schema_properties_match_tool_arg_keys() -> None:
    for name in tool_names():
        props = set(tool_schema(name)["function"]["parameters"]["properties"])
        assert props == set(TOOL_ARG_KEYS[name])


def test_schemas_omit_bridge_owned_keys() -> None:
    for name in tool_names():
        props = set(tool_schema(name)["function"]["parameters"]["properties"])
        overlap = props & BRIDGE_OWNED_KEYS
        assert not overlap, f"{name} exposes bridge-owned keys: {overlap}"


def test_required_fields_are_subsets_of_properties() -> None:
    for name in tool_names():
        params = tool_schema(name)["function"]["parameters"]
        required = params.get("required", [])
        props = set(params["properties"])
        assert set(required) <= props


def test_tool_schema_unknown() -> None:
    with pytest.raises(KeyError):
        tool_schema("not_a_tool")


def test_schemas_are_json_serializable() -> None:
    payload = json.dumps(openai_tools())
    assert "hypothesis_list" in payload
    assert "search_cycle" in payload
    for entry in openai_tools():
        props = entry["function"]["parameters"]["properties"]
        assert "data_dir" not in props
        assert "workspace" not in props


def test_search_cycle_schema_steers_large_grids() -> None:
    desc = tool_schema("search_cycle")["function"]["description"].lower()
    assert "family" in desc or "grid" in desc
    props = tool_schema("search_cycle")["function"]["parameters"]["properties"]
    assert "family" in props
    assert "status" in props
    assert props["status"]["type"] == "boolean"


def test_descriptions_steer_away_from_invented_ids() -> None:
    catalog = tool_schema("catalog")["function"]["description"].lower()
    decode = tool_schema("decode")["function"]["description"].lower()
    assert "catalog" in decode or "invent" in decode
    assert "transform" in catalog or "score" in catalog
