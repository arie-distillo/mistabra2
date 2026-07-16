"""Unit tests: extracting a JSON object from a model's raw text output.

`extract_json` is the parser at the LLM boundary. Models return JSON wrapped in
prose, fenced in ```json blocks, or (on some providers) as null content. This file
pins down what it accepts and what it rejects. Pure function, no I/O.
"""
import pytest
from counterpoint.llm import extract_json


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),                                    # plain
    ('```json\n{"a": 1}\n```', {"a": 1}),                      # fenced with lang
    ('```\n{"a": 1}\n```', {"a": 1}),                          # fenced bare
    ('sure, here: {"a": 1} — hope that helps', {"a": 1}),      # embedded in prose
    ('{"change": "more", "why": "text with } brace"}',         # brace inside a string
     {"change": "more", "why": "text with } brace"}),
])
def test_accepts_wrapped_and_fenced_json(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["not json", "", "```json\n```", "[1,2,3] not an object"])
def test_rejects_non_objects(raw):
    with pytest.raises(ValueError):
        extract_json(raw)


def test_rejects_none_content():
    # regression: a provider returning content=None crashed the parser and, via the
    # retry loop, turned one run into a 59-minute real-cost job.
    with pytest.raises(ValueError):
        extract_json(None)
