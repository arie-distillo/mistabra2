"""Unit-test configuration.

tests/ contains ONLY unit tests: pure, offline, deterministic, no HTTP, no LLM,
no external services. They import product functions directly (appropriate for
units with no API surface) and never touch the network.

Acceptance tests — which drive the system through its HTTP API and may hit a real
LLM — live in acceptance/ and are run with `python -m acceptance.run`, not pytest.
"""
