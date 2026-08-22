"""Streamlit presentation layer for the enterprise support pipeline.

Presentation and integration only (AGENTS.md section 3): these modules
invoke the compiled graph and render ``PipelineState``, session
telemetry, the JSONL fallback log, the loaded corpus, and the frozen
runtime configuration. No routing, threshold, guardrail, rewrite, or
citation logic is duplicated here, and the dependency direction is one
way: ``ui`` imports from ``src``, never the reverse.

The split follows what each part can be held to. ``labels`` and
``styles`` are data; ``formatting`` is pure functions over a state,
which is the part ``tests/test_ui_formatting.py`` asserts; ``runtime``
owns the cached graph and the session record; ``assistant`` and
``console`` render the two pages, and ``app.py`` only composes them.
"""
