"""Opt-in live tests. Nothing here runs unless it is asked for.

The offline suite is the contract (AGENTS.md section 4, invariant 11):
it must pass with no credential and no network. This package holds the
one exception that invariant allows -- a clearly-marked live smoke test
-- and every module in it skips itself unless ``RUN_LIVE_SMOKE=1`` is
set AND a credential is configured, so ``unittest discover`` over the
whole ``tests`` tree stays free and offline by default.
"""
