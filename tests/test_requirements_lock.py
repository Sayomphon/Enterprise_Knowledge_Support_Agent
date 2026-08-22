"""Unit tests that keep the direct pins and the resolved lock in step.

``requirements.txt`` is the short list a human edits; ``requirements.lock``
is what a reproducible install actually reads, hash by hash. Nothing but
discipline connects them, so a pin bumped without regenerating the lock
would otherwise surface as a CI job installing a tree nobody reviewed.
These checks are offline: they read both files as text and never resolve
anything against an index.
"""

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DIRECT_FILE = _ROOT / "requirements.txt"
_LOCK_FILE = _ROOT / "requirements.lock"

# name==version, optionally followed by an environment marker. uv writes
# every entry in this shape; anything else is a defect this module reports
# rather than parses.
_PIN_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;]+)")


def _canonical(name: str) -> str:
    """Fold a distribution name the way PEP 503 defines name equality."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_lines(path: Path) -> list[str]:
    """Read one requirement per line, continuations folded and comments cut.

    A hashed entry spans many physical lines joined by a trailing backslash,
    and uv annotates each one with a ``# via`` comment. Folding first means
    every returned line carries a requirement together with all of its
    hashes, which is the unit both tests below assert on.
    """
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    entries: list[str] = []
    for raw_line in text.splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    return entries


def _pinned_versions(path: Path) -> dict[str, set[str]]:
    """Map each canonical name to every version the file pins for it.

    The value is a set because an environment marker may pin one name more
    than once: ``numpy`` is locked twice, since the release supporting the
    3.11 floor is not the one 3.12 resolves to.
    """
    versions: dict[str, set[str]] = {}
    for entry in _requirement_lines(path):
        match = _PIN_PATTERN.match(entry)
        if match is not None:
            name = _canonical(match["name"])
            versions.setdefault(name, set()).add(match["version"])
    return versions


class TestLockCoversTheDirectPins(unittest.TestCase):
    """Every dependency a human pinned has to appear in the lock."""

    def test_every_direct_pin_is_locked_at_the_same_version(self) -> None:
        direct = _pinned_versions(_DIRECT_FILE)
        locked = _pinned_versions(_LOCK_FILE)

        self.assertTrue(direct, "requirements.txt pinned nothing")
        for name, versions in direct.items():
            with self.subTest(package=name):
                self.assertIn(name, locked, f"{name} is missing from the lock")
                self.assertEqual(versions, locked[name])

    def test_the_lock_also_pins_the_transitive_tree(self) -> None:
        # The whole point of the lock is the packages nobody typed: if it
        # only held the direct list, an install would still float.
        direct = _pinned_versions(_DIRECT_FILE)
        locked = _pinned_versions(_LOCK_FILE)

        self.assertGreater(len(locked), len(direct))


class TestLockInstallsUnderRequireHashes(unittest.TestCase):
    """``pip install --require-hashes`` refuses a lock with a soft line."""

    def test_every_entry_is_pinned_to_an_exact_version(self) -> None:
        for entry in _requirement_lines(_LOCK_FILE):
            with self.subTest(entry=entry[:40]):
                self.assertRegex(entry, _PIN_PATTERN)

    def test_every_entry_carries_at_least_one_hash(self) -> None:
        for entry in _requirement_lines(_LOCK_FILE):
            with self.subTest(entry=entry[:40]):
                self.assertIn("--hash=sha256:", entry)


if __name__ == "__main__":
    unittest.main()
