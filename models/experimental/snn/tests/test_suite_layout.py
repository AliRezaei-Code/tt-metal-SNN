# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Guards on the suite's own test/toolchain split, which belong in a test module, not a conftest.

These checks used to live in ``conftest.py``. pytest does not collect test functions from a
conftest module, so all three were inert: they read as tests, they are written as tests, and they
executed zero times in any environment, local or CI. What they guard is the ``collect_ignore``
list -- the single mechanism that decides which half of this suite runs -- so a guard that cannot
fire is worse than no guard at all, because the file reads as though the split is verified.

Moving them here is only half the fix. Parametrised over ``_DEVICE_TESTS``, ``test_device_modules_exist``
would have read the very list it is supposed to police: delete a module and drop its name in the
same change and the parametrisation simply shrinks, leaving the suite green -- precisely the
failure its own docstring claims to catch. So the two halves are recorded below as frozen tuples,
independently of ``conftest``, and every check is written against those records instead.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import _DEVICE_TESTS, _ttnn_runtime_available, collect_ignore

TESTS_DIR = Path(__file__).resolve().parent

# The independent record. A module added to or removed from either half must be added to or
# removed from the matching tuple here, or test_the_records_cover_the_directory fails.
DEVICE_HALF = (
    "test_dsl.py",
    "test_fabric_program_guards.py",
    "test_lif_neuron.py",
    "test_multicore_partition.py",
    "test_sparse_matvec.py",
    "test_spike_propagation.py",
)
FREE_HALF = (
    "test_cb_map.py",
    "test_dsl_cache_key.py",
    "test_kernel_api_symbols.py",
    "test_package_imports.py",
    "test_reference_only.py",
    "test_suite_layout.py",
    "test_ttnn_call_sites.py",
)


def test_the_records_cover_the_directory():
    """Every test module must be classified, so a new or deleted file cannot slip through unrecorded."""
    on_disk = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))
    recorded = sorted(DEVICE_HALF + FREE_HALF)
    assert on_disk == recorded, (
        "the frozen half records no longer match the directory; " f"unrecorded: {sorted(set(on_disk) ^ set(recorded))}"
    )


def test_the_device_list_matches_its_record():
    """`_DEVICE_TESTS` is what `collect_ignore` is built from, so it must not drift from the record."""
    assert sorted(_DEVICE_TESTS) == sorted(
        DEVICE_HALF
    ), f"_DEVICE_TESTS is {sorted(_DEVICE_TESTS)}, the record says {sorted(DEVICE_HALF)}"


def test_the_device_half_is_skipped_without_a_runtime():
    """The split must be active wherever the runtime is absent, and inactive where it is present.

    Kept for the runtime-present branch, which nothing else covers: it asserts that
    `collect_ignore` is empty when a real runtime exists, a state the record check never looks at.
    The no-runtime branch *is* redundant with `test_the_device_list_matches_its_record`, since
    conftest defines `collect_ignore = list(_DEVICE_TESTS)` at import, so comparing them reduces to
    the record comparison. It is left in place so the runtime-state contract is stated in one place.
    """
    if _ttnn_runtime_available():
        assert collect_ignore == [], "a runtime is available, so nothing should be ignored"
    else:
        assert sorted(collect_ignore) == sorted(
            DEVICE_HALF
        ), "without a tt-metal runtime every device test must be ignored"


def test_free_half_is_never_ignored():
    """The toolchain-free modules must be collected whatever the runtime state, and must exist."""
    for name in FREE_HALF:
        assert name not in collect_ignore, f"{name} needs no toolchain and must never be ignored"
        assert (TESTS_DIR / name).exists(), f"{name} is listed as toolchain-free but is missing"


@pytest.mark.parametrize("name", DEVICE_HALF)
def test_device_modules_exist(name):
    """A renamed or deleted device test would otherwise vanish from the ignore list silently."""
    assert (TESTS_DIR / name).exists(), f"{name} is listed as a device test but is missing"


# --- the count block in README.md -------------------------------------------------
#
# Counts decay silently: a test is added and the table keeps the old number, and nothing fails
# because nothing reads the table. Four documents drifted out of step with the tree this way while
# this suite was being written, so the block is checked rather than trusted.
#
# Counting is done by asking pytest, not by re-implementing parametrisation here: a second
# implementation of "how many tests does this module have" would itself drift from pytest's. The
# child run deselects this test by name, so the collection cannot re-enter and recurse.

DOC = TESTS_DIR / "README.md"
_COUNT_LINE = re.compile(r"^(?P<module>test_\w+\.py)\s+(?P<count>\d+)\s+tests")
_TOTAL_LINE = re.compile(r"^\s*(?P<count>\d+)\s+total on a host with no toolchain\s*$")


def _documented_counts() -> dict:
    """The `NN tests` entries of the README count block, and the total it claims."""
    counts, total = {}, None
    for line in DOC.read_text().splitlines():
        if m := _COUNT_LINE.match(line.strip()):
            counts[m["module"]] = int(m["count"])
        elif m := _TOTAL_LINE.match(line):
            total = int(m["count"])
    return {"modules": counts, "total": total}


def _collected_counts() -> dict:
    """Per-module collected-test counts, straight from pytest."""
    env = dict(os.environ, PYTEST_ADDOPTS="")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_DIR),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            # Without this the child loads the conftests above this directory, which import
            # loguru, torch, tracy and tt_umd at module scope -- the same reason the documented
            # invocation in README.md carries the flag.
            f"--confcutdir={TESTS_DIR.parent}",
            # No deselect needed: --collect-only collects without executing, so this test cannot
            # re-enter. Deselecting it here would have made the child undercount this very module.
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(f"the child collection failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    counts: dict = {}
    for line in proc.stdout.splitlines():
        if "::" in line and line.strip().endswith(("]", ")")) or "::" in line:
            module = line.split("::", 1)[0].strip().rsplit("/", 1)[-1]
            if module.startswith("test_") and module.endswith(".py"):
                counts[module] = counts.get(module, 0) + 1
    return counts


def test_documented_counts_match_the_suite():
    """Every count in the README block must be the count pytest actually produces, in both directions.

    Checking only the documented names leaves a hole: delete a row *and* correct the total, and the
    block is internally consistent while silently omitting a module that is really there. So the
    collected set is compared to the documented set, not just used to look documented names up.
    """
    doc = _documented_counts()
    assert doc["modules"], "the README count block could not be parsed"
    assert doc["total"] is not None, "the README count block states no total"

    collected = _collected_counts()
    documented = set(doc["modules"])

    # Unconditional, and valid under a real runtime too: anything the block names must collect.
    # Redundant with the loop below, and kept deliberately: `collected[module]` would raise a raw
    # KeyError naming nothing useful. This reports the undocumented names in one message instead,
    # which is the difference between "KeyError: 'test_x.py'" and a sentence saying which module
    # the block claims but pytest does not collect.
    undocumented = documented - set(collected)
    assert not undocumented, f"{DOC.name} lists {sorted(undocumented)}, which pytest does not collect"
    for module, claimed in sorted(doc["modules"].items()):
        assert (
            collected[module] == claimed
        ), f"{DOC.name} says {module} has {claimed} tests, pytest collects {collected[module]}"

    if _ttnn_runtime_available():
        # The block is explicitly labelled "on a host with no toolchain", so both the total and
        # the set equality are scoped to that case. Under a real runtime all eleven modules
        # collect while the block still describes only the five toolchain-free ones, so demanding
        # equality here would be wrong rather than stricter. The subset check above still applies.
        return

    # The other direction: a module that collects here but has no row is as stale as the reverse.
    missing = set(collected) - documented
    assert not missing, f"{sorted(missing)} collect here but have no row in {DOC.name}"
    assert doc["total"] == sum(
        doc["modules"].values()
    ), f"{DOC.name} states a total of {doc['total']}, its own rows sum to {sum(doc['modules'].values())}"
