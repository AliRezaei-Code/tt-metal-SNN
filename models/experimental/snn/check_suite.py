# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The gate for this package: everything that can be checked without a tt-metal toolchain.

Every check here ASSERTS rather than reporting, so running this under ``set -e`` aborts on the
first failure. That is the whole point of the file. A previous version of this gate printed
``PASS``/``BAD`` per check and returned 0 regardless, so ``set -e`` could not see a failure -- and
twice a state that the gate was reporting as failing still reached ``git push``. A check that
describes its result instead of enforcing it is not a check.

Run from the repository root::

    python3 models/experimental/snn/check_suite.py

It is deliberately not a pytest module: it is a gate over the package, not an assertion about it,
and ``conftest.py`` splits collection on tt-metal availability in a way that would file this
somewhere it cannot usefully run.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PKG = ROOT / "models/experimental/snn"
HOOK = re.compile(r"(?<!allow-)pytest\.raises(?!.*allow-pytest\.raises)")


def _run(*argv):
    return subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, check=False)


def _lint_detail(tool: str, proc) -> str:
    """Say "not installed" distinctly from "found problems".

    A missing linter and a lint failure are different facts. Collapsing them means a gate run on an
    interpreter without the tool reports formatting problems that do not exist, which trains you to
    ignore the message.
    """
    blob = (proc.stdout + proc.stderr).strip()
    if "No module named" in blob:
        return f"{tool} is not installed in this interpreter; the check could not run"
    return (blob.splitlines() or [""])[-1][:90]


def _environment_detail() -> tuple[bool, str]:
    """Is this machine able to run the checks at all, independently of the code?

    A full disk makes pytest fail in ways that look like real defects -- a module import dies with
    "No usable temporary directory" rather than anything to do with the package. That happened here
    and was read as a ttnn-boundary result before the cause was found. A gate that cannot tell
    "the code is broken" from "the host cannot run the checks" is a gate that will be believed
    wrongly at exactly the moment it matters.
    """
    import shutil
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(prefix="snn-gate-", suffix=".tmp"):
            pass
    except OSError as exc:
        return False, f"cannot write to the temp directory ({exc.__class__.__name__})"

    free = shutil.disk_usage(ROOT).free
    if free < 256 * 1024 * 1024:
        return False, (
            f"only {free // (1024 * 1024)} MiB free on the volume holding the checkout; pytest "
            "needs room for temp files and a full disk surfaces as unrelated failures"
        )
    return True, f"{free // (1024 * 1024)} MiB free"


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    ok, detail = _environment_detail()
    checks.append(("environment can run the checks", ok, detail))

    suite = _run(
        "-m",
        "pytest",
        "models/experimental/snn/tests/",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--confcutdir=models/experimental/snn",
    )
    tail = [ln for ln in suite.stdout.splitlines() if "passed" in ln or "failed" in ln]
    checks.append(("device-free suite green", suite.returncode == 0, tail[-1].strip() if tail else suite.stdout[-200:]))

    budget = _run(".github/scripts/utils/verify_time_budget.py")
    buckets = [ln for ln in budget.stdout.splitlines() if "buckets charged" in ln]
    checks.append(("verify_time_budget", budget.returncode == 0, buckets[0].strip() if buckets else ""))

    # black is a gate step, not something to eyeball. Run as a plain statement so a non-zero exit
    # aborts: `black --check ... && echo PASS` is the trap, because the left side of `&&` is exempt
    # from `set -e` and a formatting failure then prints nothing while execution carries on.
    black = subprocess.run(
        [sys.executable, "-m", "black", "--check", *[str(f) for f in sorted(PKG.rglob("*.py"))]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    checks.append(("black formatting", black.returncode == 0, _lint_detail("black", black)))

    # ruff is a gate step for the same reason black is: both live here, not in the command that
    # pushes, so there is no second place for a lint result to be reported from.
    #
    # `--select E,F` is deliberate. This gate is not the repo's linter -- ruff appears nowhere in
    # .pre-commit-config.yaml -- it is here for the two things that have actually caught defects
    # here: undefined names (F821) and unused imports (F401). Leaving the rule set at ruff's
    # defaults makes the gate hostage to whichever version happens to be installed, and a newer one
    # flags style nits (implicit Optional, import order) that predate this work.
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "E,F", "models/experimental/snn/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    checks.append(("ruff lint", ruff.returncode == 0, _lint_detail("ruff", ruff)))

    offenders = [
        f"{f.name}:{i}"
        for f in (PKG / "tests").glob("*.py")
        for i, ln in enumerate(f.read_text().splitlines(), 1)
        if HOOK.search(ln)
    ]
    checks.append(("prefer-expect-error hook", not offenders, str(offenders)))

    # Running the demo at the scale the README documents is what found the tile-vs-neuron
    # conflation, so it stays in the gate rather than being left to a manual run.
    demo = _run("-m", "models.experimental.snn.demo.demo", "--no-device", "--steps", "50", "--samples", "256")
    checks.append(("documented demo command", demo.returncode == 0, demo.stderr[-160:] if demo.returncode else ""))

    # A claim-level regression, not a formatting one: the sparsity wording maps the fraction of
    # input *tiles* onto bytes moved, and conflating that with the fraction of *neurons* fired is
    # false -- the demo reports 10.1% firing against 60% of weight tiles fetched.
    docs = [PKG / "README.md", PKG / "snn/synapses.py", PKG / "tests/README.md"]
    stale = [d.name for d in docs if "fraction of the input population" in d.read_text()]
    checks.append(("no conflated sparsity claim", not stale, str(stale)))

    for name, ok, detail in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    failed = [name for name, ok, _ in checks if not ok]
    assert not failed, f"GATE FAILED: {failed}"
    print("  GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
