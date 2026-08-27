"""Validate the engine against RASAero II's own output, term by term.

These are the tests that give the reimplementation its meaning. Everything
else checks that the code does what it was written to do; these check that
what it was written to do is what RASAero does.

The reference data is committed (``oracle/golden``), so this runs without
RASAero installed and without taking over the desktop. Regenerate it with::

    python -m oracle.build && python -m oracle.package

Tolerance is per-column and precision-aware -- see ``oracle.compare``. RASAero
prints CN-alpha to two decimals and everything else to three, so a single
flat tolerance would either mask a real CN-alpha error or fail a correct one.
"""

from __future__ import annotations

import gzip
import io

import pytest

from aeroengine.cdx1 import load as load_cdx1
from aeroengine.solver import Engine
from oracle.compare import compare
from oracle.package import GOLDEN, load_manifest
from oracle.runtest import Dump

CASES = load_manifest()

pytestmark = pytest.mark.oracle


def _dump_from_csv_gz(path) -> Dump:
    """Rebuild a Dump from a frozen CSV."""
    import csv

    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        parsed = {}
        for key, value in row.items():
            if value == "" or value is None:
                continue
            parsed[key] = value if key == "regime" else float(value)
        rows.append(parsed)
    return Dump(rows=rows)


@pytest.mark.skipif(not CASES, reason="no golden data; run `python -m oracle.package`")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_matches_rasaero(case):
    """Every component term, at every tabulated Mach, within print precision."""
    design = load_cdx1(GOLDEN / case["cdx1"])
    dump = _dump_from_csv_gz(GOLDEN / case["csv_gz"])
    report = compare(
        Engine(design), dump, alpha_deg=case["alpha_deg"], case=case["case"]
    )

    assert report.comparisons > 0, "golden file produced no comparisons"
    if not report.passed:
        head = "\n".join(str(m) for m in report.mismatches[:20])
        extra = (
            f"\n... and {len(report.mismatches) - 20} more"
            if len(report.mismatches) > 20 else ""
        )
        pytest.fail(
            f"{report.summary()}\n{head}{extra}\n\n"
            "A failure here means the engine and RASAero disagree by more than "
            "RASAero's own printed precision. Check the named term against i.cs "
            "before assuming the reference is stale."
        )


@pytest.mark.skipif(not CASES, reason="no golden data")
def test_golden_data_is_self_consistent():
    """The committed CDX1 files must still parse and match their digests."""
    import hashlib

    for case in CASES:
        path = GOLDEN / case["cdx1"]
        assert path.exists(), f"{case['case']}: missing {case['cdx1']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        assert digest == case["cdx1_sha256_16"], (
            f"{case['case']}: {case['cdx1']} has changed since the reference "
            "was generated. The golden numbers describe a different vehicle."
        )
