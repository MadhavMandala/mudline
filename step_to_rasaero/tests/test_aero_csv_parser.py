"""Reading a RASAero II CSV export.

The fixture is a trimmed copy of a real export from RASAero II 1.0.2.0 --
same columns, same order, same quirks -- because the quirks are the whole
difficulty. In particular RASAero writes both a bare ``CD`` column and
``CD Power-Off``/``CD Power-On``, and the bare one holds the alpha-zero value
repeated down every alpha of a given Mach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step_to_rasaero.aero_csv_parser import parse_rasaero_csv  # noqa: E402

HEADER = (
    "Mach,Alpha,CD,CD Power-Off,CD Power-On,CA Power-Off,CA Power-On,CL,CN,"
    "CN Potential,CN Viscous,CNalpha (0 to 4 deg) (per rad),CP,"
    "CP (0 to 4 deg),Reynolds Number"
)

#: Mach 0.5 and 1.5, alpha 0/2/4. Note the third column repeating.
ROWS = [
    "0.50,0,0.342177,0.342177,0.342177,0.342177,0.342177,0,0,0,0,13.12,54.15,54.15,21549085",
    "0.50,2,0.342177,0.357551,0.357551,0.342594,0.342594,0.422331,0.434552,0.4,0.03,13.12,54.30,54.15,21549085",
    "0.50,4,0.342177,0.406906,0.406906,0.343850,0.343850,0.889727,0.915944,0.8,0.11,13.12,54.31,54.15,21549085",
    "1.50,0,0.564400,0.564400,0.564400,0.564400,0.564400,0,0,0,0,12.20,55.45,55.45,64647256",
    "1.50,2,0.564400,0.590000,0.590000,0.565000,0.565000,0.410000,0.432000,0.4,0.03,12.20,55.50,55.45,64647256",
    "1.50,4,0.564400,0.626100,0.626100,0.566000,0.566000,0.840000,0.864300,0.8,0.11,12.20,53.72,55.45,64647256",
]


@pytest.fixture
def export(tmp_path) -> Path:
    path = tmp_path / "aero.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_every_mach_and_alpha_survives(export):
    database = parse_rasaero_csv(export, reference_length_m=1.85)
    assert {row.mach for row in database.rows} == {0.5, 1.5}
    assert {row.alpha_deg for row in database.rows} == {0.0, 2.0, 4.0}


def test_the_powered_column_is_carried_only_when_asked(export):
    """Both drag columns are read; ``power_on`` decides whether the second
    rides along for the flight to switch to at burnout."""
    single = parse_rasaero_csv(export, reference_length_m=1.85)
    assert not single.has_power_on

    both = parse_rasaero_csv(export, reference_length_m=1.85, power_on=True)
    assert both.has_power_on
    row = both.lookup(0.5, 4.0)
    # This export's two columns happen to agree, so equality is the check.
    assert np.isclose(row.cd_power_on, row.cd)
    assert np.isclose(row.cd, 0.406906)


def test_drag_rises_with_angle_of_attack(export):
    """The bare CD column does not move with alpha; the power-qualified one does.

    Reading the bare column gave a table whose drag was independent of angle
    of attack, so a vehicle weathercocking into a crosswind paid no induced
    drag at all -- and this is the one place a rocket spends real energy on
    being pointed slightly wrong.
    """
    database = parse_rasaero_csv(export, reference_length_m=1.85)
    at = {row.alpha_deg: row.cd for row in database.rows if row.mach == 0.5}
    assert at[0.0] < at[2.0] < at[4.0]
    assert np.isclose(at[4.0], 0.406906, rtol=1e-6)


def test_centre_of_pressure_converts_from_inches(export):
    database = parse_rasaero_csv(export, reference_length_m=1.85)
    row = next(r for r in database.rows if r.mach == 0.5 and r.alpha_deg == 4.0)
    assert np.isclose(row.x_cp_m, 54.31 * 0.0254, rtol=1e-6)


def test_normal_force_comes_from_cn_not_cl(export):
    """CL sits next to CN in the export and is the wrong one to take."""
    database = parse_rasaero_csv(export, reference_length_m=1.85)
    row = next(r for r in database.rows if r.mach == 0.5 and r.alpha_deg == 4.0)
    assert np.isclose(row.cn, 0.915944, rtol=1e-6)


def test_a_file_with_nothing_recognizable_is_refused(tmp_path):
    path = tmp_path / "junk.csv"
    path.write_text("time,altitude\n0,0\n1,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_rasaero_csv(path)
