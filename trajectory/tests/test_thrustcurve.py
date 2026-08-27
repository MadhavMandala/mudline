"""Tests for reading published motor files.

The sample data is real: an Estes D12 and an AeroTech K550W, in the formats the
manufacturers actually publish. Checking against published figures is the point
-- a parser that reads without error but lands on the wrong units produces a
plausible trajectory for the wrong motor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.propulsion.thrustcurve import (  # noqa: E402
    ThrustCurve,
    impulse_class,
    load_thrust_curve,
    load_thrust_curves,
    parse_eng,
    parse_rse,
)

# An Estes D12, abridged. 20 N-s total, 0.0246 kg propellant, 0.0564 kg loaded.
D12_ENG = """; Estes D12
; from ThrustCurve.org
D12 24 70 0-3-5-7 0.0246 0.0564 Estes
   0.049 2.569
   0.116 9.369
   0.184 17.275
   0.237 24.258
   0.282 29.396
   0.297 27.000
   0.311 22.323
   0.322 17.276
   0.348 14.049
   0.386 11.379
   0.442 10.263
   0.546 10.263
   0.718 10.263
   0.879 10.078
   1.066 10.078
   1.375 10.263
   1.502 10.263
   1.636 9.900
   1.700 9.494
   1.730 9.126
   1.760 5.864
   1.790 3.895
   1.820 1.926
   1.850 0.000
;
"""

# An AeroTech K550W, abridged. Note the grams.
K550_RSE = """<engine-database>
 <engine-list>
  <engine mfg="AeroTech" code="K550W" Type="single-use" dia="54" len="410"
          initWt="1451" propWt="797" delays="0" auto-calc-mass="1"
          Itot="1596.5" avgThrust="550" peakThrust="722" burn-time="2.9"
          Isp="204" exitDia="0">
   <data>
    <eng-data t="0.02" f="620" m="797"/>
    <eng-data t="0.10" f="700" m="770"/>
    <eng-data t="0.50" f="660" m="640"/>
    <eng-data t="1.00" f="600" m="470"/>
    <eng-data t="1.50" f="550" m="310"/>
    <eng-data t="2.00" f="480" m="170"/>
    <eng-data t="2.50" f="300" m="60"/>
    <eng-data t="2.90" f="0" m="0"/>
   </data>
  </engine>
 </engine-list>
</engine-database>
"""


# ------------------------------------------------------------------ RASP .eng


def test_eng_header_is_read():
    motor = parse_eng(D12_ENG)[0]
    assert motor.name == "D12"
    assert motor.manufacturer == "Estes"
    assert motor.diameter_mm == 24.0
    assert motor.length_mm == 70.0
    assert motor.delays == "0-3-5-7"


def test_eng_masses_are_kilograms_already():
    """RASP writes mass in kg -- dividing by 1000 here would be a 1000x error."""
    motor = parse_eng(D12_ENG)[0]
    assert np.isclose(motor.propellant_mass_kg, 0.0246)
    assert np.isclose(motor.total_mass_kg, 0.0564)
    assert np.isclose(motor.dry_mass_kg, 0.0318)


def test_eng_total_impulse_matches_the_published_class():
    """A D is 10-20 N-s. Landing outside that means the curve is misread."""
    motor = parse_eng(D12_ENG)[0]
    assert 15.0 < motor.total_impulse_ns <= 20.0
    assert motor.impulse_class == "D"


def test_eng_peak_and_burn_time_are_sane():
    motor = parse_eng(D12_ENG)[0]
    assert np.isclose(motor.peak_thrust_n, 29.396)
    assert np.isclose(motor.burn_time_s, 1.850)
    # A D12 averages about 12 N by name.
    assert 8.0 < motor.average_thrust_n < 14.0


def test_comments_and_blank_lines_are_ignored():
    noisy = D12_ENG.replace("; Estes D12", "; Estes D12\n\n;; another comment")
    assert len(parse_eng(noisy)[0].times_s) == len(parse_eng(D12_ENG)[0].times_s)


def test_a_trailing_comment_on_a_data_line_is_stripped():
    text = D12_ENG.replace("   0.049 2.569", "   0.049 2.569  ; ignition")
    motor = parse_eng(text)[0]
    assert np.isclose(motor.thrust_n[1], 2.569)


def test_several_motors_in_one_file():
    """ThrustCurve bundles are whole catalogues."""
    second = D12_ENG.replace("D12 24 70", "C6 18 70").replace("; Estes D12", "")
    motors = parse_eng(D12_ENG + second)
    assert [m.name for m in motors] == ["D12", "C6"]


def test_a_malformed_header_is_rejected():
    with pytest.raises(ValueError, match="7 fields"):
        parse_eng("D12 24 70\n  0.1 5.0\n  0.2 6.0\n")


def test_an_empty_file_is_rejected():
    with pytest.raises(ValueError, match="No motors"):
        parse_eng("; nothing but a comment\n")


# ----------------------------------------------------------------- RockSim


def test_rse_attributes_are_read():
    motor = parse_rse(K550_RSE)[0]
    assert motor.name == "K550W"
    assert motor.manufacturer == "AeroTech"
    assert motor.diameter_mm == 54.0
    assert motor.declared_isp_s == 204.0


def test_rse_masses_are_grams_and_get_converted():
    """The trap: RockSim writes grams where RASP writes kilograms."""
    motor = parse_rse(K550_RSE)[0]
    assert np.isclose(motor.propellant_mass_kg, 0.797)
    assert np.isclose(motor.total_mass_kg, 1.451)


def test_rse_total_impulse_matches_the_published_class():
    motor = parse_rse(K550_RSE)[0]
    # A K is 1280-2560 N-s. The old table in components.py put this boundary a
    # full letter out, so this pins the figure as well as the letter.
    assert 1280.0 < motor.total_impulse_ns <= 2560.0
    assert motor.impulse_class == "K"


def test_rse_that_is_not_xml_is_rejected():
    with pytest.raises(ValueError, match="valid RockSim XML"):
        parse_rse("D12 24 70 0-3 0.0246 0.0564 Estes\n")


def test_rse_without_a_curve_is_rejected():
    bare = '<engine-database><engine-list><engine code="X"/></engine-list></engine-database>'
    with pytest.raises(ValueError, match="No motors"):
        parse_rse(bare)


# ------------------------------------------------------------ both formats


def test_ignition_point_is_prepended():
    """Neither format writes (0, 0), and without it impulse is overstated."""
    motor = parse_eng(D12_ENG)[0]
    assert motor.times_s[0] == 0.0
    assert motor.thrust_n[0] == 0.0

    rse = parse_rse(K550_RSE)[0]
    assert rse.times_s[0] == 0.0
    assert rse.thrust_n[0] == 0.0


def test_a_curve_already_starting_at_zero_is_left_alone():
    text = D12_ENG.replace("   0.049 2.569", "   0.000 0.000\n   0.049 2.569")
    motor = parse_eng(text)[0]
    assert list(motor.times_s).count(0.0) == 1


def test_implied_isp_is_self_consistent():
    """Mass flow from the implied Isp must integrate back to the load."""
    motor = parse_eng(D12_ENG)[0]
    isp = motor.implied_isp_s
    burned = motor.total_impulse_ns / (isp * 9.80665)
    assert np.isclose(burned, motor.propellant_mass_kg)
    # An Estes black-powder D is around 70-90 s.
    assert 60.0 < isp < 100.0


def test_negative_thrust_is_clamped():
    text = D12_ENG.replace("   0.049 2.569", "   0.049 -5.0")
    assert parse_eng(text)[0].thrust_n.min() >= 0.0


def test_impulse_class_boundaries():
    assert impulse_class(2.5) == "A"
    assert impulse_class(2.51) == "B"
    assert impulse_class(5120.0) == "L"
    assert impulse_class(5120.1) == "M"
    assert impulse_class(0.0) == "-"


def test_an_empty_curve_reports_zero_rather_than_dividing_by_nothing():
    empty = ThrustCurve(name="none")
    assert empty.total_impulse_ns == 0.0
    assert empty.average_thrust_n == 0.0
    assert empty.implied_isp_s == 0.0


# ------------------------------------------------------------------ loading


def test_extension_dispatch(tmp_path):
    eng = tmp_path / "d12.eng"
    eng.write_text(D12_ENG, encoding="utf-8")
    rse = tmp_path / "k550.rse"
    rse.write_text(K550_RSE, encoding="utf-8")

    assert load_thrust_curve(eng).name == "D12"
    assert load_thrust_curve(rse).name == "K550W"


def test_a_mislabelled_xml_file_is_still_read(tmp_path):
    path = tmp_path / "actually_xml.eng"
    path.write_text(K550_RSE, encoding="utf-8")
    assert load_thrust_curve(path).name == "K550W"


def test_asking_for_a_motor_that_is_not_there(tmp_path):
    path = tmp_path / "d12.eng"
    path.write_text(D12_ENG, encoding="utf-8")
    with pytest.raises(IndexError):
        load_thrust_curve(path, index=7)


def test_loading_a_bundle_returns_all_of_them(tmp_path):
    second = D12_ENG.replace("D12 24 70", "C6 18 70").replace("; Estes D12", "")
    path = tmp_path / "estes.eng"
    path.write_text(D12_ENG + second, encoding="utf-8")
    assert len(load_thrust_curves(path)) == 2


# -------------------------------------------------------- into the Motor


def test_importing_sets_up_a_flyable_motor(tmp_path):
    from parametric.standard import basic_rocket

    path = tmp_path / "k550.rse"
    path.write_text(K550_RSE, encoding="utf-8")

    model = basic_rocket()
    motor = model.motors[0]
    curve = motor.import_thrust_curve(path)

    assert np.isclose(motor.get("propellant_mass"), 0.797)
    assert len(motor.curve) == len(curve.times_s)
    assert motor.impulse_class == "K"

    engine = motor.to_engine()          # must not raise
    assert engine is not None


def test_importing_keeps_thrust_propellant_and_isp_consistent(tmp_path):
    """The three-numbers-where-two-would-do trap, closed on import."""
    from parametric.standard import basic_rocket

    path = tmp_path / "d12.eng"
    path.write_text(D12_ENG, encoding="utf-8")

    motor = basic_rocket().motors[0]
    motor.import_thrust_curve(path)

    assert np.isclose(motor.effective_isp_s, motor.get("isp_vac"), rtol=1e-9)
    # isp_consistency is a relative *disagreement*, so agreement is zero.
    assert motor.isp_consistency < 1e-9


def test_importing_replaces_the_previous_curve(tmp_path):
    from parametric.standard import basic_rocket

    path = tmp_path / "d12.eng"
    path.write_text(D12_ENG, encoding="utf-8")

    motor = basic_rocket().motors[0]
    before = motor.total_impulse_ns
    motor.import_thrust_curve(path)
    assert motor.total_impulse_ns < before      # an L replaced by a D
    assert motor.impulse_class == "D"


def test_the_motor_component_agrees_with_the_published_table():
    """One table, not two.

    ``parametric.components`` carried its own copy with every boundary a full
    letter out: it called a 6,030 N-s motor an L when NAR and TRA call it an M.
    """
    from parametric.components import impulse_class as component_class

    for impulse in (2.5, 2.51, 20.0, 20.1, 1280.0, 2560.0, 5120.1, 6030.0):
        assert component_class(impulse) == impulse_class(impulse)

    assert component_class(6030.0) == "M"
