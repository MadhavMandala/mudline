"""Beyond the table: axial force recovered, and a bounded high-alpha model.

Runs under pytest, and standalone via
``python trajectory/tests/test_high_alpha.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.aero_model import (  # noqa: E402
    FLAT_PLATE_CN_90,
    HIGH_ALPHA_BLEND_DEG,
    STALL_ONSET_DEG,
    HighAlphaGeometry,
    RasaeroAeroModel,
    crossflow_drag_coefficient,
    crossflow_proportionality,
)

BODY_AXIS = np.array([0.0, 1.0, 0.0])
S_REF = 0.01
DIAMETER = float(np.sqrt(4.0 * S_REF / np.pi))
CA, CN_ALPHA, X_CP = 0.4, 4.0, 1.3
EDGE_DEG = 10.0
CG = np.array([0.0, -1.0, 0.0])

GEOMETRY = HighAlphaGeometry(
    length_m=2.0, diameter_m=DIAMETER, planform_area_m2=0.20, planform_centroid_m=1.1,
    nose_length_m=0.4, fin_area_m2=0.02, fin_centroid_m=1.7,
)


def _table(power_on: bool = False) -> AeroDatabase:
    """Constant CA, linear CN, the drag column assembled the way the engine does."""
    rows = []
    for mach in (0.0, 1.0, 3.0):
        for alpha in np.linspace(0.0, EDGE_DEG, 6):
            a = np.radians(alpha)
            cn = CN_ALPHA * a
            cd = CA * np.cos(a) + cn * np.sin(a)
            rows.append(AeroCoefficients(
                mach=mach, alpha_deg=float(alpha), cd=float(cd), cn=float(cn), cm=0.0,
                x_cp_m=X_CP, cd_power_on=(0.5 * cd if power_on else None),
            ))
    db = AeroDatabase(rows, reference_length_m=2.0)
    db.high_alpha = GEOMETRY
    return db


def _model(db: AeroDatabase | None = None) -> RasaeroAeroModel:
    return RasaeroAeroModel(db or _table(), reference_area_m2=S_REF,
                            reference_length_m=2.0, body_axis=BODY_AXIS)


def _flow(alpha_deg: float, speed: float = 100.0) -> np.ndarray:
    a = np.radians(alpha_deg)
    return np.array([speed * np.sin(a), speed * np.cos(a), 0.0])


# ------------------------------------------------------- inside the table


def test_the_axial_force_is_the_axial_coefficient_not_the_wind_drag():
    """CD along the axis counted the induced drag twice."""
    model = _model()
    forces = model.forces_and_moments(_flow(8.0), 1.2, 340.0, CG)
    assert np.isclose(forces.ca_applied, CA, atol=1e-9)
    axial = -float(np.dot(forces.force_body_n, BODY_AXIS))
    assert np.isclose(axial, 0.5 * 1.2 * 100.0 ** 2 * S_REF * CA)
    assert np.isclose(forces.cn_applied, CN_ALPHA * np.radians(8.0))


def test_power_on_uses_the_powered_column_for_the_axial_force():
    model = _model(_table(power_on=True))
    off = model.coefficients_at(1.0, 5.0)[0]
    on = model.coefficients_at(1.0, 5.0, power_on=True)[0]
    assert on < off


# -------------------------------------------------------- past the edge


def test_the_table_is_extrapolated_on_its_slope_before_stall():
    model = _model()
    for alpha in (EDGE_DEG + 1.0, EDGE_DEG + 4.0):
        _, cn, x_cp = model.coefficients_at(0.0, alpha)
        assert np.isclose(cn, CN_ALPHA * np.radians(alpha), rtol=1e-6)
        assert np.isclose(x_cp, X_CP)


def test_the_forces_are_continuous_across_edge_and_blend():
    model = _model()
    for boundary in (EDGE_DEG, STALL_ONSET_DEG, STALL_ONSET_DEG + HIGH_ALPHA_BLEND_DEG):
        below = model.forces_and_moments(_flow(boundary - 0.01), 1.2, 340.0, CG)
        above = model.forces_and_moments(_flow(boundary + 0.01), 1.2, 340.0, CG)
        assert np.allclose(below.force_body_n, above.force_body_n, rtol=2e-2), boundary
        assert np.allclose(below.static_moment_body_nm, above.static_moment_body_nm,
                           rtol=2e-2, atol=1e-6), boundary


def test_the_normal_force_keeps_growing_to_the_flat_plate_value():
    model = _model()
    cn = [model.coefficients_at(0.0, a)[1] for a in (10.0, 20.0, 45.0, 90.0)]
    assert cn[0] < cn[1] < cn[2] < cn[3]
    eta = crossflow_proportionality(GEOMETRY.length_m / GEOMETRY.diameter_m)
    expected = (
        eta * crossflow_drag_coefficient(0.0) * GEOMETRY.planform_area_m2 / S_REF
        + FLAT_PLATE_CN_90 * GEOMETRY.fin_area_m2 / S_REF
    )
    assert np.isclose(cn[3], expected, rtol=1e-6)


def test_the_axial_force_vanishes_broadside():
    model = _model()
    ca, _, _ = model.coefficients_at(0.0, 90.0)
    assert abs(ca) < 1e-9


def test_the_cp_moves_to_the_planform_broadside():
    model = _model()
    _, cn, x_cp = model.coefficients_at(0.0, 90.0)
    eta = crossflow_proportionality(GEOMETRY.length_m / GEOMETRY.diameter_m)
    body = eta * crossflow_drag_coefficient(0.0) * GEOMETRY.planform_area_m2 / S_REF
    fins = FLAT_PLATE_CN_90 * GEOMETRY.fin_area_m2 / S_REF
    expected = (body * GEOMETRY.planform_centroid_m + fins * GEOMETRY.fin_centroid_m) / (body + fins)
    assert np.isclose(x_cp, expected, rtol=1e-6)
    assert GEOMETRY.planform_centroid_m < x_cp < GEOMETRY.fin_centroid_m


def test_crossflow_drag_rises_through_the_transonic_crossflow():
    model = _model()
    subsonic = model.coefficients_at(0.5, 90.0)[1]     # crossflow Mach 0.5
    transonic = model.coefficients_at(1.0, 90.0)[1]    # crossflow Mach 1.0
    assert transonic > subsonic


def test_a_table_without_geometry_is_treated_as_a_cylinder():
    db = _table()
    db.high_alpha = None
    model = _model(db)
    assert model.high_alpha.fin_area_m2 == 0.0
    assert np.isclose(model.high_alpha.length_m, 2.0)
    forces = model.forces_and_moments(_flow(90.0), 1.2, 340.0, CG)
    assert np.all(np.isfinite(forces.force_body_n))
    assert model.coefficients_at(0.0, 90.0)[1] > model.coefficients_at(0.0, 10.0)[1]


def test_a_stable_vehicle_stays_restoring_broadside():
    """CP still aft of a forward CG at 90 degrees: the moment turns the nose back."""
    model = _model()
    v_body = _flow(80.0)
    forces = model.forces_and_moments(v_body, 1.2, 340.0, CG)
    assert np.dot(forces.static_moment_body_nm, np.cross(BODY_AXIS, v_body)) > 0.0
    assert forces.static_margin_m > 0.0


def test_a_table_covering_ninety_degrees_needs_no_extension():
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=0.4, cn=0.05 * a, cm=0.0, x_cp_m=1.3)
        for m in (0.0, 1.0) for a in (0.0, 45.0, 90.0)
    ]
    model = _model(AeroDatabase(rows, reference_length_m=2.0))
    _, cn, x_cp = model.coefficients_at(0.0, 90.0)
    assert np.isclose(cn, 4.5) and np.isclose(x_cp, 1.3)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
