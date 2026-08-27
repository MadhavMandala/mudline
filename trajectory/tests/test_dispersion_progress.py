"""Tests for reporting progress out of a dispersion batch, and stopping one.

A study is hundreds of independent flights and takes minutes. Before this the
caller got nothing until the last case landed and had no way to change its
mind, so the application simply stopped repainting -- which is indistinguishable
from a hang, and was reported as one.

Single-process throughout: the behaviour under test is the callback contract,
and spawning workers to check it would make the test slow and platform-shaped
for no extra coverage. The parallel path shares the same loop.

Runs under pytest, and standalone via
``python -m pytest trajectory/tests/test_dispersion_progress.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.analysis.monte_carlo import MonteCarlo  # noqa: E402

SPEC = {"thrust_scale": (1.0, 0.03, 0.9, 1.1)}


def _counting_sim(params: dict) -> dict:
    """A flight, reduced to the fields the reduction step reads.

    Landing points are spread with the sampled parameter so the batch has a
    real spread to compute statistics from rather than a single repeated point.
    """
    scale = params["thrust_scale"]
    return {
        "success": True,
        "landed": True,
        "thrust_scale": scale,
        "landing_east_m": 1000.0 * scale,
        "landing_north_m": -500.0 * scale,
        "max_altitude": 4000.0 * scale,
        "max_velocity": 300.0 * scale,
        "flight_time": 60.0,
        "landing_speed_mps": 7.0,
    }


def test_progress_is_reported_once_per_case():
    seen: list[tuple[int, int]] = []
    monte_carlo = MonteCarlo(sim_func=_counting_sim, seed=1)

    results = monte_carlo.run_batch(
        n_samples=6, param_distributions=SPEC,
        progress=lambda done, total: seen.append((done, total)) is None,
    )

    assert len(results) == 6
    assert seen == [(i, 6) for i in range(1, 7)]


def test_returning_false_stops_the_batch():
    monte_carlo = MonteCarlo(sim_func=_counting_sim, seed=1)

    results = monte_carlo.run_batch(
        n_samples=20, param_distributions=SPEC,
        progress=lambda done, total: done < 3,
    )

    assert len(results) == 3


def test_without_a_callback_the_batch_is_unchanged():
    """The old path stays the old path -- no callback, no per-case overhead."""
    monte_carlo = MonteCarlo(sim_func=_counting_sim, seed=1)
    assert len(monte_carlo.run_batch(n_samples=4, param_distributions=SPEC)) == 4


def test_the_same_seed_gives_the_same_cases_either_way():
    """Reporting progress must not perturb the study it is reporting on.

    ``imap`` rather than ``imap_unordered`` for exactly this reason: a case is
    identified by its position in the batch, so a run that reordered them
    would not be reproducible, and a dispersion nobody can reproduce is not
    evidence of anything.
    """
    plain = MonteCarlo(sim_func=_counting_sim, seed=99).run_batch(
        n_samples=8, param_distributions=SPEC
    )
    watched = MonteCarlo(sim_func=_counting_sim, seed=99).run_batch(
        n_samples=8, param_distributions=SPEC, progress=lambda done, total: True
    )
    assert [c["thrust_scale"] for c in plain] == [c["thrust_scale"] for c in watched]


def test_a_stopped_study_is_reduced_over_the_cases_that_flew():
    """Cancelling gives a narrower study, not a failure and not a lie.

    The statistics are honest -- they are simply over fewer samples, and
    ``n_cases`` says how many, which is what the window reports back.
    """
    from trajectory.analysis.dispersion import run_dispersion

    result = run_dispersion(
        n_cases=20, dispersions=SPEC, seed=1, case_fn=_counting_sim,
        progress=lambda done, total: done < 5,
    )
    assert result.n_cases == 5


def test_a_study_of_no_cases_says_so_plainly():
    """Better than the reduction step's "nothing reached the ground"."""
    from trajectory.analysis.dispersion import run_dispersion

    with pytest.raises(RuntimeError, match="stopped"):
        run_dispersion(
            n_cases=0, dispersions=SPEC, seed=1, case_fn=_counting_sim,
            progress=lambda done, total: True,
        )


def test_a_worker_refuses_to_start_workers_of_its_own():
    """The unguarded-script trap, caught at the point it would multiply.

    Simulated rather than actually spawned: reproducing it for real means
    starting the process explosion this exists to prevent.
    """
    from unittest.mock import patch

    from trajectory.analysis import monte_carlo

    with patch.object(monte_carlo.multiprocessing, "parent_process",
                      return_value=object()):
        with pytest.raises(RuntimeError, match="main guard"):
            MonteCarlo(sim_func=_counting_sim, seed=1).run_batch(
                n_samples=4, param_distributions=SPEC, n_processes=2
            )


def test_a_worker_running_serially_is_left_alone():
    """One process is not a pool, so there is nothing to nest."""
    from unittest.mock import patch

    from trajectory.analysis import monte_carlo

    with patch.object(monte_carlo.multiprocessing, "parent_process",
                      return_value=object()):
        assert len(MonteCarlo(sim_func=_counting_sim, seed=1).run_batch(
            n_samples=3, param_distributions=SPEC, n_processes=1
        )) == 3
