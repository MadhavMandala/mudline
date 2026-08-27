"""Statistical analysis of trajectory dispersions."""

import numpy as np


def compute_cepxy(landing_points: np.ndarray) -> tuple:
    """
    Compute Circular Error Probable in X-Y (downrange/crossrange).

    Args:
        landing_points: Array of shape (n, 2) with [x, z] positions

    Returns:
        (CEP, mean_pos, std_x, std_z)
    """
    if len(landing_points) < 2:
        return 0.0, landing_points[0] if len(landing_points) else np.zeros(2), 0.0, 0.0

    mean_pos = np.mean(landing_points, axis=0)
    # Sample standard deviation, the same estimator the landing ellipse's
    # covariance uses; the population form ran 5% low on a ten-case batch.
    std = np.std(landing_points, axis=0, ddof=1)

    # CEP approximation for a roughly circular normal distribution.
    cep = 0.5887 * (std[0] + std[1])

    return cep, mean_pos, std[0], std[1]


def landing_ellipse(landing_points: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute landing ellipse parameters.

    Args:
        landing_points: Array of shape (n, 2) with [x, z] positions
        confidence: Confidence level (0-1)

    Returns:
        (center, semi_major, semi_minor, orientation)
    """
    center = np.mean(landing_points, axis=0)
    cov = np.cov(landing_points.T)

    # Eigenvalues and eigenvectors for ellipse axes. eigh, not eig: the
    # covariance is symmetric, and eigh's contract is real output. eig may
    # return complex dtype with zero imaginary parts (numpy 2.5 always
    # does), and arctan2 below refuses complex input.
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending; the ellipse wants major first.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Chi-squared factor for confidence level
    from scipy.stats import chi2
    chi2_val = chi2.ppf(confidence, df=2)

    # Rounding can leave a zero-variance eigenvalue at -1e-30, whose square
    # root is NaN rather than a point.
    semi_major = np.sqrt(max(chi2_val * eigvals[0], 0.0))
    semi_minor = np.sqrt(max(chi2_val * eigvals[1], 0.0))
    # An eigenvector's sign is arbitrary and an ellipse has no front, so the
    # orientation is taken modulo a half turn rather than reported 180 deg
    # off whenever eigh happens to flip it.
    orientation = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) % np.pi)

    return center, semi_major, semi_minor, orientation


def flight_statistics(states: np.ndarray, times: np.ndarray) -> dict:
    """Compute statistics from a single trajectory.

    Args:
        states: (n, >=6) state history, one row per sample. Column layout is
            the simulator's: [x, y, z, vx, vy, vz, ...]. There is no time
            column -- ``solve_ivp`` returns time separately as ``result.t``.
        times: (n,) sample times [s], i.e. ``result.t``.

    ``times`` is a required argument on purpose. The previous signature took
    only the state array and read column 0 as the time base, but column 0 is
    downrange position ``x``. For a nominal vertical flight x stays 0, so
    ``apogee_time`` and ``flight_time`` both reported 0.0 and looked plausible
    enough to go unnoticed. Making the caller supply the real time base means
    the mistake cannot recur silently.
    """
    states = np.asarray(states, dtype=float)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or states.shape[1] < 6:
        raise ValueError(f"states must be (n, >=6); got {states.shape}")
    if times.shape != (len(states),):
        raise ValueError(
            f"times must be one value per state row: expected {(len(states),)}, "
            f"got {times.shape}"
        )

    altitudes = states[:, 1]
    apogee_idx = int(np.argmax(altitudes))
    return {
        "max_altitude": float(altitudes[apogee_idx]),
        "max_velocity": float(np.max(np.linalg.norm(states[:, 3:6], axis=1))),
        "apogee_time": float(times[apogee_idx]),
        "flight_time": float(times[-1] - times[0]),
        "range": float(np.linalg.norm(states[-1, [0, 2]] - states[0, [0, 2]])),
    }
