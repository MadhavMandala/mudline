"""Monte Carlo dispersion analysis."""

import multiprocessing
import numpy as np
from multiprocessing import Pool
from typing import Callable


def _refuse_to_nest(n_processes: int) -> None:
    """Stop a worker starting a pool of its own.

    Windows spawns rather than forks, so every worker re-imports the module it
    was started from. A script that calls into here without wrapping its body
    in ``if __name__ == "__main__":`` therefore re-runs that whole script
    inside each worker -- and when the re-run reaches this function again, each
    worker starts its own pool. The machine fills with processes running the
    caller's entire program over and over, and because each pass does real work
    first it looks like the study is merely slow rather than multiplying.

    Python's own bootstrap check does eventually fire, but only after every
    worker has re-executed everything ahead of the call. Catching it here turns
    minutes of silent thrashing into one sentence naming the fix.
    """
    if n_processes <= 1 or multiprocessing.parent_process() is None:
        return
    raise RuntimeError(
        "A dispersion worker tried to start workers of its own, which means "
        "the script that called it re-ran inside the worker.\n\n"
        "On Windows every worker re-imports the calling module, so a script "
        "that runs a study must keep it behind a main guard:\n\n"
        "    if __name__ == "
        '"__main__":\n'
        "        result = run_dispersion(...)\n\n"
        "Without the guard the study re-runs in every worker. Pass "
        "n_processes=1 to sidestep the pool entirely."
    )


class MonteCarlo:
    """Monte Carlo dispersion analysis runner."""

    def __init__(self, sim_func: Callable, seed: int = None):
        """
        Args:
            sim_func: Function that runs one trajectory and returns results
            seed: Random seed for reproducibility
        """
        self.sim_func = sim_func
        self.rng = np.random.default_rng(seed)

    def sample_truncated_normal(self, mean: float, std: float,
                                 low: float, high: float, n: int) -> np.ndarray:
        """Generate truncated normal samples."""
        from scipy.stats import truncnorm
        if high <= low:
            raise ValueError(
                f"truncation bounds must satisfy low < high, got [{low}, {high}]"
            )
        # A zero spread is a fixed value, not a division by zero.
        std = max(float(std), 1e-12)
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=n,
                            random_state=self.rng)

    def run_batch(self, n_samples: int, param_distributions: dict,
                  n_processes: int = 1, progress: Callable | None = None) -> list:
        """
        Run Monte Carlo batch.

        Args:
            n_samples: Number of trajectories to simulate
            param_distributions: Dict of {param_name: (mean, std, low, high)}
                for a truncated normal, or ``("uniform", low, high)`` for a
                uniform draw -- a seed, a clock angle.
            n_processes: Number of parallel processes (1 for sequential)
            progress: Called ``progress(done, total)`` as each case lands.
                Return False to stop early; the cases already finished are
                returned. A batch of a few hundred is minutes of work, and
                without this the caller has nothing to show for it and no way
                to change its mind.

        Returns:
            List of results from sim_func. Shorter than ``n_samples`` if
            ``progress`` asked to stop.
        """
        # Sample parameters
        samples = {}
        for param, spec in param_distributions.items():
            if isinstance(spec[0], str) and spec[0] == "uniform":
                _, low, high = spec
                samples[param] = self.rng.uniform(float(low), float(high), n_samples)
                continue
            mean, std, low, high = spec
            samples[param] = self.sample_truncated_normal(
                mean, std, low, high, n_samples
            )

        # Build parameter sets
        param_sets = [
            {p: samples[p][i] for p in samples}
            for i in range(n_samples)
        ]

        _refuse_to_nest(n_processes)

        if progress is None:
            # Unchanged path: map is the cheapest thing that works.
            if n_processes > 1:
                with Pool(n_processes) as pool:
                    return pool.map(self.sim_func, param_sets)
            return [self.sim_func(ps) for ps in param_sets]

        results: list = []
        if n_processes > 1:
            # imap rather than map so a result is available as soon as its
            # worker finishes, instead of after the last one. Unordered would
            # be marginally faster still, but a dispersion case is identified
            # by its position in the batch and shuffling them would make a
            # run unreproducible.
            with Pool(n_processes) as pool:
                iterator = pool.imap(self.sim_func, param_sets)
                for outcome in iterator:
                    results.append(outcome)
                    if progress(len(results), n_samples) is False:
                        # terminate, not close: close waits for the cases
                        # already dispatched, which is the wait being escaped.
                        pool.terminate()
                        break
        else:
            for parameters in param_sets:
                results.append(self.sim_func(parameters))
                if progress(len(results), n_samples) is False:
                    break

        return results

    def summarize(self, results: list, metrics: list) -> dict:
        """Summarize results by metrics."""
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p95": np.percentile(values, 95)
                }
        return summary
