"""Disperse the vehicle on screen, not a placeholder.

The dispersion study called ``trajectory.analysis.dispersion.run_dispersion``
without a case function, and without one that flies the simulator's built-in
default -- 50 kg dry, 150 kg of propellant, a 20 kN motor -- whatever
vehicle is open. The dialog then centred its dry-mass spread on 4.4 kg,
which described neither. The result was filed against the open model's
fingerprint, so a plausible CEP was reported for a rocket that does not
exist.

``ModelCaseRunner`` is the case function. It is a class rather than a
closure because a worker process has to be sent the whole job -- the model
as its serialised dictionary, the settings the nominal flight used, the
aerodynamic table and the solved mass -- and a closure will not pickle.
Every case rebuilds the model from the dictionary, applies the sampled
perturbations through the same ``perturb_simulation`` the library's own
dispersion uses, and flies through the same ``fly_model`` the Run Flight
action does. There is one launch sequence, and the dispersion disperses it.
"""

from __future__ import annotations

from dataclasses import replace

from parametric.flight import FlightSettings, fly_model
from parametric.model import VehicleModel

#: Sampled keys that perturb the vehicle rather than the launch. The rest
#: are launch conditions and travel through the flight settings.
VEHICLE_KEYS = (
    "impulse_scale", "thrust_scale", "dry_mass_kg", "aero_scale",
    "thrust_tilt_x_deg", "thrust_tilt_z_deg", "cg_offset_x_m", "cg_offset_z_m",
    "fin_cant_offset_deg", "turbulence_seed",
)


class ModelCaseRunner:
    """Fly one dispersed case of a parametric model; picklable for workers."""

    def __init__(self, model: VehicleModel, settings: FlightSettings,
                 database=None, solved=None):
        self.model_data = model.to_dict()
        # A coupled rebuild of the drag table costs three aerodynamic sweeps
        # per flight, which a batch of fifty cannot afford. The table handed
        # in should already be the coupled one from the nominal flight.
        self.settings = replace(settings, couple_aero_altitude=False)
        self.database = database
        self.solved = solved

    def settings_for(self, params: dict) -> FlightSettings:
        """The nominal settings with this case's launch conditions applied."""
        base = self.settings
        return replace(
            base,
            elevation_deg=float(params.get("launch_elevation_deg", base.elevation_deg)),
            azimuth_deg=float(params.get("launch_azimuth_deg", base.azimuth_deg)),
            wind_speed_mps=float(params.get("wind_speed_mps", base.wind_speed_mps)),
            wind_direction_deg=float(
                params.get("wind_direction_deg", base.wind_direction_deg)
            ),
            dt_s=float(params.get("dt", base.dt_s)),
        )

    def __call__(self, params: dict) -> dict:
        from trajectory.analysis.dispersion import perturb_simulation, reduce_outcome

        model = VehicleModel.from_dict(self.model_data)
        settings = self.settings_for(params)
        vehicle = {key: params[key] for key in VEHICLE_KEYS if key in params}
        t_max = params.get("t_max")
        outcome = fly_model(
            model, settings, self.database, self.solved,
            perturb=lambda sim: perturb_simulation(sim, vehicle),
            t_max=None if t_max is None else float(t_max),
        )
        return reduce_outcome(outcome.result, params,
                              ground_m=settings.pad_altitude_m)


def dispersions_about(
    settings: FlightSettings,
    dry_mass_kg: float,
    *,
    impulse_sd: float = 0.02,
    dry_mass_sd_kg: float = 0.0,
    aero_sd: float = 0.08,
    elevation_sd_deg: float = 0.5,
    azimuth_sd_deg: float = 2.0,
    wind_speed_sd_mps: float = 2.5,
    wind_direction_sd_deg: float = 60.0,
    thrust_misalignment_sd_deg: float = 0.0,
    cg_offset_sd_m: float = 0.0,
    fin_cant_sd_deg: float = 0.0,
) -> dict[str, tuple[float, float, float, float]]:
    """``(mean, std, low, high)`` per parameter, centred on a nominal flight.

    The means are the flight settings and the vehicle's own dry mass, so the
    study disperses the flight that was set up rather than a fixed 85
    degrees, no wind and 4.4 kg. Bounds are four sigma, clipped where the
    physics has an opinion: a rail cannot be set past vertical, a wind
    cannot blow at a negative speed and a vehicle cannot weigh nothing. A
    zero spread is nudged to a hair above zero because the sampler divides
    by it.

    The build imperfections -- thrust misalignment, CG offset, fin cant --
    are centred on zero and *added* to whatever the flight settings
    already carry. Each is sampled as two orthogonal components, so its
    magnitude is Rayleigh-distributed and its direction uniform, which is
    what a tolerance on a machined part looks like. A zero spread leaves
    the key out entirely.
    """
    def sd(value: float) -> float:
        return max(float(value), 1e-6)

    def centred(spread: float) -> tuple[float, float, float, float]:
        return (0.0, float(spread), -4.0 * float(spread), 4.0 * float(spread))

    dry = float(dry_mass_kg)
    elevation = float(settings.elevation_deg)
    azimuth = float(settings.azimuth_deg)
    speed = float(settings.wind_speed_mps)
    bearing = float(settings.wind_direction_deg)

    impulse, mass, aero = sd(impulse_sd), sd(dry_mass_sd_kg), sd(aero_sd)
    tilt, turn = sd(elevation_sd_deg), sd(azimuth_sd_deg)
    gust, veer = sd(wind_speed_sd_mps), min(4.0 * sd(wind_direction_sd_deg), 180.0)

    mass_low = max(dry - 4.0 * mass, 0.05 * dry)
    return {
        "impulse_scale": (1.0, impulse, max(1.0 - 4.0 * impulse, 0.5), 1.0 + 4.0 * impulse),
        "dry_mass_kg": (dry, mass, mass_low, max(dry + 4.0 * mass, mass_low + 1e-6)),
        "aero_scale": (1.0, aero, max(1.0 - 4.0 * aero, 0.25), 1.0 + 4.0 * aero),
        # Clipped to [1, 90] without ever crossing the mean: an elevation
        # below 1 deg used to clamp the low bound above the high one.
        "launch_elevation_deg": (
            elevation, tilt,
            min(max(elevation - 4.0 * tilt, 1.0), elevation),
            max(min(elevation + 4.0 * tilt, 90.0), elevation + 1e-6),
        ),
        "launch_azimuth_deg": (azimuth, turn, azimuth - 4.0 * turn, azimuth + 4.0 * turn),
        "wind_speed_mps": (speed, gust, max(speed - 4.0 * gust, 0.0), speed + 4.0 * gust),
        "wind_direction_deg": (
            bearing, sd(wind_direction_sd_deg), bearing - veer, bearing + veer,
        ),
        **({"thrust_tilt_x_deg": centred(thrust_misalignment_sd_deg),
            "thrust_tilt_z_deg": centred(thrust_misalignment_sd_deg)}
           if thrust_misalignment_sd_deg > 0.0 else {}),
        **({"cg_offset_x_m": centred(cg_offset_sd_m),
            "cg_offset_z_m": centred(cg_offset_sd_m)}
           if cg_offset_sd_m > 0.0 else {}),
        **({"fin_cant_offset_deg": centred(fin_cant_sd_deg)}
           if fin_cant_sd_deg > 0.0 else {}),
        # Every case flies its own turbulence field.
        **({"turbulence_seed": ("uniform", 0.0, 2.0 ** 31 - 1.0)}
           if getattr(settings, "turbulence", "none") != "none" else {}),
    }
