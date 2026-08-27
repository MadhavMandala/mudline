"""Simulation core."""

from .integrator import TrajectoryIntegrator
from .events import EventDetector, apogee_event, ground_impact
from .launch_rail import LaunchRail, alignment_quaternion, rail_direction

__all__ = [
    "TrajectoryIntegrator",
    "EventDetector",
    "apogee_event",
    "ground_impact",
    "LaunchRail",
    "alignment_quaternion",
    "rail_direction",
]
