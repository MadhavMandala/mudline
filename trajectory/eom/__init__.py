"""Equations of motion."""

from .translational import TranslationalEOM
from .rotational import RotationalEOM
from .kinematics import (
    euler_to_quat,
    propagate_quaternion,
    quat_to_dcm,
    quaternion_derivative,
    quaternion_rate_matrix,
)

__all__ = [
    "TranslationalEOM",
    "RotationalEOM",
    "euler_to_quat",
    "propagate_quaternion",
    "quat_to_dcm",
    "quaternion_derivative",
    "quaternion_rate_matrix",
]
