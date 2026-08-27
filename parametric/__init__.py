"""Parametric vehicle geometry, built the way OpenVSP builds it.

A vehicle is a tree of components, and a body is an ordered stack of
cross-sections rather than a set of named primitives. That single choice
removes the special cases: an ogive nose, a conical transition, a boattail and
a payload bulge are the same object with different sections.

    VehicleModel
     |- Stack "airframe"      lofted from XSecs -> B-rep solid
     |   |- FinSet "fins"     attached at a station on the parent
     |   '- Motor "motor"     propellant and thrust
     '- PointMass "avionics"

Everything numeric is a Parm with bounds and change tracking, so edits
propagate, only dirty geometry rebuilds, and a design variable or optimiser has
something stable to address.
"""

from parametric.components import Component, FinSet, Motor, PointMass, Stack
from parametric.model import FORMAT, MassSummary, VehicleModel
from parametric.parm import Parm, ParmContainer
from parametric.xsec import (
    MIN_TIP_RADIUS_M,
    NoseProfile,
    XSec,
    XSecShape,
    generate_nose_sections,
    generate_transition_sections,
    generate_tube_sections,
    nose_radius,
)

__all__ = [
    "FORMAT",
    "MIN_TIP_RADIUS_M",
    "Component",
    "FinSet",
    "MassSummary",
    "Motor",
    "NoseProfile",
    "Parm",
    "ParmContainer",
    "PointMass",
    "Stack",
    "VehicleModel",
    "XSec",
    "XSecShape",
    "generate_nose_sections",
    "generate_transition_sections",
    "generate_tube_sections",
    "nose_radius",
]
