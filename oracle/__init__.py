"""Ground truth extracted from RASAero II, for validating a reimplementation.

RASAero's own Tools > Run Test emits the per-term aerodynamic breakdown at
0.01 Mach steps. That breakdown -- not the CSV export's totals -- is what
makes a port debuggable: a wrong total says something is wrong, a wrong term
says what.
"""

from .runtest import Dump, Event, parse_dump
from .vehicles import Fin, Vehicle, test_matrix, write_cdx1

__all__ = ["Dump", "Event", "parse_dump", "Fin", "Vehicle", "test_matrix", "write_cdx1"]
