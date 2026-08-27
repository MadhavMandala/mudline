"""Undo and redo, by snapshot.

Two ways to build this. A command pattern records a do/undo pair for every kind
of edit -- precise, small in memory, and requiring that *every* mutation site
remember to participate. Miss one and the stack silently desynchronises from
the model, which is worse than having no undo at all, because the user trusts
it.

A snapshot stack instead captures the whole model after each edit. It costs
memory, and for a rocket -- a few dozen components serialising to a handful of
kilobytes -- that is not a real constraint. What it buys is that it cannot miss
an edit: anything that changes the model changes the snapshot, including
features added later that nobody remembered to teach about undo.

The model already round-trips through ``to_dict``/``from_dict``, so the
snapshot is exactly the thing that is already known to reconstruct a vehicle
faithfully.

Coalescing
----------
Dragging a slider emits a change on every step. Without coalescing, one drag
becomes two hundred undo entries and Ctrl-Z crawls back a pixel at a time. Edits
that carry the same label in quick succession collapse into one entry, so a
drag undoes as a single action -- which is how the user thinks of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


#: Same-labelled edits closer together than this collapse into one entry.
COALESCE_SECONDS = 0.7

#: How far back you can go. Deep enough to cover a working session, bounded so
#: a long one cannot grow without limit.
DEFAULT_LIMIT = 120


@dataclass
class Snapshot:
    """One recorded state of the model."""

    state: dict
    label: str
    stamp: float
    #: Path of the component selected when this was taken, so undo can put the
    #: selection back where the user left it rather than collapsing it.
    selection: str | None = None


class UndoStack:
    """A bounded history of model states."""

    def __init__(self, limit: int = DEFAULT_LIMIT):
        self.limit = limit
        self._entries: list[Snapshot] = []
        self._index = -1
        #: Set while an undo or redo is being applied, so the resulting model
        #: changes do not push themselves back onto the stack.
        self.applying = False

    # ------------------------------------------------------------------

    def reset(self, state: dict, selection: str | None = None) -> None:
        """Start a new history from this state, as when a document is opened."""
        self._entries = [Snapshot(state, "open", time.monotonic(), selection)]
        self._index = 0

    def push(self, state: dict, label: str, selection: str | None = None) -> None:
        """Record a state reached by an edit."""
        if self.applying:
            return

        now = time.monotonic()
        # Redoable future is discarded the moment a new edit is made, which is
        # what every editor does and what users expect.
        del self._entries[self._index + 1:]

        top = self._entries[self._index] if self._index >= 0 else None
        if (
            top is not None
            and top.label == label
            and now - top.stamp < COALESCE_SECONDS
            and self._index > 0
        ):
            # Same action continuing: replace rather than append, so a slider
            # drag is one undo step.
            self._entries[self._index] = Snapshot(state, label, now, selection)
            return

        self._entries.append(Snapshot(state, label, now, selection))
        self._index += 1

        if len(self._entries) > self.limit:
            trim = len(self._entries) - self.limit
            del self._entries[:trim]
            self._index -= trim

    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._entries) - 1

    @property
    def undo_label(self) -> str:
        if not self.can_undo:
            return ""
        return self._entries[self._index].label

    @property
    def redo_label(self) -> str:
        if not self.can_redo:
            return ""
        return self._entries[self._index + 1].label

    def undo(self) -> Snapshot | None:
        if not self.can_undo:
            return None
        self._index -= 1
        return self._entries[self._index]

    def redo(self) -> Snapshot | None:
        if not self.can_redo:
            return None
        self._index += 1
        return self._entries[self._index]

    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self._entries)

    @property
    def position(self) -> int:
        return self._index

    def labels(self) -> list[str]:
        return [entry.label for entry in self._entries]
