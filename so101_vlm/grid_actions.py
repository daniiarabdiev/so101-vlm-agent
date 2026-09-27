"""Task-independent categorical grid geometry and episode state.

The adapter converts model-selected categories to metric targets.  It never
reads object or destination coordinates; privileged target selection belongs
in :mod:`grid_oracle` only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GridQuestion:
    kind: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class GridSpec:
    xy_min: tuple[float, float] = (.19, -.14)
    xy_max: tuple[float, float] = (.285, .14)
    columns: int = 3
    rows: int = 7
    fine_size: int = 3

    def __post_init__(self):
        low = np.asarray(self.xy_min, dtype=float)
        high = np.asarray(self.xy_max, dtype=float)
        if low.shape != (2,) or high.shape != (2,) or not np.isfinite([*low, *high]).all():
            raise ValueError("grid bounds must be finite XY pairs")
        if np.any(high <= low):
            raise ValueError("grid maximum must exceed minimum")
        if not 1 <= self.columns <= 26 or self.rows < 1 or self.rows > 9:
            raise ValueError("coarse labels require 1..26 columns and 1..9 rows")
        if self.fine_size != 3:
            raise ValueError("run 2 requires a 3x3 fine grid")
        object.__setattr__(self, "xy_min", tuple(map(float, low)))
        object.__setattr__(self, "xy_max", tuple(map(float, high)))

    @property
    def column_labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + index) for index in range(self.columns))

    @property
    def row_labels(self) -> tuple[str, ...]:
        return tuple(str(index + 1) for index in range(self.rows))

    @property
    def fine_labels(self) -> tuple[str, ...]:
        return tuple(str(index + 1) for index in range(self.fine_size**2))

    @property
    def coarse_size(self) -> np.ndarray:
        return (np.asarray(self.xy_max) - np.asarray(self.xy_min)) / [self.columns, self.rows]

    def _coarse_indices(self, column: str, row: str) -> tuple[int, int]:
        if column not in self.column_labels or row not in self.row_labels:
            raise ValueError(f"unknown coarse cell {column}{row}")
        return self.column_labels.index(column), self.row_labels.index(row)

    def coarse_bounds(self, column: str, row: str) -> tuple[np.ndarray, np.ndarray]:
        column_index, display_row = self._coarse_indices(column, row)
        # Row 1 is at the top of the overhead image/world +Y.
        y_index = self.rows - 1 - display_row
        low = np.asarray(self.xy_min) + self.coarse_size * [column_index, y_index]
        return low, low + self.coarse_size

    def coarse_for_point(self, point) -> tuple[str, str]:
        point = np.asarray(point, dtype=float)
        low, high = np.asarray(self.xy_min), np.asarray(self.xy_max)
        if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < low) or np.any(point > high):
            raise ValueError("point outside grid")
        fraction = (point - low) / (high - low)
        column = min(int(fraction[0] * self.columns), self.columns - 1)
        y_index = min(int(fraction[1] * self.rows), self.rows - 1)
        display_row = self.rows - 1 - y_index
        return self.column_labels[column], self.row_labels[display_row]

    def fine_for_point(self, column: str, row: str, point) -> str:
        point = np.asarray(point, dtype=float)
        low, high = self.coarse_bounds(column, row)
        if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < low-1e-12) or np.any(point > high+1e-12):
            raise ValueError("point outside selected coarse cell")
        fraction = np.clip((point - low) / (high - low), 0., 1.)
        x_index = min(int(fraction[0] * self.fine_size), self.fine_size - 1)
        y_index = min(int(fraction[1] * self.fine_size), self.fine_size - 1)
        display_row = self.fine_size - 1 - y_index
        return str(display_row * self.fine_size + x_index + 1)

    def target_xy(self, column: str, row: str, fine: str) -> np.ndarray:
        if fine not in self.fine_labels:
            raise ValueError(f"unknown fine cell {fine}")
        low, high = self.coarse_bounds(column, row)
        fine_index = int(fine) - 1
        display_row, x_index = divmod(fine_index, self.fine_size)
        y_index = self.fine_size - 1 - display_row
        size = (high - low) / self.fine_size
        return low + size * ([x_index, y_index] + np.full(2, .5))


class GridAdapter:
    """State machine driven only by selected actions and actuator feedback."""

    MACRO_LABELS = tuple("ABCDEF")

    def __init__(self, spec: GridSpec | None = None):
        self.spec = spec or GridSpec()
        self.reset()

    def reset(self):
        self._stage = "coarse"
        self.coarse = None
        self.fine = None
        self.location_context = None
        self.last_choice = None

    def stage(self, observation) -> str:
        return self._stage

    def available_macros(self, observation) -> tuple[str, ...]:
        if observation.get("holding"):
            return ("lift", "release", "reselect", "done")
        if observation.get("gripper_state") == "open":
            return ("grasp", "open", "lift", "reselect", "done")
        return ("open", "lift", "reselect", "done")

    def questions(self, observation) -> tuple[GridQuestion, ...]:
        if self._stage == "coarse":
            return (
                GridQuestion("coarse_column", {label: label for label in self.spec.column_labels}),
                GridQuestion("coarse_row", {label: label for label in self.spec.row_labels}),
            )
        if self._stage == "fine":
            return (GridQuestion("fine_cell", {label: label for label in self.spec.fine_labels}),)
        macros = self.available_macros(observation)
        return (GridQuestion("macro", dict(zip(self.MACRO_LABELS, macros))),)

    def accept(self, choice, observation):
        if self._stage == "coarse":
            if not isinstance(choice, (tuple, list)) or len(choice) != 2:
                raise ValueError("coarse choice must contain column and row")
            column, row = map(str, choice)
            self.spec.coarse_bounds(column, row)
            self.coarse = (column, row)
            self.fine = None
            self.location_context = "carry" if observation.get("holding") else "approach"
            self._stage = "fine"
        elif self._stage == "fine":
            fine = str(choice)
            if self.coarse is None:
                raise ValueError("fine choice requires a coarse cell")
            self.spec.target_xy(*self.coarse, fine)
            self.fine = fine
            self._stage = "macro"
        else:
            choice = str(choice)
            if choice not in self.available_macros(observation):
                raise ValueError(f"macro {choice!r} unavailable for current robot feedback")
            if choice in ("reselect", "lift"):
                self._stage = "coarse"
                self.coarse = None
                self.fine = None
                self.location_context = None
        self.last_choice = choice

    def selected_target_xy(self) -> np.ndarray:
        if self.coarse is None or self.fine is None:
            raise ValueError("complete coarse and fine choices are required")
        return self.spec.target_xy(*self.coarse, self.fine)

    def apply(self, sim, choice, observation):
        """Apply a categorical choice; only fine and macro choices touch physics."""
        stage = self._stage
        if stage == "coarse":
            self.accept(choice, observation)
            return {"physics_step": False, "actual_displacement": [0., 0., 0.],
                    "contact": bool((observation.get("last_feedback") or {}).get("contact", False)),
                    "holding": observation.get("holding"), "grid_stage": "coarse"}
        if stage == "fine":
            target = self.spec.target_xy(*self.coarse, str(choice))
            feedback = dict(sim.step_target_xy(target))
            self.accept(choice, observation)
            feedback.update(physics_step=True, grid_stage="fine", grid_target_xy=target.tolist())
            return feedback
        macro = str(choice)
        if macro == "reselect":
            self.accept(macro, observation)
            return {"physics_step": False, "actual_displacement": [0., 0., 0.],
                    "contact": bool((observation.get("last_feedback") or {}).get("contact", False)),
                    "holding": observation.get("holding"), "grid_stage": "macro"}
        feedback = dict(sim.step(macro))
        self.accept(macro, observation)
        feedback.update(physics_step=True, grid_stage="macro")
        return feedback

    def get_state(self) -> dict:
        spec = asdict(self.spec)
        spec["xy_min"] = list(spec["xy_min"]); spec["xy_max"] = list(spec["xy_max"])
        return {
            "schema_version": 1,
            "spec": spec,
            "stage": self._stage,
            "coarse": list(self.coarse) if self.coarse else None,
            "fine": self.fine,
            "location_context": self.location_context,
            "last_choice": self.last_choice,
        }

    def set_state(self, state):
        if state.get("schema_version") != 1:
            raise ValueError("unsupported grid adapter state")
        if GridSpec(**state["spec"]) != self.spec:
            raise ValueError("grid adapter spec mismatch")
        if state["stage"] not in ("coarse", "fine", "macro"):
            raise ValueError("invalid grid adapter stage")
        self._stage = state["stage"]
        self.coarse = tuple(state["coarse"]) if state.get("coarse") else None
        self.fine = state.get("fine")
        self.location_context = state.get("location_context")
        self.last_choice = state.get("last_choice")
