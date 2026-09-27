"""Privileged grid baselines, kept separate from model-visible pipelines."""
from __future__ import annotations

import numpy as np

from .grid_actions import GridAdapter
from .tasks import _half_extents, score


class GridOraclePolicy:
    def __init__(self, adapter: GridAdapter):
        self.adapter = adapter
        self.last_decision = None

    def reset(self, seed=0):
        self.last_decision = None

    @staticmethod
    def _target_xy(observation):
        obj = observation["objects"][0]
        if not observation.get("holding"):
            return np.asarray(obj["pos"][:2], dtype=float)
        destination_index = 1 if observation.get("task") == "B" else 0
        destination = observation["containers"][destination_index]
        grasp_offset = np.asarray(observation["ee_pos"][:2]) - np.asarray(obj["pos"][:2])
        return np.asarray(destination["pos"][:2], dtype=float) + grasp_offset

    def choose(self, observation):
        stage = self.adapter.stage(observation)
        target = None
        if score(observation)["success"]:
            choice = "done"
        elif stage == "coarse":
            target = self._target_xy(observation)
            bounded = np.clip(target, self.adapter.spec.xy_min, self.adapter.spec.xy_max)
            choice = self.adapter.spec.coarse_for_point(bounded)
        elif stage == "fine":
            target = self._target_xy(observation)
            # A prior noisy/model coarse choice may exclude the true target.
            # Keep that mistake visible and choose the closest available fine
            # center; never smuggle the target into a different crop.
            centers = {label: self.adapter.spec.target_xy(*self.adapter.coarse, label)
                       for label in self.adapter.spec.fine_labels}
            choice = min(centers, key=lambda label: np.linalg.norm(centers[label]-target, ord=np.inf))
        elif observation.get("holding"):
            if self.adapter.location_context != "carry":
                choice = "lift"
            else:
                obj = observation["objects"][0]
                destination = observation["containers"][1 if observation.get("task") == "B" else 0]
                offset = np.abs(np.asarray(obj["pos"][:2])-np.asarray(destination["pos"][:2]))
                usable = np.asarray(destination["inner_size"][:2])/2-_half_extents(obj)[:2]-.001
                choice = "release" if np.all(offset <= usable) else "reselect"
        elif observation.get("gripper_state") != "open":
            choice = "open"
        else:
            carry_height = float(observation.get("carry_height", .085))
            if float(observation["ee_pos"][2]) < carry_height - .012 and self.adapter.last_choice == "open":
                choice = "lift"
            elif self.adapter.location_context == "approach" and self.adapter.fine is not None:
                obj_xy = np.asarray(observation["objects"][0]["pos"][:2])
                aligned = np.max(np.abs(np.asarray(observation["ee_pos"][:2])-obj_xy)) <= .01
                choice = "grasp" if aligned else "reselect"
            else:
                choice = "reselect"
        self.last_decision = {"stage": stage, "choice": choice,
                              "target_xy": target.tolist() if target is not None else None}
        return choice


class GridRandomPolicy:
    def __init__(self, adapter: GridAdapter):
        self.adapter = adapter
        self.reset()

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.last_decision = None

    def choose(self, observation):
        questions = self.adapter.questions(observation)
        if len(questions) == 2:
            choice = tuple(str(self.rng.choice(tuple(question.options.values()))) for question in questions)
        else:
            choice = str(self.rng.choice(tuple(questions[0].options.values())))
        self.last_decision = {"stage": self.adapter.stage(observation), "choice": choice}
        return choice


class GridNoisyOraclePolicy(GridOraclePolicy):
    def __init__(self, adapter: GridAdapter, noise_probability=.15):
        super().__init__(adapter)
        self.noise_probability = float(noise_probability)
        if not 0 <= self.noise_probability <= 1:
            raise ValueError("noise_probability must lie in [0, 1]")
        self.reset()

    def reset(self, seed=0):
        super().reset(seed)
        self.rng = np.random.default_rng(seed)

    def choose(self, observation):
        stage = self.adapter.stage(observation)
        reference = super().choose(observation)
        replaced = bool(self.rng.random() < self.noise_probability)
        if replaced:
            questions = self.adapter.questions(observation)
            if len(questions) == 2:
                choice = tuple(str(self.rng.choice(tuple(question.options.values()))) for question in questions)
            else:
                choice = str(self.rng.choice(tuple(questions[0].options.values())))
        else:
            choice = reference
        self.last_decision = {"stage": stage, "reference_choice": reference,
                              "choice": choice, "noise_injected": replaced}
        return choice


GridNoisyOracle = GridNoisyOraclePolicy
