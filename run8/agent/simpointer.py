"""Run 8 copy of run7/agent/simpointer.py (pointing through the camera the agent is asking about: top or wrist).

Run 7 copy of run6/agent/simpointer.py (Run 7 world; release-spot phrases for every relation, matched on the relation
so "put it in, then take it out" finds the right sub-goal).

Simulated pointer for CPU-only development (no model): answers the agent's pointing requests with the true 3D points
plus Gaussian table-plane noise, projected into the top camera like a model's answer. Used only on dev seeds to test the
Run 6 skill code paths (two-end bars, four-corner trays, jaw-offset placement); never in a test cell.
"""
from __future__ import annotations

import math

import numpy as np

from run3.phase2_photoreal import mjexport
from run7.env.policy import place_spot

PHRASE_RELATION = (("a free spot on the table just outside the ", "out"), ("a free spot on the table right next to the ", "next_to"),
                   ("a free spot inside the ", "in"), ("the top of the ", "on"))


class SimPointer:
    def __init__(self, agent, sigma_m: float = 0.0, seed: int = 0):
        self.agent, self.sigma, self.rng = agent, float(sigma_m), np.random.default_rng(seed)

    def _obj(self, env, noun: str) -> int | None:
        for i, o in enumerate(env.scene["objects"]):
            if noun.endswith(o["noun"]):
                return i
        return None

    def _cont(self, env, noun: str) -> int | None:
        for j, c in enumerate(env.scene["containers"]):
            if noun.endswith(c["noun"]):
                return j
        return None

    def _world_points(self, env, noun: str) -> list[np.ndarray]:
        if noun.startswith("FOUR:"):
            j = self._cont(env, noun[5:]); xy, yaw, half, _ = env.dest_frame(("container", j))
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            return [np.r_[xy + R @ (np.array([sx, sy]) * half), .01] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        if noun.startswith("the two ends of the "):
            i = self._obj(env, noun); o = env.scene["objects"][i]; xy, yaw = env.body_xy_yaw(f"object_{i}")
            ax = np.array([math.cos(yaw), math.sin(yaw)]) * o["size"][0] / 2
            return [np.r_[xy - ax, 2 * o["half_z"]], np.r_[xy + ax, 2 * o["half_z"]]]
        if noun.startswith("two neighbouring corners of the top face of the "):
            i = self._obj(env, noun); o = env.scene["objects"][i]; xy, yaw = env.body_xy_yaw(f"object_{i}")
            h = np.asarray(o["size"][:2]) / 2; R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            return [np.r_[xy + R @ np.array([-h[0], -h[1]]), 2 * o["half_z"]], np.r_[xy + R @ np.array([h[0], -h[1]]), 2 * o["half_z"]]]
        rel = next((r for pre, r in PHRASE_RELATION if noun.startswith(pre)), None)
        if rel is not None:
            held = env.held()
            for k, sg in enumerate(env.scene["subgoals"]):
                if sg["relation"] == rel and (held is None or held == f"object_{sg['object']}"):
                    d = sg["dest"]
                    ok = (d[0] == "container" and noun.endswith(env.scene["containers"][d[1]]["noun"])) or \
                         (d[0] == "object" and noun.endswith(env.scene["objects"][d[1]]["noun"]))
                    if ok:
                        z = {"in": .01, "on": float(env.dest_frame(d)[3])}.get(rel, 0.0)
                        return [np.r_[place_spot(env, k), z]]
            return []
        i = self._obj(env, noun)
        if i is None:
            return []
        xy, _ = env.body_xy_yaw(f"object_{i}")
        return [np.r_[xy, env.scene["objects"][i]["half_z"]]]

    def point(self, noun: str, image, n: int = 1) -> dict:
        env = self.agent._env
        name = getattr(self.agent, "_point_camera", "overhead")  # Run 8: wrist too
        cam = self.agent._true_top(env) if name == "overhead" else mjexport.camera_pose(env.model, env.data, name)
        pts = []
        for p in self._world_points(env, noun):
            p = p.copy(); p[:2] += self.rng.normal(0, self.sigma, 2)
            uv = mjexport.project(cam, p)
            if uv is not None:
                pts.append(uv)
        want = 4 if noun.startswith("FOUR:") else n
        return {"text": f"sim {noun}", "points": pts[:want] if len(pts) >= want else None, "latency_s": 0.0}

    def generate(self, prompt: str, images, max_tokens: int = 300) -> dict:
        """Only the block-corner prompt (agent dest_point=four_corners): four top-face corners in Qwen's 0-1000 format."""
        import json as _json
        env = self.agent._env
        noun = prompt.split("top face of the ", 1)[1].split(" in the image", 1)[0]
        i = self._obj(env, noun); o = env.scene["objects"][i]; xy, yaw = env.body_xy_yaw(f"object_{i}")
        h = np.asarray(o["size"][:2]) / 2; R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        cam = self.agent._true_top(env)
        out = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.r_[xy + R @ (np.array([sx, sy]) * h) + self.rng.normal(0, self.sigma, 2), 2 * o["half_z"]]
            u, v = mjexport.project(cam, p)
            out.append({"point_2d": [round(u / cam["width"] * 1000), round(v / cam["height"] * 1000)], "label": str(len(out) + 1)})
        return {"text": _json.dumps(out), "latency_s": 0.0}
