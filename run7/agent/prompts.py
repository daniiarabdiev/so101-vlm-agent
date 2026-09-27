"""Run 7 copy of run5/agent/prompts.py. Texts for the old relations ("in", "on") are unchanged character for character;
added: the table relations "out" / "next_to" and follow-up commands (PARSE_FOLLOWUP).

Every text the Run 5 models see. Shared by the agent, the screen, the few-shot library and the LoRA data, so
calibration, examples and fine-tuning all match the prompts used in the closed loop.

A sub-task (agent side) is {"object": noun, "dest": noun, "relation": "in" | "on", "repeat": bool}.
"repeat" sub-tasks ("all the red objects") keep going until no matching object is left outside the destination.
"""
from __future__ import annotations

from run7.env.policy import STEPS

LABELS = [chr(65 + i) for i in range(len(STEPS))]  # A..I
VIEWS = {"top": ["top"], "top+side": ["top", "side"], "top+side+wrist": ["top", "side", "wrist"]}
VIEW_TEXT = {"top": "top camera (looking straight down at the table; the robot base is at the left edge)",
             "side": "side camera (fixed, looking across the table from the robot's right; the robot base is at the left edge)",
             "wrist": "wrist camera (mounted on the gripper, looking down past the jaws; a jaw tip shows at the bottom left; "
                      "when the open gripper is directly above an object, that object fills the middle-right of this image)"}
START = "nothing yet (the robot is at its home pose)"


def target_noun(sub: dict) -> str:
    """What to point at for the object of this sub-task."""
    if sub.get("repeat"):
        return f"a {sub['object']} that is not inside the {sub['dest']}"
    return f"the {sub['object']}"


def dest_phrase(sub: dict) -> str:
    if sub["relation"] == "out":
        return f"a free spot on the table just outside the {sub['dest']}"
    if sub["relation"] == "next_to":
        return f"a free spot on the table right next to the {sub['dest']}"
    return f"a free spot inside the {sub['dest']}" if sub["relation"] == "in" else f"the top of the {sub['dest']}"


def subtask_text(sub: dict) -> str:
    if sub["relation"] == "out":
        return f"take the {sub['object']} out of the {sub['dest']} and put it on the table"
    if sub["relation"] == "next_to":
        return f"put the {sub['object']} on the table next to the {sub['dest']}"
    rel = "in" if sub["relation"] == "in" else "on top of"
    if sub.get("repeat"):
        return f"put every {sub['object']} {rel} the {sub['dest']}, one at a time"
    return f"put the {sub['object']} {rel} the {sub['dest']}"


def step_text(step: str, sub: dict) -> str:
    o = "the object" if sub.get("repeat") else f"the {sub['object']}"
    first = f"a {sub['object']} that is not in place yet" if sub.get("repeat") else o
    return {"move_to_object": f"move the open gripper above {first}",
            "grasp": "lower the open gripper and close it on the object below it",
            "lift": "raise the gripper to carrying height",
            "move_to_place": f"move the held object above {dest_phrase(sub)}",
            "rotate": f"turn the gripper to line up with {o} (when holding it: line it up with the {sub['dest']})",
            "release": "lower the held object until it touches, then open the gripper",
            "open": "open the gripper (it is closed on nothing)",
            "done": "this sub-task is complete",
            "give_up": "this sub-task cannot be completed (for example, the object is out of the robot's reach)"}[step]


def last_text(step: str, sub: dict) -> str:
    """What the robot last attempted (never whether it worked)."""
    o = "the object" if sub.get("repeat") else f"the {sub['object']}"
    return {"move_to_object": f"moved the open gripper above {o}",
            "grasp": "lowered the gripper and closed it",
            "lift": "raised the gripper to carrying height",
            "move_to_place": f"moved the held object above {dest_phrase(sub)}",
            "rotate": "turned the gripper to line it up",
            "release": "lowered the held object until it touched and opened the gripper",
            "open": "opened the gripper"}[step]


OUT_OF_REACH = "tried to move above {o}, but that spot is out of the robot's reach"
CHECK_FAILED_RELEASE = "was about to release, but the check said the held object is not above {d}; moved it again"
CHECK_FAILED_DONE = "said the sub-task was complete, but the check said it is not"
INVALID = "the previous answer could not be understood"


def gripper_text(holding: bool, closed: bool) -> str:
    if holding:
        return "closed on an object (the fingers stopped before fully closing)"
    return "closed on nothing (the fingers closed fully)" if closed else "open"


def context(instruction: str, subs: list[dict], k: int, last: str, gripper: str, views: list[str]) -> str:
    done = [subtask_text(s) for s in subs[:k]]
    lines = [f"Task: {instruction}",
             f"Current sub-task ({k + 1} of {len(subs)}): {subtask_text(subs[k])}."]
    if done:
        lines.append("Already completed: " + "; ".join(done) + ".")
    lines += [f"Last step the robot attempted: {last}. It may not have succeeded; check the images.",
              f"Gripper sensor: {gripper}.",
              "Images: " + "; ".join(f"image {i + 1} = {VIEW_TEXT[v]}" for i, v in enumerate(views)) + "."]
    return "\n".join(lines)


def next_step_options(sub: dict) -> dict:
    return {lab: f"{st}: {step_text(st, sub)}" for lab, st in zip(LABELS, STEPS)}


NEXT_Q = "What should the robot do next?"


def check_question(kind: str, sub: dict) -> str:
    if sub["relation"] in ("out", "next_to"):
        where = f"outside the {sub['dest']}" if sub["relation"] == "out" else f"right next to the {sub['dest']}"
        if kind == "over_dest":
            return (f"Is the held object directly above a free spot on the table {where}, "
                    f"so that lowering and releasing it now would leave it on the table {where}?")
        if kind == "done":
            return f"Is the {sub['object']} resting on the table {where}?"
    if kind == "over_dest":
        if sub["relation"] == "in":
            return (f"Is the held object directly above the inside of the {sub['dest']} (and lined up with it), "
                    f"so that lowering and releasing it now would leave it inside?")
        return f"Is the held object directly above the {sub['dest']}, so that lowering and releasing it now would leave it on top?"
    if kind == "done":
        if sub.get("repeat"):
            return f"Is every {sub['object']} resting inside the {sub['dest']}, with none left outside?"
        rel = "inside" if sub["relation"] == "in" else "on top of"
        return f"Is the {sub['object']} resting {rel} the {sub['dest']}?"
    raise ValueError(kind)


YES_NO = {"A": "yes", "B": "no"}

PARSE_RULES = ("Split it into pick-and-place sub-tasks, in order. For each sub-task give the object to move (colour and "
               "shape, e.g. \"red cube\"; for 'all the X objects' write \"X object\" and set repeat to true), the destination "
               "(e.g. \"blue container\", \"green block\", \"white tray\"), and the relation: \"in\" for containers and trays, "
               "\"on\" for placing on top of another object, \"out\" for taking an object out of a container onto the table "
               "(the destination is that container), \"next_to\" for putting it on the table beside another object (the "
               "destination is that object). Answer only with JSON.")
PARSE_PROMPT = "A robot arm receives this instruction: \"{instruction}\"\n" + PARSE_RULES
PARSE_FOLLOWUP = ("A robot arm has just completed this instruction: \"{previous}\"\n"
                  "It now receives a follow-up instruction: \"{instruction}\"\n"
                  "Words like \"it\" refer to the objects of the completed instruction. For the follow-up only: " + PARSE_RULES)
RELATIONS = ["in", "on", "out", "next_to"]
PARSE_SCHEMA = {"type": "object", "properties": {"subtasks": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
    "type": "object", "properties": {"object": {"type": "string"}, "dest": {"type": "string"},
                                     "relation": {"enum": RELATIONS}, "repeat": {"type": "boolean"}},
    "required": ["object", "dest", "relation", "repeat"]}}}, "required": ["subtasks"]}

POINT_ONE = 'Locate {noun} in the image. Output its center point as JSON: [{{"point_2d": [x, y], "label": "target"}}], coordinates normalized to 0-1000.'
POINT_TWO = ('Locate {what} in the image. Output the two points as JSON: [{{"point_2d": [x, y], "label": "1"}}, '
             '{{"point_2d": [x, y], "label": "2"}}], coordinates normalized to 0-1000.')
# Tray axis: four inner corners -> principal axis (tuning-split probe: median error 1.9 deg vs 18.5 deg for "two ends")
POINT_FOUR = ('Locate the four inner corners of the {tray} in the image. Output the four points as JSON: [{{"point_2d": [x, y], "label": "1"}}, '
              '{{"point_2d": [x, y], "label": "2"}}, {{"point_2d": [x, y], "label": "3"}}, {{"point_2d": [x, y], "label": "4"}}], '
              'coordinates normalized to 0-1000.')


def two_point_target(sub: dict, shape_word: str, holding: bool) -> str | None:
    """What to point at to measure an angle (None: rotation is not meaningful here)."""
    if holding:
        return f"FOUR:{sub['dest']}" if sub["dest"].endswith("tray") else None
    if shape_word == "bar":
        return f"the two ends of the {sub['object']}"
    if shape_word in ("cube", "block"):
        return f"two neighbouring corners of the top face of the {sub['object']}"
    return None


def subtasks_from_scene(scene: dict) -> list[dict]:
    """Ground-truth agent-level sub-tasks (the parse target)."""
    objs, conts = scene["objects"], scene["containers"]
    if scene["task"] == "gather":
        return [{"object": f"{objs[0]['color']} object", "dest": conts[0]["noun"], "relation": "in", "repeat": True}]
    out = []
    for sg in scene["subgoals"]:  # (a follow-up scene: the first command's sub-tasks, then the follow-up's)
        kind, j = sg["dest"]
        dest = conts[j]["noun"] if kind == "container" else objs[j]["noun"]
        out.append({"object": objs[sg["object"]]["noun"], "dest": dest, "relation": sg["relation"], "repeat": False})
    return out
