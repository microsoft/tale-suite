"""Situated Awareness failure detector.

Replays a transcript through its TALES environment and reads the ground-truth world
state after every step. After the first GRACE_STEPS steps, an action that is not
admissible and leaves the world state unchanged is classified by each entity it names:
  scope_violation  the entity exists, but neither here nor in the agent's inventory;
  stale_reference  the entity was in an earlier world state but no longer exists;
  hallucination    the entity was never in any world state.
The replay must reproduce every recorded feedback, so the transcript needs the game seed
it was played with ("game_seed", null for benchmark.py's default).

Usage: python analysis/situated_awareness.py TRANSCRIPT.json [...] > sa.jsonl
"""

import json
import os
import re
import sys
from pathlib import Path

import gymnasium as gym

import tales  # noqa: F401 (registers the tales/* environments)

GRACE_STEPS = 10
DIRECTIONS = set(
    """north south east west northeast northwest southeast southwest up down
    forward back left right out in""".split()
)
NON_ENTITY_VERBS = set(
    """say answer learn memorize cast shout yell scream whisper sing chant pray think
    wait sleep wake save restore restart quit score verbose brief superbrief diagnose
    help hint undo again""".split()
)
NON_ENTITY_PHRASES = {"stand up", "lie down", "sit down", "wake up", "take off"}
PREPOSITIONS = re.compile(r"\b(?:from|to|with|in|on|into|using|at|through)\b", re.I)
TW_RECEPTACLES = {"c", "s", "oven", "toaster", "stove", "bbq"}  # TextWorld types
ARTICLED = ("a ", "an ", "the ")
TRACKED = ("agent_at", "objects", "inventory", "open")

# World state: agent_at (str, "" when the agent has no known location), objects
# ({name: parent}), inventory ([name]), receptacles ({name: location}), open ([name])
# and admissible ([action]).


def alfworld_state(env, info):
    state = env.unwrapped.env.unwrapped.state
    name = lambda arg: str(state["_entity_infos"][arg.name]).strip()
    at, objects, receptacles, inventory, opened = None, {}, {}, [], []
    for fact in state["_facts"]:
        args = fact.arguments
        if fact.name == "atlocation":
            at = args[1].name
        elif fact.name == "inreceptacle":
            objects[name(args[0])] = name(args[1])
        elif fact.name == "receptacleatlocation":
            receptacles[name(args[0])] = args[1].name
        elif fact.name == "holds":
            inventory.append(name(args[1]))
        elif fact.name == "isopened":
            opened.append(name(args[0]))
    assert at is not None
    # "start" when the agent is at no receptacle.
    agent_at = next((r for r, loc in receptacles.items() if loc == at), "start")
    return dict(
        agent_at=agent_at,
        objects=objects,
        inventory=inventory,
        receptacles=receptacles,
        open=opened,
        admissible=state["admissible_commands"],
    )


def jericho_state(env, info):
    frotz = env.unwrapped.env._wrapped_env._jericho
    world = frotz.get_world_objects()
    names = {o.num: o.name.strip() for o in world}
    # Parents outside the world objects are unnamed.
    hidden = {o.parent for o in world} - names.keys()
    assert not any(map(frotz.get_object, hidden)), hidden
    names |= dict.fromkeys(hidden, "")
    location = frotz.get_player_location()
    return dict(
        agent_at=location.name.strip() if location else "",
        objects={
            o.name.strip(): names[o.parent]
            for o in world
            if o.name.strip() and o.parent
        },
        inventory=[o.name.strip() for o in frotz.get_inventory()],
        receptacles={},
        open=[],
        admissible=[],  # not computed
    )


def textworld_state(env, info):
    at, objects, receptacles, inventory, opened = None, {}, {}, [], []
    for fact in info["facts"]:
        names = [arg.name for arg in fact.arguments]
        if fact.name == "at" and names[0] == "P":
            at = names[1]
        elif fact.name == "at" and fact.arguments[0].type in TW_RECEPTACLES:
            receptacles[names[0]] = names[1]
        elif fact.name == "in" and names[1] == "I":
            inventory.append(names[0])
        elif fact.name in ("at", "in", "on"):
            objects[names[0]] = names[1]
        elif fact.name == "open":
            opened.append(names[0])
    assert at is not None
    return dict(
        agent_at=at,
        objects=objects,
        inventory=inventory,
        receptacles=receptacles,
        open=opened,
        admissible=info["admissible_commands"],
    )


def articled(text):
    """The lines of text that start with an article."""
    return [l.strip() for l in text.split("\n") if l.strip().startswith(ARTICLED)]


def twx_state(env, info):
    room = re.match(r"You are in (?:the |a )([^.]+)\.", info["look"])
    rooms = env.unwrapped.game_name != "simonsays"
    assert bool(room) == rooms, info["look"]
    return dict(
        agent_at=room[1].strip() if rooms else "",
        objects={},
        inventory=articled(info["inventory"]),
        receptacles={},
        open=[],
        admissible=info["admissible_commands"],
    )


def scienceworld_state(env, info):
    look = env.unwrapped.env.look()
    room = re.search(r"is called the ([^.]+)\.", look)
    assert room, look
    agent_at, objects, container = room[1].strip(), {}, None
    for line in look.split("\n"):
        item = line.strip()
        if not item.startswith(ARTICLED):
            continue
        if "containing" in item.lower():
            container = item.split("(")[0].strip()
            objects[container] = agent_at
        elif container and line.startswith("\t\t"):
            objects[item] = container
        else:
            objects[item], container = agent_at, None
    return dict(
        agent_at=agent_at,
        objects=objects,
        inventory=articled(env.unwrapped.env.inventory()),
        receptacles={},
        open=[],
        admissible=info["admissible_commands"],
    )


def scienceworld_text(text):
    """Order-insensitive word multiset, ignoring temperatures and bee counts."""
    text = re.sub(r"temperature of -?\d+", "temperature of N", text + "\n")
    text = re.sub(r"(a adult bee\s+)+", "a adult bee ", text)
    return sorted(re.findall(r"\w+", text))


def scienceworld_action(action, menu, recorded_menu):
    """Map an option number of the recorded disambiguation menu to the replayed one."""
    if not (action.strip().isdigit() and "Ambiguous request:" in menu):
        return action
    option = re.compile(r"^[ \t]*(\d+):[ \t]*(.*\S)", re.M)
    chosen = {int(n): o for n, o in option.findall(recorded_menu)}.get(int(action))
    if chosen is None:  # out of range
        return action
    return {o: n for n, o in option.findall(menu)}[chosen]


def textworld_text(text):
    """Whitespace-insensitive, ignoring turn counts."""
    return " ".join(re.sub(r"in \d+ turn\(s\)", "in N turn(s)", text).split())


STATE = {
    "alfworld": alfworld_state,
    "jericho": jericho_state,
    "textworld": textworld_state,
    "textworld_express": twx_state,
    "scienceworld": scienceworld_state,
}
TEXT = {"scienceworld": scienceworld_text, "textworld": textworld_text}


def replay(path):
    """Yield (action, world state before, world state after) for each step."""
    transcript = json.loads(Path(path).read_text())
    turns = transcript["turns"]
    env = gym.make(f"tales/{transcript['game']}-v0", disable_env_checker=True)
    framework = type(env.unwrapped).__module__.split(".")[1]
    state, text = STATE[framework], TEXT.get(framework, str)
    if framework == "textworld":
        env.unwrapped.infos.facts = env.unwrapped.infos.admissible_commands = True
    obs, info = env.reset(seed=transcript["game_seed"])
    assert text(obs) == text(turns[0]["content"]), f"{path}: wrong game or seed"
    done, before = False, state(env, info)
    for i in range(1, len(turns), 2):
        action = turns[i]["content"]
        # benchmark.py does not send multi-line actions.
        if "\n" not in action.strip():
            if framework == "scienceworld":
                action = scienceworld_action(
                    action, info["feedback"], turns[i - 1]["content"]
                )
            _, _, done, info = env.step(action)
        got, recorded = info["feedback"], turns[i + 1]["content"]
        message = f"{path}: step {i // 2 + 1} feedback differs:\n{got!r}\n{recorded!r}"
        assert text(got) == text(recorded), message
        if done:  # benchmark.py restarts the game
            _, info = env.reset()
        after = state(env, info)
        yield turns[i]["content"].strip(), before, after
        before = after
    env.close()


def entity_refs(action):
    words = action.split()
    if not words or action.startswith('"') and action.endswith('"'):
        return []
    verb, two = words[0].lower(), " ".join(words[:2]).lower()
    if verb in NON_ENTITY_VERBS or two in NON_ENTITY_PHRASES:
        return []
    if two in ("turn on", "turn off", "pick up", "put down"):
        return [" ".join(words[2:])] if words[2:] else []
    if len(words) == 2 and two in ("look around", "look here", "look up", "look down"):
        return []
    if verb in ("go", "walk", "run", "follow"):
        target = " ".join(words[1:]).lower()
        target = target[3:].strip() if target.startswith("to ") else target
        return [target] if target and target not in DIRECTIONS else []
    if verb in ("ask", "tell", "talk"):  # the person, not the topic
        person = " ".join(re.split(r"\babout\b", action, flags=re.I)[0].split()[1:])
        person = person[3:].strip() if person.lower().startswith("to ") else person
        return [person] if person else []
    first, *rest = PREPOSITIONS.split(action)
    return [ref for ref in [" ".join(first.split()[1:]), *map(str.strip, rest)] if ref]


def reachable(obj, here, objects):
    """Whether obj's containment chain (at most 10 levels) leads to here."""
    chain = set()
    while obj in objects and obj not in chain and len(chain) < 10:
        chain.add(obj)
        obj = objects[obj]
        if obj.lower() == here:
            return True
    return False


def real(parent):
    """False for parents of Z-machine templates, directions and internal containers."""
    wrapped = parent.startswith("(") and parent.endswith(")")
    template = parent in ("Class", "Object", "Nowhere", "")
    return not (wrapped or template or parent.lower() == "compass")


def classify(action, ws, seen):
    """(type, entity) for each entity the action names that is out of scope, gone or
    nonexistent."""
    if "\n" in action:
        return []
    here, objects = ws["agent_at"].lower(), ws["objects"]
    in_scope = lambda o: o in ws["inventory"] or reachable(o, here, objects)
    lower = lambda names: {name.lower(): name for name in names}
    receptacles, objs = lower(ws["receptacles"]), lower(objects)
    found = []
    for ref in entity_refs(action):
        r = ref.lower()
        near = lambda names: [names[n] for n in names if r in n or n in r]
        if r in receptacles:
            if here != r:
                found.append(("scope_violation", receptacles[r]))
        elif r in objs:
            if real(objects[objs[r]]) and not in_scope(objs[r]):
                found.append(("scope_violation", objs[r]))
        elif r in lower(ws["inventory"]):
            pass
        elif r in lower(seen):
            found.append(("stale_reference", ref))
        elif near(objs):
            candidates = [o for o in near(objs) if real(objects[o])]
            if candidates and not any(map(in_scope, candidates)):
                found.append(("scope_violation", candidates[0]))
        elif not near(receptacles):
            kind = "stale_reference" if near(lower(seen)) else "hallucination"
            found.append((kind, ref))
    return found


def detect(path):
    seen, failures, steps = set(), [], 0
    for steps, (action, before, after) in enumerate(replay(path), 1):
        seen |= {*before["objects"], *before["receptacles"], *before["inventory"]}
        if steps <= GRACE_STEPS or not before["agent_at"]:
            continue
        unchanged = all(before[k] == after[k] for k in TRACKED)
        if unchanged and action not in before["admissible"]:
            failures += [
                dict(step=steps, type=kind, action=action, entity=entity)
                for kind, entity in classify(action, before, seen)
            ]
    return {
        "transcript": str(path),
        "steps": steps,
        "situated_awareness_failures": len(failures),
        "failures": failures,
    }


if __name__ == "__main__":
    # Only results go to stdout; game engines write to it directly.
    out = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    for path in sys.argv[1:]:
        print(json.dumps(detect(path)), file=out, flush=True)
