"""Rule Learning failure detector.

Pass one flags an action that repeats an action the environment rejected, with only
rejected actions in between. Pass two flags the steps of verbatim (action, feedback)
loops of period 2 to 10: the first period is exploration, every later step a failure.
A step flagged by both passes counts once. A feedback is a rejection if it is listed in
error_messages/<game>.txt (lines "count | message").

Usage: python analysis/rule_learning.py TRANSCRIPT.json [...] > rule_learning.jsonl
"""

import json
import sys
from pathlib import Path

ERRORS = Path(__file__).parent / "error_messages"
MAX_PERIOD = 10


def load_errors(game):
    lines = (ERRORS / f"{game}.txt").read_text().split("\n")
    messages = [l.split(" | ", 1)[1] for l in lines if l and l[0] != "#"]
    return {m.replace("\\n", "\n").replace("\\t", "\t") for m in messages}


def repeats(pairs, errors):
    rejected, flagged = set(), []
    for k, (action, feedback) in enumerate(pairs):
        if feedback not in errors:
            rejected.clear()
            continue
        if action in rejected:
            flagged.append(k)
        rejected.add(action)
    return flagged


def cycles(pairs):
    """(start, period, repetitions) of each loop, shortest periods first; no loop starts
    inside an earlier one."""
    found, covered = [], set()
    for p in range(2, MAX_PERIOD + 1):
        i = 0
        while i <= len(pairs) - 2 * p:
            reps = 1
            while pairs[i + reps * p : i + (reps + 1) * p] == pairs[i : i + p]:
                reps += 1
            if reps > 1 and i not in covered:
                found.append((i, p, reps))
                covered.update(range(i, i + reps * p))
                i += reps * p
            else:
                i += 1
    return found


def detect(path):
    transcript = json.loads(Path(path).read_text())
    roles = [t["role"] for t in transcript["turns"]]
    assert roles == ["environment"] + ["agent", "environment"] * (len(roles) // 2), path
    contents = [t["content"].strip() for t in transcript["turns"]]
    pairs = list(zip(contents[1::2], contents[2::2]))  # (action, feedback)
    errors = load_errors(transcript["game"])
    rep, cyc = repeats(pairs, errors), cycles(pairs)
    failures = set(rep).union(*(range(i + p, i + r * p) for i, p, r in cyc))
    return {
        "transcript": str(path),
        "steps": len(pairs),
        "rejected_steps": sum(feedback in errors for _, feedback in pairs),
        "repeats": [k + 1 for k in rep],  # steps are 1-based
        "cycles": [dict(first_step=i + 1, period=p, repetitions=r) for i, p, r in cyc],
        "rule_learning_failures": len(failures),
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(detect(path)), flush=True)
