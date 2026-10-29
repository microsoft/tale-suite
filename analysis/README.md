# Failure detectors

Detectors for the Rule Learning and Situated Awareness failures defined in the TALES paper.
Each reads transcripts (one JSON file each) and prints one JSON line per transcript.

A transcript is `{"game": ..., "game_seed": ..., "turns": [...]}`:
- `game`: the environment name without `tales/` and `-v0`, e.g. `JerichoEnvZork1`.
- `game_seed`: `benchmark.py --game-seed`, `null` by default. Every run in
  [talesuite/tale_suite_trajectories](https://huggingface.co/datasets/talesuite/tale_suite_trajectories) used the default.
- `turns`: `{"role": "environment" | "agent", "content": ...}` entries: the initial observation, then each action and
  its feedback (the `Observation`, `Action` and `Feedback` columns of the `benchmark.py` logs).

```bash
python analysis/rule_learning.py transcripts/*.json > rule_learning.jsonl
ls transcripts/*.json | xargs -n1 -P8 timeout 900 python analysis/situated_awareness.py > situated_awareness.jsonl
```

The paper reports `rule_learning_failures + situated_awareness_failures` per game, averaged over seeds, then games.

## Rule Learning

A feedback is a rejection if `error_messages/<game>.txt` lists it.

## Situated Awareness

The detector replays the transcript through its environment, so it needs every framework installed.
A replay that does not reproduce a recorded feedback raises an `AssertionError`, so run one process per transcript.
