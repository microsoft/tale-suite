"""Test that replaying recorded actions through an environment produces the same trajectory.

This script runs a short game with the random agent, records the trajectory,
then replays the same actions and verifies the observations match.
"""

import gymnasium as gym
import numpy as np

import tales

NB_STEPS = 15
GAME_SEED = 42


def run_game(env_name, seed, nb_steps, actions=None):
    """Run a game and return the trajectory.

    If actions is provided, replay those actions instead of generating random ones.
    Returns a list of dicts with step, obs, action, feedback, score, done.
    """
    env = gym.make(
        f"tales/{env_name}-v0",
        disable_env_checker=True,
        admissible_commands=False,
    )

    obs, info = env.reset(seed=seed)
    rng = np.random.RandomState(seed)
    trajectory = []

    for step in range(1, nb_steps + 1):
        if actions is not None:
            action = actions[step - 1]
        else:
            # Use a simple deterministic action for testing.
            action = rng.choice(
                [
                    "look",
                    "inventory",
                    "north",
                    "south",
                    "east",
                    "west",
                    "take all",
                    "open door",
                    "examine room",
                ]
            )

        prev_obs = obs
        if "\n" in action.strip():
            obs = "The game only allows one action per step."
            done = False
            info_after = info
        else:
            obs, _, done, info_after = env.step(action)

        trajectory.append(
            {
                "step": step,
                "prev_obs": prev_obs,
                "action": action,
                "obs_after": obs,
                "feedback": info_after.get("feedback", ""),
                "score": info_after["score"],
                "done": done,
            }
        )

        info = info_after

        if done:
            last_obs = obs
            obs, info = env.reset()
            obs = last_obs + "\n\n-= Restarting =-\n" + obs

    env.close()
    return trajectory


def test_env(env_name):
    """Test replay determinism for a single environment."""
    print(f"\n{'='*60}")
    print(f"Testing: {env_name}")
    print(f"{'='*60}")

    # Run 1: generate trajectory
    traj1 = run_game(env_name, GAME_SEED, NB_STEPS)
    actions = [t["action"] for t in traj1]

    print(f"  Run 1: {len(traj1)} steps, final score={traj1[-1]['score']}")

    # Run 2: replay the same actions
    traj2 = run_game(env_name, GAME_SEED, NB_STEPS, actions=actions)

    print(f"  Run 2: {len(traj2)} steps, final score={traj2[-1]['score']}")

    # Compare trajectories
    mismatches = 0
    for i, (t1, t2) in enumerate(zip(traj1, traj2)):
        if t1["prev_obs"].strip() != t2["prev_obs"].strip():
            print(f"  MISMATCH at step {t1['step']} (prev_obs):")
            print(f"    Run 1: {t1['prev_obs'][:100]!r}")
            print(f"    Run 2: {t2['prev_obs'][:100]!r}")
            mismatches += 1
        if t1["obs_after"].strip() != t2["obs_after"].strip():
            print(f"  MISMATCH at step {t1['step']} (obs_after):")
            print(f"    Run 1: {t1['obs_after'][:100]!r}")
            print(f"    Run 2: {t2['obs_after'][:100]!r}")
            mismatches += 1
        if t1["score"] != t2["score"]:
            print(
                f"  MISMATCH at step {t1['step']} (score): {t1['score']} vs {t2['score']}"
            )
            mismatches += 1

    if mismatches == 0:
        print(f"  ✅ PASS: All {len(traj1)} steps match perfectly.")
    else:
        print(f"  ❌ FAIL: {mismatches} mismatches found.")

    return mismatches == 0


def main():
    # Test one environment from each framework.
    test_envs = []

    # Pick one env per framework.
    for task in tales.tasks:
        envs = tales.envs_per_task.get(task, [])
        if envs:
            test_envs.append((task, sorted(envs)[0]))

    print(f"Testing replay determinism for {len(test_envs)} environments:")
    for task, env in test_envs:
        print(f"  {task}: {env}")

    results = {}
    for task, env_name in test_envs:
        try:
            passed = test_env(env_name)
            results[env_name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[env_name] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for env_name, result in results.items():
        status = "✅" if result == "PASS" else "❌"
        print(f"  {status} {env_name}: {result}")


if __name__ == "__main__":
    main()
