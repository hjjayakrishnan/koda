#!/usr/bin/env python3
"""The PPO settings must match rl_zoo3_balancer, and travel with the policy.

    ./.venv/bin/python tests/test_trainer_settings.py

Two kinds of check.

  1. Every value equals the one in rl_zoo3_balancer's hyperparams.yml, quoted
     below, and the units are what they are claimed to be. A transcription slip
     here is invisible: training still runs, it just trains the wrong thing.

  2. A policy survives being saved and loaded. Observation normalisation lives
     in a SEPARATE file from policy.zip, so a caller that loads the zip alone
     feeds raw observations to a network trained on normalised ones. That does
     not raise -- it silently looks like a bad policy, and would have been read
     as "the reward function is poor".
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koda import reward_env  # noqa: E402
from koda import trainer_pybullet as T  # noqa: E402

FAILURES = []

# rl_zoo3_balancer, hyperparams.yml, Upkie-PyBullet-Pendulum-Shaped:
#
#   n_timesteps:   1_000_000        learning_rate: lin_0.003
#   n_envs:        1                n_epochs:      10
#   n_steps:       2048             ent_coef:      0.0
#   batch_size:    64               gamma:         0.999
#   policy:        'MlpPolicy'      gae_lambda:    0.95
#   normalize:     enabled          clip_range:    0.1
#   net_arch:      pi [16,16], vf [32,32]
ZOO = {
    "gamma": 0.999,
    "learning_rate_start": 0.003,
    "clip_range": 0.1,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "net_arch": {"pi": [16, 16], "vf": [32, 32]},
}


def check(label, passed, detail=""):
    print(f"  {'ok  ' if passed else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not passed:
        FAILURES.append(label)


def values_match_the_source():
    print("Every setting matches rl_zoo3_balancer's hyperparams.yml")
    check("gamma 0.999", T.GAMMA == ZOO["gamma"], str(T.GAMMA))
    check("learning rate starts at 0.003",
          T.LEARNING_RATE_START == ZOO["learning_rate_start"],
          str(T.LEARNING_RATE_START))
    check("clip_range 0.1", T.CLIP_RANGE == ZOO["clip_range"], str(T.CLIP_RANGE))
    check("n_steps 2048", T.N_STEPS == ZOO["n_steps"], str(T.N_STEPS))
    check("batch_size 64", T.BATCH_SIZE == ZOO["batch_size"], str(T.BATCH_SIZE))
    check("n_epochs 10", T.N_EPOCHS == ZOO["n_epochs"], str(T.N_EPOCHS))
    check("gae_lambda 0.95", T.GAE_LAMBDA == ZOO["gae_lambda"], str(T.GAE_LAMBDA))
    check("ent_coef 0.0", T.ENT_COEF == ZOO["ent_coef"], str(T.ENT_COEF))
    check("net_arch pi[16,16] vf[32,32]",
          T.NET_ARCH == ZOO["net_arch"], str(T.NET_ARCH))
    check("observation normalisation on", T.NORMALIZE_OBSERVATIONS is True)
    check("batch_size divides n_steps", T.N_STEPS % T.BATCH_SIZE == 0,
          f"{T.N_STEPS} / {T.BATCH_SIZE}")


def units_are_what_we_claim():
    print("The units behind gamma are what the comment says")
    horizon_steps = 1.0 / (1.0 - T.GAMMA)
    horizon_seconds = horizon_steps / reward_env.FREQUENCY
    check("gamma 0.999 is ~1000 control steps",
          abs(horizon_steps - 1000) < 1, f"{horizon_steps:.0f}")
    check("which is 5.0 s at 200 Hz, not 5 steps or 5 ms",
          abs(horizon_seconds - 5.0) < 0.01, f"{horizon_seconds:.2f} s")
    check("SB3's old default 0.99 was 0.5 s, the number we are replacing",
          abs((1/(1-0.99))/reward_env.FREQUENCY - 0.5) < 0.01)
    check("the horizon is SHORTER than an episode, so it is a real horizon",
          horizon_steps < T.EVAL_STEPS, f"{horizon_steps:.0f} < {T.EVAL_STEPS}")
    check("FREQUENCY is 200 Hz, as the arithmetic above assumes",
          reward_env.FREQUENCY == 200.0, str(reward_env.FREQUENCY))


def learning_rate_actually_decays():
    print("lin_0.003 decays linearly to zero, it is not a constant")
    f = T.learning_rate_schedule()
    check("at the start (progress_remaining 1.0) it is 0.003",
          abs(f(1.0) - 0.003) < 1e-12, str(f(1.0)))
    check("halfway it is 0.0015", abs(f(0.5) - 0.0015) < 1e-12, str(f(0.5)))
    check("at the end it is 0.0", abs(f(0.0)) < 1e-12, str(f(0.0)))
    check("it is not SB3's constant 0.0003", f(1.0) != 0.0003)
    check("start is 10x SB3's default, as claimed",
          abs(f(1.0) / 0.0003 - 10.0) < 1e-9, f"{f(1.0)/0.0003:.1f}x")


def normalisation_travels_with_the_policy():
    """The failure this is here for: policy.zip loaded without its statistics."""
    print("A saved policy keeps its observation statistics")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = reward_env.make_env(lambda s, a: {"x": 0.0}, seed=11, gui=False,
                              max_steps=64)
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True,
                        norm_reward=False, clip_obs=T.CLIP_OBS)
    model = PPO("MlpPolicy", venv, seed=11, verbose=0, device="cpu",
                n_steps=64, batch_size=32,
                policy_kwargs={"net_arch": dict(T.NET_ARCH)})
    model.learn(total_timesteps=128, progress_bar=False)
    venv.training = False

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        model.save(d / "policy.zip")
        venv.save(str(d / T.VECNORM_FILENAME))

        loaded = T.load_policy(d / "policy.zip")
        check("load_policy finds the statistics file",
              loaded.vecnorm is not None)
        check("statistics are frozen, so scoring cannot keep updating them",
              loaded.vecnorm.training is False)
        check("reward normalisation is off, so the candidate's own reward "
              "is not rescaled", loaded.vecnorm.norm_reward is False)

        raw = np.array([0.4, 2.0, -3.0, 0.1, 0.6], dtype=np.float32)
        normed = loaded.vecnorm.normalize_obs(raw)
        check("normalisation actually changes the observation",
              not np.allclose(raw, normed), f"{raw} -> {np.round(normed,3)}")

        # The round trip that matters: same observation, same action.
        before = model.predict(loaded.vecnorm.normalize_obs(raw),
                               deterministic=True)[0]
        after = loaded.predict(raw, deterministic=True)[0]
        check("a reloaded policy produces the SAME action for the same "
              "observation", np.allclose(before, after),
              f"{np.round(before,5)} vs {np.round(after,5)}")

        # And the bug: loading the zip alone.
        bare = PPO.load(str(d / "policy.zip"), device="cpu")
        bare_action = bare.predict(raw, deterministic=True)[0]
        check("loading policy.zip WITHOUT the statistics gives a different "
              "action -- this is the silent failure being prevented",
              not np.allclose(bare_action, after),
              f"bare {np.round(bare_action,5)} vs correct {np.round(after,5)}")

        # A policy with no statistics file must still load, not crash.
        (d / T.VECNORM_FILENAME).unlink()
        old = T.load_policy(d / "policy.zip")
        check("a policy saved before normalisation existed still loads",
              old.vecnorm is None)
        check("and it passes observations through untouched",
              np.allclose(old.predict(raw)[0], bare_action))
    env.close()


def nothing_loads_a_policy_the_unsafe_way():
    print("No caller bypasses load_policy")
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in list(root.glob("*.py")) + list((root / "koda").glob("*.py")):
        if path.name == "trainer_pybullet.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if "PPO.load" in line and not line.strip().startswith("#"):
                offenders.append(f"{path.name}:{n}")
    check("no PPO.load outside trainer_pybullet.py", not offenders,
          str(offenders))


def main():
    values_match_the_source()
    units_are_what_we_claim()
    learning_rate_actually_decays()
    normalisation_travels_with_the_policy()
    nothing_loads_a_policy_the_unsafe_way()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("29 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
