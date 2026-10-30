#!/usr/bin/env python3
"""Train one reward function with rl_zoo3_balancer's PPO settings.

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python <this file> --reward <file.py> --id <name>

Nothing in koda/ is modified. This is an experiment, not a change to the
measurement code: if it works, the settings get proposed properly afterwards.

rl_zoo3_balancer is a third-party SB3/PPO balancer for this same robot.
Its hyperparams.yml, against ours:

                   theirs              ours (SB3 defaults)
    n_timesteps    1_000_000           500_000
    learning_rate  lin_0.003           0.0003
    gamma          0.999               0.99
    clip_range     0.1                 0.2
    net_arch       pi[16,16] vf[32,32] [64,64] both
    normalize      enabled             not used

gamma is the one I expect to matter. Effective horizon is about 1/(1-gamma)
control steps, and this runs at 200 Hz:

    gamma 0.99  -> ~100 steps  = 0.5 s      <- ours
    gamma 0.999 -> ~1000 steps = 5.0 s      <- theirs

The robot must drive its wheels BACKWARD for roughly half a second before it
can accelerate forward. At gamma 0.99 that manoeuvre never pays off inside the
horizon the agent can see.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from koda import gate1, reward_env, scorer
from koda import trainer_pybullet as T


def linear(start):
    """lin_0.003 -- decays to zero across training, as rl_zoo3 does."""
    def f(progress_remaining):
        return progress_remaining * start
    return f


class NormalisedPolicy:
    """Applies the training-time observation statistics at rollout time.

    `_rollout_episode` drives a plain environment, so the normalisation has to
    live with the policy rather than wrap the env -- otherwise the scored
    episodes would feed raw observations to a network trained on normalised
    ones, which would look like the reward failing.
    """

    def __init__(self, model, vecnorm):
        self.model = model
        self.vecnorm = vecnorm

    def predict(self, observation, deterministic=True):
        obs = self.vecnorm.normalize_obs(np.asarray(observation, dtype=np.float32))
        return self.model.predict(obs, deterministic=deterministic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--default-net", action="store_true",
                   help="keep SB3's [64,64] instead of theirs")
    args = p.parse_args()

    reward_fn = gate1.check(Path(args.reward).read_text(), args.id)["reward_fn"]

    print(f"reward     {args.reward}")
    print(f"steps      {args.steps:,}")
    print(f"gamma      {args.gamma}   (horizon ~{1/(1-args.gamma):.0f} steps "
          f"= {1/(1-args.gamma)/200:.2f} s at 200 Hz)")
    print(f"lr         linear from {args.lr}")
    print(f"clip_range {args.clip}")
    print(f"normalize  {not args.no_normalize}")
    print(f"net_arch   {'SB3 default [64,64]' if args.default_net else 'pi[16,16] vf[32,32]'}")
    print()

    started = time.time()
    venv = DummyVecEnv([lambda: reward_env.make_env(
        reward_fn, seed=args.seed, gui=False, max_steps=T.EVAL_STEPS)])
    vecnorm = VecNormalize(venv, norm_obs=not args.no_normalize,
                           norm_reward=False, clip_obs=10.0)

    kwargs = dict(seed=args.seed, verbose=0, device="cpu",
                  learning_rate=linear(args.lr), gamma=args.gamma,
                  clip_range=args.clip, n_steps=2048, batch_size=64,
                  n_epochs=10, gae_lambda=0.95, ent_coef=0.0)
    if not args.default_net:
        kwargs["policy_kwargs"] = dict(net_arch=dict(pi=[16, 16], vf=[32, 32]))

    model = PPO("MlpPolicy", vecnorm, **kwargs)
    model.learn(total_timesteps=args.steps, progress_bar=False)
    vecnorm.training = False
    print(f"trained in {time.time()-started:.0f} s")

    policy = NormalisedPolicy(model, vecnorm)
    out = Path("runs") / args.id
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "policy.zip")
    vecnorm.save(str(out / "vecnormalize.pkl"))
    venv.close()

    # Scored by the project's own grader, a fresh environment per episode.
    episodes = []
    for ep in T.eval_set.episodes():
        env = reward_env.make_env(reward_fn, seed=ep["seed"], gui=False,
                                  max_steps=T.EVAL_STEPS)
        w = reward_env.find_reward_wrapper(env)
        try:
            episodes.append(T._rollout_episode(policy, env, w, ep))
        finally:
            env.close()
    rollout = {"schema_version": T.SCHEMA_VERSION, "episodes": episodes}
    verdict = scorer.score(rollout, args.id)
    m = verdict["measurements"]
    print()
    print(f"SCORE      {verdict['score']:.4f}")
    print(f"  survival {m['survival_frac']:.3f}")
    print(f"  headroom {1-m['saturated_step_frac_wheel']:.3f}")
    print(f"  spd err  {m['mean_speed_error']:.3f}")
    print(f"  distance {m['displacement_frac'] or 0:+.3f}")
    print(f"  full 20s episodes: {sum(1 for e in episodes if not e['fell'])}/20")
    json.dump({"rollout": rollout, "score": verdict,
               "hparams": vars(args)}, (out / "result.json").open("w"), indent=2)
    print(f"\nwrote runs/{args.id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
