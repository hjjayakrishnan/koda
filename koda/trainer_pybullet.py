# SPDX-License-Identifier: Apache-2.0
"""The real trainer. PPO, PyBullet, CPU only.

Same two functions as `trainer_stub.py` -- `name()` and `train()` -- returning
the same TrainerResult shape, so `orchestrator.py` cannot tell them apart.

What is real here that was made up in the stub: all of it. The reward function
is called inside a physics simulation, a policy network is actually trained
against it, and every number in `rollout` is measured from the simulated robot.

No GPU. The machine has no NVIDIA hardware and torch is the CPU build.
"""

import math
import time
import traceback
from pathlib import Path

import numpy as np

from koda import SCHEMA_VERSION, eval_set
from koda import reward_env
from koda import state as state_mod

CHECKPOINTS = (0.10, 0.25, 0.50, 1.00)

# Hip and knee travel limits, read from the installed upkie model. The wheels
# spin freely so they have no angle margin.
JOINT_LIMITS_RAD = {
    "left_hip": 1.260,
    "left_knee": 2.510,
    "right_hip": 1.260,
    "right_knee": 2.510,
}

# Matches `scorer.TORQUE_TOLERANCE`. A torque clipped to exactly the rating and
# one a rounding error below it must be counted the same way.
SATURATION_TOLERANCE = 0.001

TORQUE_LIMIT_NM = {
    "leg": state_mod.LEG_TORQUE_LIMIT_NM,
    "wheel": state_mod.WHEEL_TORQUE_LIMIT_NM,
}

DEFAULT_STEPS = 30_000
EVAL_SECONDS = eval_set.EPISODE_SECONDS
EVAL_STEPS = eval_set.step_limit(reward_env.FREQUENCY)
PUSH_STEP = eval_set.push_step(reward_env.FREQUENCY)
PUSH_HOLD_STEPS = max(1, int(eval_set.PUSH_DURATION_S * reward_env.FREQUENCY))


def name() -> str:
    return "pybullet-ppo"


class _TermRecorder:
    """Averages each reward term over an evaluation, at one checkpoint."""

    def __init__(self):
        self.sums = {}
        self.count = 0

    def add(self, terms):
        for key, value in terms.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.count += 1

    def means(self):
        if not self.count:
            return {}
        return {k: v / self.count for k, v in self.sums.items()}


def _make_checkpoint_callback(total_steps):
    """Record each reward term's average, at 10/25/50/100% of training.

    It reads the terms out of the training rollouts as they happen. The first
    version instead paused training at each checkpoint and drove the SAME
    environment for 600 evaluation steps -- which reset the env and changed its
    commanded speed underneath PPO, between chunks of `learn()`. Measured cost
    of that interference, same reward and same seed at 500k steps:

        pausing to evaluate on the training env:   32.5 s survived in total
        reading terms from the rollouts:           52.3 s survived in total

    So the measurement was damaging the thing it measured. Reading the rollouts
    costs nothing, disturbs nothing, and reports what the policy actually did.
    """
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415

    class _Checkpoints(BaseCallback):
        def __init__(self):
            super().__init__()
            self.values = {}
            self.totals = []
            self.checkpoints_hit = []
            self._window = _TermRecorder()
            self._next = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                terms = info.get("reward_terms")
                if terms:
                    self._window.add(terms)

            while (self._next < len(CHECKPOINTS)
                   and self.num_timesteps >= total_steps * CHECKPOINTS[self._next]):
                means = self._window.means()
                for key, value in means.items():
                    self.values.setdefault(key, []).append(round(value, 4))
                self.totals.append(round(sum(means.values()), 4))
                self.checkpoints_hit.append(CHECKPOINTS[self._next])
                self._window = _TermRecorder()
                self._next += 1
            return True

    return _Checkpoints()


def _rollout_episode(model, env, wrapper, episode):
    """One evaluation episode with the trained policy. Real measurements.

    Reuses the open environment -- see `reward_env.make_env`.
    """
    wrapper.fixed_speed = episode["commanded_speed"]
    if True:
        observation, info = env.reset(seed=episode["seed"])
        backend = wrapper.env.unwrapped.backend

        worst_pitch = 0.0
        peak = {"leg": 0.0, "wheel": 0.0}
        square_sum = {"leg": 0.0, "wheel": 0.0}
        # Control steps where at least one motor of that class sat at its rated
        # limit. Counted per step, not per joint, so the denominator is just
        # steps_survived and the fraction reads as "share of the episode".
        saturated_steps = {"leg": 0, "wheel": 0}
        margin_rad = min(JOINT_LIMITS_RAD.values())
        speed_error_sum = 0.0
        steps_done = 0
        fell = False

        recovered = None
        recovery_step = None
        hold_needed = int(eval_set.RECOVERY_HOLD_S * reward_env.FREQUENCY)
        hold_run = 0

        is_push = episode["kind"] == "push"
        pushed_at = None
        disturbed = False  # has the shove actually moved it past the threshold

        for step in range(EVAL_STEPS):
            if is_push:
                if step == PUSH_STEP:
                    _apply_push(backend, eval_set.PUSH_NEWTONS)
                    pushed_at = step
                elif step == PUSH_STEP + PUSH_HOLD_STEPS:
                    _apply_push(backend, 0.0)

            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            steps_done = step + 1

            spine = info["spine_observation"]
            pitch = float(spine["base_orientation"]["pitch"])
            worst_pitch = max(worst_pitch, abs(pitch))
            speed_error_sum += abs(
                float(spine["wheel_odometry"]["velocity"])
                - episode["commanded_speed"]
            )

            at_limit = {"leg": False, "wheel": False}
            for joint, servo in spine["servo"].items():
                torque = abs(float(servo["torque"]))
                kind = "wheel" if "wheel" in joint else "leg"
                peak[kind] = max(peak[kind], torque)
                square_sum[kind] += torque * torque
                if torque >= TORQUE_LIMIT_NM[kind] * (1.0 - SATURATION_TOLERANCE):
                    at_limit[kind] = True
                if joint in JOINT_LIMITS_RAD:
                    margin_rad = min(
                        margin_rad,
                        JOINT_LIMITS_RAD[joint] - abs(float(servo["position"])),
                    )
            for kind, hit in at_limit.items():
                if hit:
                    saturated_steps[kind] += 1

            # Recovery is only meaningful once the shove has actually
            # disturbed the robot. Counting from the moment of the push made
            # every episode report "recovered" in about zero seconds, because
            # the robot was still upright at the instant the force was applied.
            if is_push and pushed_at is not None and recovery_step is None:
                if not disturbed:
                    if abs(pitch) >= eval_set.RECOVERY_PITCH_RAD:
                        disturbed = True
                    hold_run = 0
                elif abs(pitch) < eval_set.RECOVERY_PITCH_RAD:
                    hold_run += 1
                    if hold_run >= hold_needed:
                        recovery_step = step - hold_needed
                else:
                    hold_run = 0

            if terminated or truncated:
                # `truncated` is TimeLimit firing because the episode reached its
                # step limit -- that is the robot COMPLETING the episode, not
                # failing it. Only `terminated` is a fall: Upkie's env terminates
                # when pitch passes its fall threshold.
                #
                # This read `fell = True` for both, so every policy that survived
                # the full 20 s was recorded as having fallen and the `fell_over`
                # violation fired on successful episodes. It did not reach the
                # score -- survival_frac comes from steps_survived, which was
                # correct -- but it was wrong in the record, and wrong in the
                # direction that made our own candidates look worse.
                fell = bool(terminated)
                break

        seconds = steps_done / reward_env.FREQUENCY
        leg_steps = max(1, steps_done * 4)
        wheel_steps = max(1, steps_done * 2)

        record = {
            "kind": episode["kind"],
            "seed": episode["seed"],
            "commanded_speed": episode["commanded_speed"],
            "commanded_yaw_rate": 0.0,
            "episode_seconds": round(seconds, 3),
            "steps_survived": steps_done,
            "step_limit": EVAL_STEPS,
            "mean_speed_error": round(speed_error_sum / max(1, steps_done), 4),
            "net_displacement_m": round(
                float(spine["wheel_odometry"]["position"]), 4
            ),
            "heading_error_deg": 0.0,  # the Pendulum env has no yaw at all
            "max_pitch_excursion_deg": round(math.degrees(worst_pitch), 3),
            "closest_joint_margin_deg": round(math.degrees(margin_rad), 3),
            "peak_torque_nm": {
                "leg": round(peak["leg"], 4),
                "wheel": round(peak["wheel"], 4),
            },
            "rms_torque_nm": {
                "leg": round(math.sqrt(square_sum["leg"] / leg_steps), 4),
                "wheel": round(math.sqrt(square_sum["wheel"] / wheel_steps), 4),
            },
            "saturated_steps": dict(saturated_steps),
            "fell": fell,
        }

        if is_push:
            # Never knocked past the threshold at all: the shove did not
            # disturb it, which is not the same as recovering from one.
            recovered = recovery_step is not None
            record["disturbed"] = disturbed
            record["recovered"] = recovered
            record["recovery_time_s"] = (
                round((recovery_step - PUSH_STEP) / reward_env.FREQUENCY, 3)
                if recovered
                else round(seconds, 3)  # the episode cap, per INTERFACE.md
            )
        return record


def _apply_push(backend, newtons):
    """The pre-registered shove, applied to the base link."""
    from upkie.utils.external_force import ExternalForce  # noqa: PLC0415

    backend.set_external_forces(
        {"base": ExternalForce(force=[newtons, 0.0, 0.0], local=False)}
    )


def train(
    reward_fn,
    candidate_id: str,
    round_number: int,
    goal: str,
    seed: int = 0,
    total_steps: int = DEFAULT_STEPS,
    out_dir: str = "runs",
    eval_episodes: int | None = None,
) -> dict:
    """Train one candidate with PPO and measure the result."""
    started = time.time()
    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "round": round_number,
        "trainer": name(),
        "status": "trained",
        "error": None,
        "terms": None,
        "rollout": None,
        "policy_path": None,
        "video_path": None,
        "train_seconds": 0.0,
        "compute_seconds": 0.0,
    }

    try:
        from stable_baselines3 import PPO  # noqa: PLC0415

        env = reward_env.make_env(reward_fn, seed=seed, gui=False,
                                  max_steps=EVAL_STEPS)
        wrapper = reward_env.find_reward_wrapper(env)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")

        callback = _make_checkpoint_callback(total_steps)
        model.learn(total_timesteps=total_steps, callback=callback,
                    progress_bar=False)

        policy_dir = Path(out_dir) / candidate_id
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = policy_dir / "policy.zip"
        model.save(policy_path)

        result["terms"] = {
            "checkpoints": callback.checkpoints_hit,
            "values": callback.values,
            "total": callback.totals,
        }
        result["policy_path"] = str(policy_path)
        result["train_seconds"] = round(time.time() - started, 1)

        wanted = eval_set.episodes()
        if eval_episodes is not None:
            wanted = wanted[:eval_episodes]
        episodes = [_rollout_episode(model, env, wrapper, e) for e in wanted]
        env.close()
        result["rollout"] = {
            "schema_version": SCHEMA_VERSION,
            "episodes": episodes,
        }
    except ValueError as error:
        if "returned" in str(error):
            result["status"] = "dead_nan"
        else:
            result["status"] = "dead_gate2"
        result["error"] = traceback.format_exc()
    except Exception:
        result["status"] = "dead_gate2"
        result["error"] = traceback.format_exc()

    result["compute_seconds"] = round(time.time() - started, 1)
    if not result["train_seconds"]:
        result["train_seconds"] = result["compute_seconds"]
    return result
