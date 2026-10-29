# SPDX-License-Identifier: Apache-2.0
"""Run Upkie's own shipped PD controller and report what it did.

    .venv/bin/python check_pd_baseline.py

This is the Stage B gate check, and nothing more. It is NOT the baseline
integration -- how a hand-engineered controller should be recorded alongside the
reward candidates is still an open decision, so this script writes its own files
and touches neither `koda/orchestrator.py` nor the run record.

What it does:

  * runs `pd_balancing.py`'s controller, verbatim, for 5 episodes of 20 s
  * collects a rollout in the `INTERFACE.md` section 3 shape
  * grades that rollout with the stand-in grader
  * saves camera frames as one PNG contact sheet, written with the standard
    library -- there is no ffmpeg on this machine and no new dependency here

Out:
    out/pd_baseline/rollout.json      the physical measurements
    out/pd_baseline/score.json        the grader's numbers
    out/pd_baseline/contact_sheet.png what it looked like
"""

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet

import upkie.envs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koda import SCHEMA_VERSION, eval_set, scorer_stub  # noqa: E402
from koda import state as state_mod  # noqa: E402

upkie.envs.register()

FREQUENCY = 200.0
EPISODE_SECONDS = 20.0
STEPS = int(FREQUENCY * EPISODE_SECONDS)

# Hip and knee travel limits, read from the installed upkie model, not guessed.
# The wheels spin freely (+/- infinity) so they have no angle margin.
JOINT_LIMITS_RAD = {
    "left_hip": 1.260,
    "left_knee": 2.510,
    "right_hip": 1.260,
    "right_knee": 2.510,
}

FRAME_STEPS = 8  # frames in the contact sheet
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


def pd_action(observation):
    """Verbatim from upkie/examples/pybullet/pd_balancing.py."""
    pitch = observation[0]
    ground_position = observation[1]
    ground_velocity = observation[3]
    return np.clip(
        a=[10.0 * pitch + 1.0 * ground_position + 0.1 * ground_velocity],
        a_min=-0.99,
        a_max=0.99,
    )


def write_png(path: Path, image: np.ndarray) -> None:
    """Save an (H, W, 3) uint8 array as a PNG. Standard library only."""
    height, width, _ = image.shape
    raw = b"".join(
        b"\x00" + image[row].tobytes() for row in range(height)
    )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def grab_frame(bullet_client) -> np.ndarray:
    """One camera image, rendered on the CPU. No display needed."""
    view = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.35],
        distance=1.3,
        yaw=50.0,
        pitch=-15.0,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=bullet_client,
    )
    projection = pybullet.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=FRAME_WIDTH / FRAME_HEIGHT,
        nearVal=0.1,
        farVal=10.0,
        physicsClientId=bullet_client,
    )
    _, _, rgba, _, _ = pybullet.getCameraImage(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=pybullet.ER_TINY_RENDERER,
        physicsClientId=bullet_client,
    )
    return np.reshape(np.array(rgba, dtype=np.uint8), (FRAME_HEIGHT, FRAME_WIDTH, 4))[
        :, :, :3
    ]


def label_frame(frame: np.ndarray, seconds: float, pitch_deg: float) -> np.ndarray:
    """A crude tick mark so the frames can be told apart in order."""
    marked = frame.copy()
    bar = int((seconds / EPISODE_SECONDS) * (FRAME_WIDTH - 8))
    marked[-6:-2, 4 : 4 + max(2, bar)] = [255, 80, 80]
    middle = FRAME_HEIGHT // 2
    lean = int(np.clip(pitch_deg * 4.0, -middle + 6, middle - 6))
    marked[middle - 1 : middle + 1, 4:10] = [60, 200, 60]
    marked[middle - 1 + lean : middle + 1 + lean, 12:24] = [60, 200, 60]
    return marked


def run_episode(seed: int, want_frames: bool, gui: bool = False):
    """One 20-second episode. Returns the rollout record and any frames.

    gui=True opens a PyBullet window so a person can watch it run. It is
    slower than headless and it needs a display, so it is off by default.
    """
    frames = []
    frame_every = STEPS // FRAME_STEPS

    with gym.make(
        "Upkie-PyBullet-Pendulum",
        frequency=FREQUENCY,
        gui=gui,
        frequency_checks=False,
    ) as env:
        observation, info = env.reset(seed=seed)
        bullet_client = env.unwrapped.backend._bullet

        worst_pitch = 0.0
        peak = {"leg": 0.0, "wheel": 0.0}
        square_sum = {"leg": 0.0, "wheel": 0.0}
        margin_rad = min(JOINT_LIMITS_RAD.values())
        speed_error_sum = 0.0
        steps_done = 0
        fell = False

        for step in range(STEPS):
            action = pd_action(observation)
            observation, _reward, terminated, truncated, info = env.step(action)
            steps_done = step + 1

            spine = info["spine_observation"]
            pitch = float(spine["base_orientation"]["pitch"])
            worst_pitch = max(worst_pitch, abs(pitch))

            # The PD controller is never given a speed command, so the task it
            # is performing is "hold still". Error is measured against zero.
            speed_error_sum += abs(float(spine["wheel_odometry"]["velocity"]))

            for joint, servo in spine["servo"].items():
                torque = abs(float(servo["torque"]))
                kind = "wheel" if "wheel" in joint else "leg"
                peak[kind] = max(peak[kind], torque)
                square_sum[kind] += torque * torque
                if joint in JOINT_LIMITS_RAD:
                    margin_rad = min(
                        margin_rad,
                        JOINT_LIMITS_RAD[joint] - abs(float(servo["position"])),
                    )

            if want_frames and step % frame_every == 0 and len(frames) < FRAME_STEPS:
                frames.append(
                    label_frame(
                        grab_frame(bullet_client),
                        step / FREQUENCY,
                        math.degrees(pitch),
                    )
                )

            if terminated or truncated:
                fell = True
                break

        seconds = steps_done / FREQUENCY
        final_ground = float(spine["wheel_odometry"]["position"])
        leg_steps = steps_done * 4
        wheel_steps = steps_done * 2

        record = {
            "kind": "task",
            "seed": seed,
            "commanded_speed": 0.0,
            "commanded_yaw_rate": 0.0,
            "episode_seconds": round(seconds, 3),
            "steps_survived": steps_done,
            "step_limit": STEPS,
            "mean_speed_error": round(speed_error_sum / steps_done, 4),
            "net_displacement_m": round(final_ground, 4),
            "heading_error_deg": 0.0,  # the pendulum env has no yaw at all
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
            "fell": fell,
        }
        return record, frames


def contact_sheet(frames, path: Path) -> None:
    columns = 4
    rows = math.ceil(len(frames) / columns)
    sheet = np.full(
        (rows * FRAME_HEIGHT, columns * FRAME_WIDTH, 3), 20, dtype=np.uint8
    )
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[
            row * FRAME_HEIGHT : (row + 1) * FRAME_HEIGHT,
            column * FRAME_WIDTH : (column + 1) * FRAME_WIDTH,
        ] = frame
    write_png(path, sheet)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Upkie's own PD balancing controller and score it."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window, numbers only. A window is the default -- a person "
             "should be able to watch the robot without knowing a flag.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=len(eval_set.SEEDS),
        help="how many episodes to run (default: the whole eval set)",
    )
    args = parser.parse_args()

    out = Path("out/pd_baseline")
    out.mkdir(parents=True, exist_ok=True)

    seeds = eval_set.SEEDS[: max(1, args.episodes)]
    print(f"Upkie's shipped PD controller, {len(seeds)} episode(s) of "
          f"{EPISODE_SECONDS:.0f} s at {FREQUENCY:.0f} Hz, "
          f"{'headless' if args.headless else 'GUI window'}.\n")
    if not args.headless:
        print("  A PyBullet window should open. Close it or press Ctrl-C to stop.\n")

    episodes = []
    frames = []
    for index, seed in enumerate(seeds):
        record, got = run_episode(seed, want_frames=(index == 0), gui=not args.headless)
        episodes.append(record)
        frames += got
        print(
            f"  seed {seed:3d}  "
            f"{'FELL at' if record['fell'] else 'stayed up'} "
            f"{record['episode_seconds']:5.2f} s  "
            f"worst pitch {record['max_pitch_excursion_deg']:5.2f} deg  "
            f"drift {record['net_displacement_m']:+.3f} m  "
            f"wheel torque peak {record['peak_torque_nm']['wheel']:.3f} Nm"
        )

    rollout = {"schema_version": SCHEMA_VERSION, "episodes": episodes}
    (out / "rollout.json").write_text(json.dumps(rollout, indent=2))

    score = scorer_stub.score(rollout, "upkie_pd_controller")
    (out / "score.json").write_text(json.dumps(score, indent=2))

    if frames:
        contact_sheet(frames, out / "contact_sheet.png")

    survived = sum(1 for e in episodes if not e["fell"])
    print()
    print(f"  stayed up for the full {EPISODE_SECONDS:.0f} s: "
          f"{survived} of {len(episodes)} episodes")
    print()
    print("  the grader's numbers on this rollout:")
    for key, value in score["measurements"].items():
        print(f"    {key:28} {value}")
    print(f"    {'violations':28} {score['violations'] or 'none'}")
    print()
    print(f"  wheel motors are rated {state_mod.WHEEL_TORQUE_LIMIT_NM} Nm, "
          f"leg motors {state_mod.LEG_TORQUE_LIMIT_NM} Nm "
          "(read from the installed upkie model)")
    print()
    print(f"  wrote {out}/rollout.json, {out}/score.json, "
          f"{out}/contact_sheet.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
