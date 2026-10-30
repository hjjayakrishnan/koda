#!/usr/bin/env python3
"""Render a policy or the MPC controller straight to PNG frames.

    ./.venv/bin/python record_clip.py --what mpc      --out clips/1_mpc
    ./.venv/bin/python record_clip.py --what thrash   --out clips/2_thrash
    ./.venv/bin/python record_clip.py --what fixed    --out clips/3_fixed

No screen recorder, no window to position, no desktop capture. The frames come
from the simulator itself with `getCameraImage`, so the clip is identical every
time it is run and does not depend on what else is on screen.

Encode afterwards, once ffmpeg exists:

    ffmpeg -framerate 30 -i clips/1_mpc/%05d.png -pix_fmt yuv420p clips/1_mpc.mp4
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pybullet

sys.path.insert(0, str(Path(__file__).resolve().parent))

from koda import eval_set, gate1, reward_env  # noqa: E402
from koda import trainer_pybullet as T  # noqa: E402

W, H = 960, 540
EVERY = 7          # control steps per frame: 200 Hz / 7 ~= 29 fps

CLIPS = {
    "mpc": ("Upkie's hand-written MPC controller", None, None),
    "thrash": ("Ranked best by our first grader",
               "runs/search_12/seed11/r3c2/policy.zip",
               "results/search_12/candidates/round3/r3c2.py"),
    "fixed": ("Same search, after the learner fix",
              "runs/r3c1_zoohp/policy.zip",
              "results/search_12/candidates/round3/r3c1.py"),
}


def write_png(path, image):
    """Save an (H, W, 3) uint8 array. Standard library only."""
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(h))

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def frame(client, follow_x):
    view = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[follow_x, 0.0, 0.35], distance=2.2,
        yaw=50.0, pitch=-15.0, roll=0.0, upAxisIndex=2,
        physicsClientId=client)
    proj = pybullet.computeProjectionMatrixFOV(
        fov=60.0, aspect=W / H, nearVal=0.1, farVal=30.0,
        physicsClientId=client)
    _, _, rgba, _, _ = pybullet.getCameraImage(
        width=W, height=H, viewMatrix=view, projectionMatrix=proj,
        renderer=pybullet.ER_TINY_RENDERER, physicsClientId=client)
    return np.reshape(np.array(rgba, dtype=np.uint8), (H, W, 4))[:, :, :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", required=True, choices=sorted(CLIPS))
    p.add_argument("--out", required=True)
    p.add_argument("--speed", type=float, default=0.6)
    p.add_argument("--seconds", type=float, default=20.0)
    args = p.parse_args()

    label, policy_path, reward_path = CLIPS[args.what]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{label}\n  {args.speed} m/s, {args.seconds:.0f} s -> {out}/")

    if args.what == "mpc":
        reward_fn = lambda state, action: {}          # noqa: E731
    else:
        reward_fn = gate1.check(Path(reward_path).read_text(), "clip")["reward_fn"]

    env = reward_env.make_env(reward_fn, seed=11, gui=False,
                              fixed_speed=args.speed,
                              max_steps=T.EVAL_STEPS)
    wrapper = reward_env.find_reward_wrapper(env)
    client = int(wrapper.env.unwrapped.backend._bullet)

    if args.what == "mpc":
        from upkie.controllers import MPCBalancer
        balancer = MPCBalancer()
        dt = 1.0 / reward_env.FREQUENCY

        class Policy:
            def __init__(self): self.info = {}
            def predict(self, obs, deterministic=True):
                spine = self.info.get("spine_observation")
                if spine is None:
                    return np.array([0.0], dtype=np.float32), None
                try:
                    v = balancer.step(target_ground_velocity=args.speed,
                                      spine_observation=spine, dt=dt)
                except Exception:
                    v = 0.0
                return np.array([v], dtype=np.float32), None
        policy = Policy()
    else:
        policy = T.load_policy(policy_path)

    obs, info = env.reset(seed=11)
    if args.what == "mpc":
        policy.info = info
    n = 0
    for step in range(int(args.seconds * reward_env.FREQUENCY)):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        if args.what == "mpc":
            policy.info = info
        if step % EVERY == 0:
            x = float(info["spine_observation"]["wheel_odometry"]["position"])
            write_png(out / f"{n:05d}.png", frame(client, x))
            n += 1
        if term or trunc:
            print(f"  fell at {step / reward_env.FREQUENCY:.2f} s")
            break
    env.close()
    print(f"  {n} frames")
    print(f"  ffmpeg -framerate 30 -i {out}/%05d.png -pix_fmt yuv420p {out}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
