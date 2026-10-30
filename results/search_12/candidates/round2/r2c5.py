import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Multi‑objective reward using hyperbolic tangent to bound terms.
    - speed_tanh: tanh of scaled speed error (reward when close to command).
    - pitch_tanh: negative tanh of absolute pitch (penalises lean).
    - yaw_tanh: negative tanh of absolute yaw (keeps heading straight).
    - effort_tanh: negative tanh of normalized total torque.
    - step_bonus: small constant for taking a step (encourages longevity).
    """
    cmd = state.get('commanded_speed', 0.0)
    vel = state.get('ground_velocity', 0.0)
    speed_err = cmd - vel
    speed_tanh = math.tanh(2.0 * speed_err)   # range [-1,1]; positive when slower than cmd

    pitch = state.get('pitch', 0.0)
    pitch_tanh = -math.tanh(abs(pitch))       # always ≤0, zero when upright

    yaw = state.get('yaw', 0.0)
    yaw_tanh = -math.tanh(abs(yaw))           # penalise heading drift

    # total torque normalisation (approx max 10 Nm per joint → 60 Nm)
    tot_torque = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            tot_torque += abs(state.get(f'{side}_{joint}_torque', 0.0))
    effort_tanh = -math.tanh(tot_torque / 30.0)   # scale so typical values give modest penalty

    step_bonus = 0.001

    return {
        'speed_tanh': speed_tanh,
        'pitch_tanh': pitch_tanh,
        'yaw_tanh': yaw_tanh,
        'effort_tanh': effort_tanh,
        'step_bonus': step_bonus,
    }
