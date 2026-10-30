import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Alternative formulation:
    - speed_exp: exponential penalty for speed error (softer near zero, harsher far away).
    - yaw_exp:   same for yaw rate.
    - balance_gain: reward proportional to inverse of pitch variance over short window (approximated by |pitch|).
    - joint_acc_penalty: penalizes rapid changes in joint velocities approximated by current velocity magnitude.
    - slip_penalty: heavy penalty when contact is lost.
    - efficiency_bonus: reward for low torque magnitude (encourages gliding).
    """
    # speed error → exponential cost
    sp_err = state.get('commanded_speed', 0.0) - state.get('ground_velocity', 0.0)
    speed_exp = -math.exp(abs(sp_err) / 0.5) + 1.0   # max 0 at zero error, goes negative

    # yaw error → exponential cost
    y_err = state.get('commanded_yaw_rate', 0.0) - state.get('yaw_rate', 0.0)
    yaw_exp = -math.exp(abs(y_err) / 0.5) + 1.0

    # balance gain: higher when upright (small |pitch|)
    pitch = abs(state.get('pitch', 0.0))
    balance_gain = math.exp(-pitch * 5.0)   # decays quickly with tilt

    # joint acceleration proxy: use joint velocity magnitude
    joint_acc = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_acc += abs(v)
    joint_acc_penalty = -0.0002 * joint_acc

    # slip penalty: large negative if contact lost
    contact = state.get('contact', 0.0)
    slip_penalty = -10.0 * (1.0 - contact)   # -10 when no contact, 0 when contact=1

    # efficiency bonus: low torque
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += abs(t)
    efficiency_bonus = 0.001 * (1.0 / (1.0 + torque_sum))   # small positive

    return {
        'speed_exp': speed_exp,
        'yaw_exp': yaw_exp,
        'balance_gain': balance_gain,
        'joint_acc_penalty': joint_acc_penalty,
        'slip_penalty': slip_penalty,
        'efficiency_bonus': efficiency_bonus,
    }
