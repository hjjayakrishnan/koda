import math

def compute_reward(state: dict, action: dict) -> dict:
    """Survival‑shaped reward: stay alive (contact & upright) gives base reward,
    speed tracking adds a bonus, and large torques or joint angles are punished
    with a hyperbolic penalty that stays bounded."""
    r = {}

    # Survival bonus: 1.0 per step when robot has contact and is not tilted too far
    upright_ok = abs(state['pitch']) < 0.4   # ~23°
    contact_ok = state['contact'] > 0.5
    r['survival'] = 1.0 if (upright_ok and contact_ok) else 0.0

    # Speed tracking bonus: scaled tanh of error (max 0.5)
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_bonus'] = 0.5 * math.tanh(-abs(speed_err) + 0.1)  # peaks near zero error

    # Joint limit penalty: hyperbolic that grows slowly then steeply
    joint_names = [
        'left_hip_position', 'right_hip_position',
        'left_knee_position', 'right_knee_position'
    ]
    joint_pen = 0.0
    for jn in joint_names:
        ang = abs(state.get(jn, 0.0))
        # penalty = ang^2 / (1 + ang^2)  → 0 at 0, →1 as ang→∞
        joint_pen += -(ang*ang) / (1.0 + ang*ang)
    r['joint_hyperbolic'] = joint_pen

    # Effort penalty: linear in absolute torque (less aggressive than quadratic)
    torque_abs = (
        abs(state['left_hip_torque']) + abs(state['right_hip_torque']) +
        abs(state['left_knee_torque']) + abs(state['right_knee_torque']) +
        abs(state['left_wheel_torque']) + abs(state['right_wheel_torque'])
    )
    r['effort_lin'] = -0.02 * torque_abs

    return r
