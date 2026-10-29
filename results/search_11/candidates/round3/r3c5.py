def compute_reward(state: dict, action: dict) -> dict:
    """Multi‑objective shaping: combine exponential survival, cosine alignment,
    and a smooth speed‑error penalty. All terms stay bounded and smooth."""
    r = {}

    # Exponential survival: decays with pitch error and loss of contact
    pitch_err = abs(state['pitch'])
    contact_factor = 1.0 if state['contact'] > 0.5 else 0.0
    r['survival_exp'] = math.exp(-pitch_err * 3.0) * contact_factor

    # Cosine alignment for forward direction: reward when velocity points forward
    # (assumes forward is positive x). Use sign of velocity vs commanded speed.
    vel = state['ground_velocity']
    cmd = state['commanded_speed']
    align = 1.0 if (vel * cmd) > 0.0 else -1.0
    r['forward_align'] = 0.4 * align

    # Speed error penalty: soft L1 with smoothing
    speed_err = abs(vel - cmd)
    r['speed_penalty'] = -0.6 * (speed_err / (1.0 + speed_err))

    # Joint limit soft constraint: use logistic penalty
    joint_names = [
        'left_hip_position', 'right_hip_position',
        'left_knee_position', 'right_knee_position'
    ]
    joint_pen = 0.0
    for jn in joint_names:
        ang = abs(state.get(jn, 0.0))
        # penalty = -1 / (1 + exp(-k*(ang - a0)))  → 0 at small ang, → -1 at large
        joint_pen += -1.0 / (1.0 + math.exp(-5.0 * (ang - 0.8)))
    r['joint_logistic'] = joint_pen

    # Small torque cost to discourage excessive actuation
    torque_sq = (
        state['left_hip_torque']**2 + state['right_hip_torque']**2 +
        state['left_knee_torque']**2 + state['right_knee_torque']**2 +
        state['left_wheel_torque']**2 + state['right_wheel_torque']**2
    )
    r['torque_cost'] = -0.0004 * torque_sq

    return r
