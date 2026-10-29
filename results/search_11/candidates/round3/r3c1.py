import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward based on upright orientation, speed tracking, contact maintenance,
    and smooth actuator effort. Uses exponential decay for pitch and speed errors
    to give soft penalties that never blow up."""
    r = {}

    # Upright bonus: cos(pitch) ∈ [0,1] when near vertical, negative when tipping
    pitch = state['pitch']
    r['upright'] = math.cos(pitch)  # max 1 at pitch=0, decreases smoothly

    # Speed tracking reward: exponential penalty on error
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_track'] = math.exp(-abs(speed_err))  # 1 at zero error, →0 as error grows

    # Contact encouragement: binary bonus when wheels touch ground
    r['contact'] = 1.0 if state['contact'] > 0.5 else 0.0

    # Effort penalty: quadratic cost on all torques (legs + wheels)
    torque_sq = (
        state['left_hip_torque']**2 + state['right_hip_torque']**2 +
        state['left_knee_torque']**2 + state['right_knee_torque']**2 +
        state['left_wheel_torque']**2 + state['right_wheel_torque']**2
    )
    r['effort'] = -0.0005 * torque_sq

    # Joint safety: soft barrier that blows up near ±1.5 rad (~86°)
    joint_names = [
        'left_hip_position', 'right_hip_position',
        'left_knee_position', 'right_knee_position',
        'left_wheel_position', 'right_wheel_position'
    ]
    barrier = 0.0
    limit = 1.5
    for jn in joint_names:
        ang = abs(state.get(jn, 0.0))
        if ang < limit:
            # -log(1 - (ang/limit)^2) → 0 at centre, →∞ as ang→limit
            barrier += -math.log(1.0 - (ang/limit)*(ang/limit))
        else:
            # If exceeded, assign a large negative constant
            barrier += -10.0
    r['joint_barrier'] = barrier

    return r
