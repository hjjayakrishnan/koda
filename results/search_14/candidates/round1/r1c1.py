import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward components:
    - speed_error: negative quadratic penalty for forward speed tracking.
    - yaw_error: negative quadratic penalty for yaw rate tracking.
    - upright_exp: exponential reward for staying near upright (max 1 at zero pitch).
    - power_cost: penalizes instantaneous mechanical power (|torque*velocity|) summed over joints.
    - contact_reward: small bonus when wheels are in contact with ground.
    - joint_symmetry: penalizes asymmetry between left and right hip/knee positions."""
    # speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    sp_err = cmd_speed - act_speed
    speed_error = -0.6 * sp_err * sp_err

    # yaw tracking
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    y_err = cmd_yaw - act_yaw
    yaw_error = -0.4 * y_err * y_err

    # uprightness via exponential (softer than Gaussian near zero)
    pitch = state.get('pitch', 0.0)
    upright_exp = math.exp(-abs(pitch) / 0.15)  # decays to ~0.5 at 0.15 rad (~8.6°)

    # power consumption (torque * angular velocity) penalty
    power_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            power_sum += abs(t * v)
    power_cost = -0.0005 * power_sum

    # contact encouragement
    contact = state.get('contact', 0.0)
    contact_reward = 0.02 * contact

    # symmetry reward/punishment: aim for mirrored hip/knee angles
    sym_err = 0.0
    for joint in ('hip', 'knee'):
        lp = state.get(f'left_{joint}_position', 0.0)
        rp = state.get(f'right_{joint}_position', 0.0)
        sym_err += (lp + rp) ** 2  # penalize when sum != 0 (mirrored about zero)
    joint_symmetry = -0.001 * sym_err

    return {
        'speed_error': speed_error,
        'yaw_error': yaw_error,
        'upright_exp': upright_exp,
        'power_cost': power_cost,
        'contact_reward': contact_reward,
        'joint_symmetry': joint_symmetry,
    }
