import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - speed_err_lin: negative absolute speed error (encourages reaching commanded speed).
    - yaw_err_lin: negative absolute yaw rate error.
    - upright_bonus: reward near upright using a cosine shape (max 1 at pitch=0).
    - action_magnitude: penalize large action values (smooth control).
    - joint_symmetry: reward for keeping left/right wheel speeds similar.
    """
    # speed tracking (L1)
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    speed_err_lin = -abs(cmd_speed - act_speed)

    # yaw tracking (L1)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    yaw_err_lin = -abs(cmd_yaw - act_yaw)

    # upright bonus (cosine based, max 1 at pitch=0, zero at +/-pi/2)
    pitch = state.get('pitch', 0.0)
    upright_bonus = max(0.0, math.cos(pitch))

    # action magnitude penalty
    ground_vel_cmd = action.get('ground_velocity', 0.0)
    yaw_vel_cmd = action.get('yaw_velocity', 0.0)
    action_magnitude = -(ground_vel_cmd * ground_vel_cmd + yaw_vel_cmd * yaw_vel_cmd) * 0.001

    # wheel symmetry reward (encourage equal wheel speeds)
    left_wheel_v = state.get('left_wheel_velocity', 0.0)
    right_wheel_v = state.get('right_wheel_velocity', 0.0)
    wheel_symmetry = -abs(left_wheel_v - right_wheel_v) * 0.005

    return {
        'speed_err_lin': speed_err_lin,
        'yaw_err_lin': yaw_err_lin,
        'upright_bonus': upright_bonus,
        'action_magnitude': action_magnitude,
        'wheel_symmetry': wheel_symmetry,
    }
