import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Components:
    - pos_error: negative squared error of forward position relative to commanded integration.
    - yaw_align: cosine similarity reward for yaw alignment.
    - stability: exponential decay on pitch and pitch_rate.
    - action_smooth: penalty on large changes in action (jerk).
    - contact: small reward for maintaining ground contact.
    """
    # integrate commanded speed to get desired position increment
    dt = state.get('dt', 0.0)
    cmd_speed = state.get('commanded_speed', 0.0)
    desired_pos_inc = cmd_speed * dt
    actual_pos_inc = state.get('ground_velocity', 0.0) * dt
    pos_err = desired_pos_inc - actual_pos_inc
    pos_error = -0.8 * pos_err * pos_err

    # yaw alignment: reward cos of yaw error (max 1 when aligned)
    yaw_err = state.get('yaw', 0.0)  # yaw relative to start; we want zero yaw unless commanded turn
    yaw_align = math.cos(yaw_err)  # range [-1,1]; shift later to [0,1] by (1+cos)/2
    yaw_align = 0.5 * (yaw_align + 1.0)  # now 0..1
    yaw_align = yaw_align - 0.5  # make zero at random orientation, positive when aligned

    # stability: penalize pitch and pitch_rate
    pitch = state.get('pitch', 0.0)
    pitch_rate = state.get('pitch_rate', 0.0)
    stability = -0.5 * (pitch * pitch + 0.1 * pitch_rate * pitch_rate)

    # action smoothness: penalize large action values (jerk)
    ground_vel_cmd = action.get('ground_velocity', 0.0)
    yaw_vel_cmd = action.get('yaw_velocity', 0.0)
    action_smooth = -0.01 * (ground_vel_cmd * ground_vel_cmd + yaw_vel_cmd * yaw_vel_cmd)

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.02 * contact

    return {
        'pos_error': pos_error,
        'yaw_align': yaw_align,
        'stability': stability,
        'action_smooth': action_smooth,
        'contact_bonus': contact_bonus,
    }
