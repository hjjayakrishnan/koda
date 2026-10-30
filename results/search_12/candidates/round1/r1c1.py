import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - bal_pen: quadratic penalty for pitch deviation and pitch rate.
    - speed_err: negative squared error between actual ground velocity and commanded speed.
    - yaw_pen: penalty for yaw rate deviation from commanded yaw rate (encourages straight travel).
    - effort: small penalty on summed squared motor torques (encourages efficient control).
    - contact_bonus: small bonus when at least one wheel touches the ground.
    """
    # Balancing: keep pitch near zero and avoid fast tilting
    pitch = state.get('pitch', 0.0)
    pitch_rate = state.get('pitch_rate', 0.0)
    bal_pen = -(pitch * pitch + 0.1 * pitch_rate * pitch_rate)

    # Speed tracking: reward matching commanded forward speed
    ground_vel = state.get('ground_velocity', 0.0)
    cmd_speed = state.get('commanded_speed', 0.0)
    speed_err = -(ground_vel - cmd_speed) ** 2

    # Yaw tracking: keep yaw rate close to commanded yaw rate (often zero)
    yaw_rate = state.get('yaw_rate', 0.0)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    yaw_pen = -(yaw_rate - cmd_yaw) ** 2

    # Effort: penalize large motor torques (sum over all actuators)
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            torque_sum += state.get(f'{side}_{joint}_torque', 0.0) ** 2
    effort = -0.001 * torque_sum

    # Contact bonus: encourage staying grounded
    contact = state.get('contact', 0)
    contact_bonus = 0.5 if contact > 0.5 else 0.0

    return {
        'bal_pen': bal_pen,
        'speed_err': speed_err,
        'yaw_pen': yaw_pen,
        'effort': effort,
        'contact_bonus': contact_bonus,
    }
