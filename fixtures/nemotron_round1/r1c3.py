import math

def compute_reward(state: dict, action: dict) -> dict:
    # Joint velocity smoothness penalty
    joints = ['left_hip', 'left_knee', 'left_wheel',
              'right_hip', 'right_knee', 'right_wheel']
    vel_pen = 0.0
    for j in joints:
        v = state.get(j + '_velocity', 0.0)
        vel_pen += v * v
    vel_pen = -0.02 * vel_pen

    # Torque effort penalty
    torques = ['left_hip', 'left_knee', 'left_wheel',
               'right_hip', 'right_knee', 'right_wheel']
    torque_pen = 0.0
    for t in torques:
        tau = state.get(t + '_torque', 0.0)
        torque_pen += tau * tau
    torque_pen = -0.001 * torque_pen

    # Small upright bonus
    upright = 1.0 / (1.0 + 10.0 * abs(state['pitch']))

    return {
        'smooth': vel_pen,
        'effort': torque_pen,
        'upright': upright
    }
