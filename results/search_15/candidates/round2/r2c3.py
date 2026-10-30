import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Components:
    - forward_progress: reward for moving in direction of commanded speed (sign-sensitive).
    - lateral_stability: penalty for yaw rate when no turn commanded.
    - pose_potential: negative quadratic potential based on pitch and pitch_rate.
    - motor_effort: L1 norm of torques (encourages sparsity).
    - exploration_bonus: small reward for non-zero wheel speed to avoid sticking.
    """
    # forward progress: only reward if moving forward when command>0, else penalize reverse motion
    cmd = state.get('commanded_speed', 0.0)
    vel = state.get('ground_velocity', 0.0)
    if cmd > 0.0:
        forward_progress = min(vel, cmd) / (cmd + 1e-6)  # fraction of desired speed achieved, capped at 1
        forward_progress = forward_progress * 0.1  # scale
    else:
        # when no forward command, discourage moving backward
        forward_progress = -0.05 * max(-vel, 0.0)  # penalize reverse motion

    # lateral stability: penalize yaw rate when commanded yaw rate is near zero
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_err = yaw_rate - cmd_yaw
    lateral_stability = -0.02 * yaw_err * yaw_err if abs(cmd_yaw) < 0.05 else 0.0

    # pose potential: Lyapunov-like term for upright stability
    pitch = state.get('pitch', 0.0)
    pitch_rate = state.get('pitch_rate', 0.0)
    pose_potential = -0.4 * (pitch * pitch + 0.05 * pitch_rate * pitch_rate)

    # motor effort: L1 norm of torques (less aggressive than L2)
    effort = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            effort += abs(t)
    motor_effort = -0.0003 * effort

    # exploration bonus: reward any wheel spin to encourage locomotion
    wheel_speed = (abs(state.get('left_wheel_velocity', 0.0)) +
                   abs(state.get('right_wheel_velocity', 0.0))) * 0.5
    exploration_bonus = 0.001 * wheel_speed

    return {
        'forward_progress': forward_progress,
        'lateral_stability': lateral_stability,
        'pose_potential': pose_potential,
        'motor_effort': motor_effort,
        'exploration_bonus': exploration_bonus,
    }
