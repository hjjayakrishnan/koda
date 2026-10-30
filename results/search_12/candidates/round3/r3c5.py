import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward focused on task progress and safety:
    - progress: forward displacement projected onto the commanded heading.
    - safety_pitch: heavy penalty when pitch exceeds safe threshold.
    - safety_yaw: penalty for large yaw deviation from commanded heading.
    - energy: penalty for total absolute torque.
    - contact_reward: reward for maintaining ground contact.
    - idle_penalty: small penalty when commanded speed is zero but robot moves (avoids drift).
    """
    # progress: dot product of world velocity vector with forward body axis
    yaw = state.get('yaw', 0.0)
    ground_vel = state.get('ground_velocity', 0.0)
    # forward component in world frame
    progress = ground_vel * math.cos(yaw)  # positive when moving forward aligned with body

    # safety pitch: quadratic beyond threshold
    pitch = state.get('pitch', 0.0)
    pitch_thr = 0.25  # ~14 deg
    if abs(pitch) > pitch_thr:
        safety_pitch = -1.0 * (abs(pitch) - pitch_thr) ** 2
    else:
        safety_pitch = 0.0

    # safety yaw deviation from commanded heading (we approximate desired heading as zero)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_thr = 0.5  # rad/s
    safety_yaw = -0.5 * max(0.0, abs(yaw_rate) - yaw_thr) ** 2

    # energy: sum of absolute torques
    torque_abs_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            tau = state.get(f'{side}_{joint}_torque', 0.0)
            torque_abs_sum += abs(tau)
    energy = -0.005 * torque_abs_sum

    # contact reward
    contact = state.get('contact', 0.0)
    contact_reward = 0.02 * contact

    # idle penalty: discourage movement when no forward command
    cmd_speed = state.get('commanded_speed', 0.0)
    idle_penalty = -0.01 * (abs(ground_vel) if abs(cmd_speed) < 0.05 else 0.0)

    return {
        'progress': progress,
        'safety_pitch': safety_pitch,
        'safety_yaw': safety_yaw,
        'energy': energy,
        'contact_reward': contact_reward,
        'idle_penalty': idle_penalty,
    }
