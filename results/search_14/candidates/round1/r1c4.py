import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward based on task-oriented potentials:
    - speed_track_linear: linear reward for moving at commanded speed (negative if opposite).
    - yaw_rate_match: reward proportional to product of commanded and actual yaw rate.
    - upright_cosine: cosine‑based reward for pitch (max 1 at zero, 0 at ±π/2).
    - minimal_motion: penalizes unnecessary joint motion when no command.
    - torque_rate: penalizes rate of change of torque approximated by torque itself.
    - energy_efficiency: reward for low power consumption when moving."""
    # speed track – linear, encourages correct direction
    cmd_sp = state.get('commanded_speed', 0.0)
    act_sp = state.get('ground_velocity', 0.0)
    speed_track_lin = 1.0 * cmd_sp * act_sp  # positive when signs match, zero if no command

    # yaw rate match
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    yaw_rate_match = 0.5 * cmd_yaw * act_yaw  # positive when aligned

    # upright cosine (bounded)
    pitch = state.get('pitch', 0.0)
    upright_cos = math.cos(pitch)  # cos(0)=1, cos(±π/2)=0, negative beyond
    # clip to [0,1] to avoid rewarding upside down
    if upright_cos < 0.0:
        upright_cos = 0.0

    # minimal motion: when no command, penalize any joint movement
    no_cmd = (abs(cmd_sp) < 1e-3) and (abs(cmd_yaw) < 1e-3)
    joint_move = 0.0
    if no_cmd:
        for side in ('left', 'right'):
            for joint in ('hip', 'knee', 'wheel'):
                v = state.get(f'{side}_{joint}_velocity', 0.0)
                joint_move += abs(v)
        minimal_motion = -0.01 * joint_move
    else:
        minimal_motion = 0.0

    # torque rate penalty (use torque as proxy for its derivative)
    torque_rate = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_rate += t * t
    torque_rate = -0.0003 * torque_rate

    # energy efficiency: reward low power when actually moving forward
    power = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            power += abs(t * v)
    # only give benefit when forward speed positive
    eff_bonus = 0.0
    if act_sp > 0.0:
        eff_bonus = -0.0004 * power  # negative penalty, smaller power -> less negative => higher reward
    else:
        eff_bonus = 0.0

    return {
        'speed_track_lin': speed_track_lin,
        'yaw_rate_match': yaw_rate_match,
        'upright_cos': upright_cos,
        'minimal_motion': minimal_motion,
        'torque_rate': torque_rate,
        'eff_bonus': eff_bonus,
    }
