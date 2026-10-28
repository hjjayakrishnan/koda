import math

def compute_reward(state: dict, action: dict) -> dict:
    # Progress toward commanded speed over time using position change approximated by velocity*dt
    progress = state['ground_velocity'] * state['dt'] - action['ground_velocity'] * state['dt']
    prog_rew = -10.0 * progress * progress  # penalize deviation from expected advance

    # Energy cost proxy: sum of absolute torques
    torque_names = ['left_hip', 'left_knee', 'left_wheel',
                    'right_hip', 'right_knee', 'right_wheel']
    energy = 0.0
    for n in torque_names:
        energy += abs(state.get(n + '_torque', 0.0))
    energy_pen = -0.05 * energy

    # Small alive bonus
    alive = 0.2

    return {
        'progress': prog_rew,
        'energy': energy_pen,
        'alive': alive
    }
}
