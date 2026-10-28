import math

def compute_reward(state: dict, action: dict) -> dict:
    # Pitch penalty
    pitch_pen = -6.0 * state['pitch'] * state['pitch']

    # Contact encouragement: reward when wheels touch ground
    contact_rew = 2.0 * state['contact']

    # Speed tracking using commanded_speed from state (fallback to action if missing)
    cmd_speed = state.get('commanded_speed', action.get('ground_velocity', 0.0))
    speed_err = state['ground_velocity'] - cmd_speed
    speed_pen = -3.0 * speed_err * speed_err

    return {
        'pitch': pitch_pen,
        'contact': contact_rew,
        'speed': speed_pen
    }
