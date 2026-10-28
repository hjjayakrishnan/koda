import math

def compute_reward(state: dict, action: dict) -> dict:
    # Exponential balance reward: higher when pitch near zero
    bal = math.exp(-12.0 * abs(state['pitch']))  # range (0,1]

    # Speed tracking with exponential decay of error
    spd_err = state['ground_velocity'] - action['ground_velocity']
    spd = math.exp(-3.0 * spd_err * spd_err) - 1.0  # zero at perfect match, negative otherwise

    # Contact bonus
    cont = 1.5 * state['contact']

    return {
        'balance': bal,
        'speed': spd,
        'contact': cont
    }
