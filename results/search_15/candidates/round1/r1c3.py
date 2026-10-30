import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Idea: shape reward as product of normalized sub‑rewards (all in [0,1]).
    Each term is a multiplicative factor; the final dict stores log‑terms so
    that summation yields log(product) → maximizing product.
    Terms:
    - log_speed:   log of Gaussian similarity of speed.
    - log_yaw:     log of Gaussian similarity of yaw rate.
    - log_upright: log of cosine based uprightness.
    - log_smooth:  log of penalty for joint jerk approximated by velocity.
    - log_contact: log of contact reward.
    - log_effort:  log of torque penalty.
    """
    # helpers to keep within [0,1]
    def guassian_sim(x, sigma):
        return math.exp(-0.5 * (x / sigma) ** 2)

    # speed similarity
    sp_err = state.get('commanded_speed', 0.0) - state.get('ground_velocity', 0.0)
    speed_sim = guassian_sp_err = guassian_sim(sp_err, 0.3)   # sigma=0.3 m/s
    log_speed = math.log(max(speed_sim, 1e-8))

    # yaw similarity
    y_err = state.get('commanded_yaw_rate', 0.0) - state.get('yaw_rate', 0.0)
    yaw_sim = guassian_sim(y_err, 0.5)   # sigma=0.5 rad/s
    log_yaw = math.log(max(yaw_sim, 1e-8))

    # upright cosine (already [0,1])
    pitch = state.get('pitch', 0.0)
    upright = max(math.cos(pitch), 0.0)
    log_upright = math.log(max(upright, 1e-8))

    # smoothness: penalize high joint velocities
    vel_sq_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            vel_sq_sum += v * v
    smooth = math.exp(-0.0001 * vel_sq_sum)   # in (0,1]
    log_smooth = math.log(max(smooth, 1e-8))

    # contact reward
    contact = state.get('contact', 0.0)
    log_contact = math.log(max(contact, 1e-8))

    # effort: low torque
    torque_sq_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sq_sum += t * t
    effort = math.exp(-0.0002 * torque_sq_sum)
    log_effort = math.log(max(effort, 1e-8))

    return {
        'log_speed': log_speed,
        'log_yaw': log_yaw,
        'log_upright': log_upright,
        'log_smooth': log_smooth,
        'log_contact': log_contact,
        'log_effort': log_effort,
    }
