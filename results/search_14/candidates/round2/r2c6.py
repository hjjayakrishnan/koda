import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward based on Lyapunov-like stability:
    - V = pitch^2 + pitch_rate^2 (penalize).
    - forward progress reward: change in ground position approximated by ground_velocity * dt.
    - directional alignment reward: dot product of velocity vector [ground_velocity, 0]
                                  with desired direction [commanded_speed, 0].
    - control effort: squared action values.
    - contact bonus.
    """
    # Lyapunov penalty
    pitch = state.get('pitch', 0.0)
    pitch_rate = state.get('pitch_rate', 0.0)
    lyap_pen = -0.06 * (pitch * pitch + pitch_rate * pitch_rate)

    # forward progress (using dt as scaling factor)
    dt = state.get('dt', 0.01)  # fallback small dt if missing
    progress = state.get('ground_velocity', 0.0) * dt
    progress_reward = 0.5 * progress  # reward forward motion

    # directional alignment (cosine of speed error)
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    # simple alignment reward: higher when signs match and magnitudes close
    align = math.tanh(cmd_speed) * math.tanh(act_speed)  # range [-1,1]
    align_reward = 0.08 * align

    # control effort penalty (actions)
    gcmd = action.get('ground_velocity', 0.0)
    ycmd = action.get('yaw_velocity', 0.0)
    effort_pen = -0.004 * (gcmd * gcmd + ycmd * ycmd)

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.012 * contact

    return {
        'lyap_pen': lyap_pen,
        'progress_reward': progress_reward,
        'align_reward': align_reward,
        'effort_pen': effort_pen,
        'contact_bonus': contact_bonus,
    }
}
