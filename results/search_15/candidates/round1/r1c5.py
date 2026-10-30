import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Energy‑shaping reward inspired by LQR‑like quadratic forms but with
    hand‑tuned state‑weight matrices.
    State error vector e = [speed_err, yaw_err, pitch, pitch_rate].
    Action vector u = [ground_velocity_cmd, yaw_velocity_cmd].
    Reward = -(e^T Q e + u^T R u) + small contact bonus.
    Q and R are diagonal matrices chosen to reflect task priorities.
    """
    # state errors
    speed_err = state.get('commanded_speed', 0.0) - state.get('ground_velocity', 0.0)
    yaw_err   = state.get('commanded_yaw_rate', 0.0) - state.get('yaw_rate', 0.0)
    pitch     = state.get('pitch', 0.0)
    pitch_rate= state.get('pitch_rate', 0.0)

    # Q diagonal weights (tuned)
    q_speed   = 1.0
    q_yaw     = 0.5
    q_pitch   = 2.0      # penalize tilt heavily
    q_pitchr  = 0.1

    quad_state = (q_speed   * speed_err   * speed_err +
                  q_yaw     * yaw_err     * yaw_err   +
                  q_pitch   * pitch       * pitch     +
                  q_pitchr  * pitch_rate  * pitch_rate)

    # action vector
    u_sp = action.get('ground_velocity', 0.0)
    u_yw = action.get('yaw_velocity', 0.0)

    # R diagonal weights
    r_sp = 0.01
    r_yw = 0.01

    quad_action = r_sp * u_sp * u_sp + r_yw * u_yw * u_yw

    # contact bonus (encourage keeping wheels on ground)
    contact = state.get('contact', 0.0)
    contact_bonus = 0.03 * contact

    # final reward = negative quadratic cost + bonus
    reward_val = -(quad_state + quad_action) + contact_bonus

    # Return as a single term; caller sums terms, so we just give one entry.
    return {'quadratic_cost': reward_val}
</answer>
