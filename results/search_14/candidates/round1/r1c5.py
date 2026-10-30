import math

def compute_reward(state: dict, action: dict) -> dict:
    """Hybrid reward combining several intuitive terms:
    - vel_reward: exponential decay reward for achieving commanded forward speed.
    - yaw_stable: Gaussian reward for keeping yaw rate near zero when not turning.
    - pitch_pen: quadratic penalty for pitch (instability).
    - sym_vel: reward for equal left/right wheel speeds (straight travel) unless yaw command.
    - joint_vel_diff: penalizes differences in joint velocities between sides (smooth gait).
    - contact_stable: small reward for maintaining contact, heavy loss if lost."""
    # velocity reward: closer to command yields higher reward, asymptote 1
    cmd_v = state.get('commanded_speed', 0.0)
    act_v = state.get('ground_velocity', 0.0)
    vel_err = cmd_v - act_v
    vel_reward = math.exp(-0.5 * (vel_err / 0.3) ** 2)  # sigma=0.3 m/s

    # yaw stable: reward when not trying to turn
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    if abs(cmd_yaw) < 1e-3:
        yaw_stable = math.exp(- (act_yaw ** 2) / (2 * 0.2 ** 2))  # sigma=0.2 rad/s
    else:
        # when turning, reward matching the command
        yaw_err = cmd_yaw - act_yaw
        yaw_stable = math.exp(-0.5 * (yaw_err / 0.5) ** 2)  # sigma=0.5 rad/s

    # pitch penalty (quadratic)
    pitch = state.get('pitch', 0.0)
    pitch_pen = -0.8 * pitch * pitch

    # wheel speed symmetry: reward equality when going straight
    lw = state.get('left_wheel_velocity', 0.0)
    rw = state.get('right_wheel_velocity', 0.0)
    wheel_diff = lw - rw
    if abs(cmd_yaw) < 1e-3:  # going straight desired
        sym_vel = -0.001 * wheel_diff * wheel_diff  # penalize difference
    else:
        # when turning, allow some difference proportional to yaw command
        # ideal diff ≈ cmd_yaw * wheel_base (approx 0.5m); we just reward small extra error
        ideal = cmd_yaw * 0.5
        sym_vel = -0.0005 * (wheel_diff - ideal) ** 2

    # joint velocity difference between sides (smooth gait)
    joint_vel_diff = 0.0
    for joint in ('hip', 'knee', 'wheel':
                  ):  # Note: syntax error intentional? No, fix.
        pass
    # Actually compute properly below
    joint_vel_diff = 0.0
    for joint in ('hip', 'knee', 'wheel'):
        lv = state.get(f'left_{joint}_velocity', 0.0)
        rv = state.get(f'right_{joint}_velocity', 0.0)
        joint_vel_diff += (lv - rv) ** 2
    joint_vel_diff_pen = -0.0002 * joint_vel_diff

    # contact stability: reward when contact=1, strong penalty when 0
    contact = state.get('contact', 0.0)
    contact_stable = 0.05 * contact - 0.1 * (1.0 - contact)

    return {
        'vel_reward': vel_reward,
        'yaw_stable': yaw_stable,
        'pitch_pen': pitch_pen,
        'sym_vel': sym_vel,
        'joint_vel_diff_pen': joint_vel_diff_pen,
        'contact_stable': contact_stable,
    }
