# Upkie's MPC balancer, re-scored on the fixed harness

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python sweep_mpc_baseline.py              # a window
    ./.venv/bin/python sweep_mpc_baseline.py --headless   # numbers only

The hand-engineered baseline, measured under exactly the conditions the trained
policies are measured under: starting tilt +-0.02 rad, push 0.0 N, a fresh
environment per episode, graded by `koda/scorer.py` @ `6bfd77d`.

`score.json` carries a `provenance` block (code sha, dirty flag, tilt, push) as
RULES.md R3 requires.

## Result

```
score                      0.5341        (0.6031 before torque_headroom entered
                                         the score; 0.5348 on the broken harness)
survival_frac              1.0           never falls, in any of the 20 episodes
mean_speed_error           0.2831 m/s
displacement_frac          0.4533
max_pitch_excursion_deg    8.49
peak_torque_frac_wheel     1.0           saturates the wheels
violations                 wheel_torque_exceeded, never_recovered_from_disturbance
```

Distance covered, by commanded speed:

| commanded | travelled in 20 s | of what was asked |
|---|---|---|
| 0.3 m/s | ~1.3 m | ~22% |
| 0.6 m/s | ~5.8 m | ~48% |
| 0.9 m/s | ~11.9 m | ~66% |

It undershoots hardest at the lowest setpoint.

## Against the search

| | score | survival | torque_headroom | speed error |
|---|---|---|---|---|
| MPC, hand-built | **0.5341** | 1.000 | **0.910** | **0.2831** |
| r2c3, best candidate | 0.0943 | 0.420 | 0.866 | 0.5064 |
| r3c2, trained with the ramp | 0.0310 | 0.630 | 0.293 | 0.8053 |
| r3c2, no ramp | 0.0026 | 1.000 | 0.041 | 1.0823 |

**The reward search did not beat the hand-engineered controller.** The baseline
is roughly twice the best candidate.

The useful part is *where* it loses. Both never fall, so r3c2 matches MPC on
balance outright. The entire gap is speed tracking: r3c2's mean speed error is
1.0823 m/s against MPC's 0.2831, about four times worse. The search produced a
balancer, not a tracker.

`never_recovered_from_disturbance` fires here too, for the same reason it fires on every
candidate: with `PUSH_NEWTONS = 0.0` there is no shove to recover from. See
RULES.md defect 9. It is recorded, not priced into the score.
