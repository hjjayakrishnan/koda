# fixtures/

Hand-written Python files that stand in for the model's output, so the whole loop
can be built, run and debugged with no API call, no money and no simulator.

**Nothing in here was written by Nemotron.** Every file says so in its own header.
Once a real Nemotron call is made, its response is saved alongside these and
replayed the same way.

```
round1/  round2/  round3/   six candidates each, the same six every run
env_excerpt.txt            the environment source the prompt quotes, saved so
                           Stage A does not need upkie installed
```

Four files are broken on purpose. They are the only way Gate 1's error paths and
the "send the traceback back to the model" path get tested before they matter:

| File | Fault | Gate 1 verdict |
|---|---|---|
| `round1/c2_syntax_error.py` | missing colon, the file does not parse | `dead_compile` |
| `round1/c3_infinite_tracking.py` | `1/error` is infinite when the error is zero | `dead_nan` |
| `round1/c4_wrong_shape.py` | returns a list instead of a dict of named terms | `dead_gate1` |
| `round2/c2_wrong_signature.py` | three parameters instead of two | `dead_gate1` |

Two more pass Gate 1 and then behave badly, which is the other path worth having:
`round1/c5_stand_still.py` pays the robot to do nothing, and
`round1/c6_distance_only.py` leaves balance out of the objective entirely.

**These are not the R0 cheating rewards.** Those three are separate, live in
`cheaters/`, are written by the other pane, and exist to red-team the grader.
