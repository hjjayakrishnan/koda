# Defects found in our own measurement code

Nine defects were found in Koda's own grader, trainer and test suite while
investigating why a reward search was not improving. Every one was found by us,
every one is listed here with the commit that closed it, and two of them we
closed too early and had to reopen.

This list exists because a reward-search result is only as good as the code that
measured it, and ours was wrong in nine ways. Two of them — a dead random seed
and a force leaking between episodes — silently corrupted every number the
project produced before 2026-10-29.

Updated 2026-10-29 after the sweep. Eight of nine closed and pushed.

| # | Where | What was wrong | Closed by |
|---|---|---|---|
| 1 | `reward_env.py` reset | `reset(seed=...)` ignored the seed. All five seeds started identically, so the 20-episode set was 3 episodes repeated 5 times and no result carried variance. | `7b96671` — `INIT_PITCH_RAD = 0.02` turns on Upkie's own seeded randomisation. `tests/test_seeding.py`, 8 checks, fails loudly if it regresses |
| 2 | `results/search_11/run.json` | Does not reproduce. No committed policy produces its recorded episodes, under the code that produced them. Cause never found. | `862712e` — carries a `RETRACTION` block as its first key |
| 3 | `eval_set.py` + `trainer_pybullet.py` | A 25 N shove was never cleared if the robot fell inside the 20-step window after it, and `reset()` does not clear forces. It leaked into every later episode in that environment. | `862712e` — `PUSH_NEWTONS = 0.0` by the user's decision, forces cleared after every reset, and one fresh environment per scoring episode |
| 4 | `out/pd_baseline/score.json` | Said 0.7046 from the stub grader, one episode, `survival_frac` 0.4268. Real number 0.4263. | `862712e` — `RETRACTION` block. Note `out/` is gitignored, so the stamp is local only |
| 5 | `check_pd_baseline.py` | Took `--gui`, so it ran headless unless you knew a flag. | `862712e`, and `b0fecdf` did the same for `sweep_mpc_baseline.py`. All four viewers now take `--headless` and open a window by default |
| 6 | `run.json` `bill` | Said "nothing was spent" on a run that made 3 paid Nemotron calls. | **REOPENED then closed.** `862712e` corrected the *output file* and never touched the code. `orchestrator.py` still hardcoded the note before a single call was made — and wrote the same key twice, so one copy silently overwrote the other. The next real search, `search_12`, reproduced the lie word for word on a run with 3 paid calls. Closed properly: the note is now computed at the end from `model_calls`, and `candidate_source` and `model` are recorded beside it |
| 7 | `trainer_pybullet.py:126` | A bare `if True:` — the scar left when the per-episode environment was removed. | `862712e` — gone, body dedented, docstring corrected |
| 8 | `tests/test_loop.py` + `orchestrator.py` | Both `run()` and `search_all()` defaulted `source="nemotron"`, so running the test suite made real paid calls. Introduced by `33ff2fc`; the tests were correct when written in `2822a64`. | `862712e` — both library defaults are `"fixtures"`, both call sites name their choice, `cli.py` keeps `"nemotron"` by the user's decision |

**All nine closed.**

| # | Where | What was wrong | Closed by |
|---|---|---|---|
| 9 | `koda/scorer.py` | **Closed twice — the first fix was half of one.** `never_recovered_from_push` fired whenever a push episode existed and none reported recovery. With `PUSH_NEWTONS = 0.0` there is no shove to recover from, so what tripped `disturbed` was the robot's own wobble crossing `RECOVERY_PITCH_RAD`. The flag fired on steady candidates, which never wobble that far, and cleared on unsteady ones that wobble across and come back — backwards. Introduced 2026-10-29 by the reviewer when zeroing the push. | Gated on the rollout's own `disturbed` field, not on `PUSH_NEWTONS`, because this file grades from physical measurements and must not read the eval-set configuration. `non_recovery_count` now counts only shoves that actually moved the robot, and `disturbed_count` is reported beside it. 8 result files had the false violation removed and not one score changed. **That was only half.** The live `search_12` run then told the model "knocked off upright 5 time(s) and never came back" and handed it a violation with "push" in the name, on a run where `PUSH_NEWTONS` was 0.0 and nothing had pushed anything — what crosses `RECOVERY_PITCH_RAD` with no shove is the robot's own instability. Renamed to `never_recovered_from_disturbance`, which is true whether a shove or the robot's own wobble caused it |

---

## The rule this list produced

> **Fix the code, not the file it produced.**
>
> A defect is not closed until the thing that *writes* the bad output is
> changed. Correcting a JSON file by hand and marking the defect closed means
> the next run reproduces it — which happened twice here. Defect 6 was "fixed"
> by editing `run.json`; the next search wrote the same false billing note on a
> run with 3 paid calls. Defect 9 was "fixed" by gating a flag whose *name* was
> still wrong; the next search sent that name to the model.
>
> Before closing a defect, name the **code** file that changed, and run the
> thing that produces the output to prove it.

## Tests

```
tests/test_scorer.py           102 checks   the grader, red-teamed
tests/test_seeding.py            8 checks   seeds must change the start
tests/test_parent_choice.py     14 checks   the search keeps its best reward
tests/test_trainer_settings.py  29 checks   PPO settings, units, and that a
                                            saved policy reloads identically
```

Run them all:

```bash
for t in tests/*.py; do ./.venv/bin/python "$t"; done
```
