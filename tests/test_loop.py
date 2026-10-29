# SPDX-License-Identifier: Apache-2.0
"""Checks for Stage A. Standard library only.

    python3 tests/test_loop.py

Prints one line per check and exits non-zero on the first failure.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koda import gate1, orchestrator, report, sanitize  # noqa: E402
from koda import eval_set, prompt, state  # noqa: E402

PASSED = 0


def check(label, condition, detail=""):
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        sys.exit(1)


def gate1_paths():
    print("Gate 1 catches every way a candidate can be broken")
    cases = (
        ("passes", "def compute_reward(s, a):\n return {'x': 1.0 - abs(s['pitch'])}\n", "passed"),
        ("does not parse", "def compute_reward(s, a)\n return {}\n", "dead_compile"),
        ("returns infinity", "def compute_reward(s, a):\n return {'x': float('inf')}\n", "dead_nan"),
        ("returns NaN", "def compute_reward(s, a):\n return {'x': float('nan')}\n", "dead_nan"),
        ("returns a list", "def compute_reward(s, a):\n return [1.0]\n", "dead_gate1"),
        ("returns an empty dict", "def compute_reward(s, a):\n return {}\n", "dead_gate1"),
        ("wrong parameter count", "def compute_reward(s):\n return {'x': 1.0}\n", "dead_gate1"),
        ("no compute_reward", "def reward(s, a):\n return {'x': 1.0}\n", "dead_gate1"),
        ("reads an unknown field", "def compute_reward(s, a):\n return {'x': s['foot']}\n", "dead_gate1"),
        ("imports os", "import os\ndef compute_reward(s, a):\n return {'x': 1.0}\n", "dead_gate1"),
        ("opens a file", "def compute_reward(s, a):\n open('/etc/passwd')\n return {'x': 1.0}\n", "dead_gate1"),
        ("term names vary", "def compute_reward(s, a):\n d={'x':1.0}\n"
                            " if s['pitch']>0.5: d['y']=1.0\n return d\n", "dead_gate1"),
    )
    for label, source, expected in cases:
        verdict = gate1.check(source, label)
        check(f"{label} -> {expected}", verdict["status"] == expected,
              f"got {verdict['status']}: {verdict['error']}")


def fixture_verdicts():
    print("The fixtures break in the ways their headers claim")
    from koda import candidates

    expected = {
        "fixtures/round1/c2_syntax_error.py": "dead_compile",
        "fixtures/round1/c3_infinite_tracking.py": "dead_nan",
        "fixtures/round1/c4_wrong_shape.py": "dead_gate1",
        "fixtures/round2/c2_wrong_signature.py": "dead_gate1",
    }
    seen = {}
    for round_number in (1, 2, 3):
        for candidate in candidates.from_fixtures(round_number, 6):
            seen[candidate["source_path"]] = gate1.check(
                candidate["source"], candidate["candidate_id"]
            )["status"]

    check("18 fixtures found", len(seen) == 18, f"found {len(seen)}")
    for path, status in expected.items():
        check(f"{Path(path).name} -> {status}", seen.get(path) == status,
              f"got {seen.get(path)}")
    broken = [p for p, s in seen.items() if s != "passed"]
    check("exactly four fixtures are broken", len(broken) == 4, str(broken))


def prompt_is_clean():
    print("The prompt never leaves with the robot's name in it")
    text = prompt.build("balance upright and drive forward at the commanded speed", 6)
    check("raw prompt does name the robot", sanitize.leaks(text) != [])
    clean = sanitize.sanitize_or_raise(text)
    check("sanitized prompt names nothing", sanitize.leaks(clean) == [])
    check("the reward hole survives sanitizing", "reward = 0.0" in clean)
    check("every state field is listed",
          all(key in clean for key in state.state_keys()))
    try:
        # Nothing in BANNED should survive, so force the failure with a word
        # that is checked for but deliberately not substituted.
        sanitize.BANNED = sanitize.BANNED + ("balance",)
        sanitize.sanitize_or_raise("the robot must balance")
        check("a surviving banned word raises", False, "no exception")
    except ValueError:
        check("a surviving banned word raises", True)
    finally:
        sanitize.BANNED = tuple(w for w in sanitize.BANNED if w != "balance")

    for odd in ("UPKIE", "Upkie", "upkie", "UpKiE", "QDD100", "MJ5208"):
        check(f"'{odd}' is removed", sanitize.leaks(sanitize.sanitize(odd)) == [])


def grader_sees_physics_only():
    print("The grader is handed physical measurements and nothing else")
    import inspect

    from koda import scorer_stub

    source = inspect.getsource(scorer_stub.score)
    check("score() takes a rollout, not a TrainerResult",
          "rollout" in inspect.signature(scorer_stub.score).parameters)
    for forbidden in ("terms", "reward", "upright_term"):
        check(f"no '{forbidden}' anywhere in score()", forbidden not in source)


def full_run():
    print("A whole 3-round run finishes and writes a report")
    statuses = {}
    classes = set()
    for seed in range(6):
        with tempfile.TemporaryDirectory() as temporary:
            record = orchestrator.run(
                "balance upright and drive forward at the commanded speed",
                seed=seed,
                out_dir=temporary,
                source="fixtures",
            )
            check(f"seed {seed}: three rounds ran", len(record["rounds"]) == 3)
            check(f"seed {seed}: run.json written",
                  (Path(temporary) / "run.json").exists())
            check(f"seed {seed}: three prompts saved",
                  len(list((Path(temporary) / "prompts").glob("*.txt"))) == 3)
            for round_record in record["rounds"]:
                check(f"seed {seed} round {round_record['round']}: a winner",
                      round_record["winner"] is not None)
                for candidate in round_record["candidates"]:
                    statuses[candidate["trainer"]["status"]] = 1
                    if candidate["score"]:
                        check(
                            f"seed {seed}: {candidate['candidate_id']} graded "
                            f"{eval_set.EPISODE_COUNT} episodes",
                            len(candidate["trainer"]["rollout"]["episodes"])
                            == eval_set.EPISODE_COUNT,
                        )
            classes.update(p["class"] for p in record["pathologies"])
            html = report.build(record)
            check(f"seed {seed}: report built", html.startswith("<!doctype html>"))
            check(f"seed {seed}: report says it is not a scored run",
                  "This is not a scored run" in html)

    for status in ("trained", "dead_compile", "dead_gate1", "dead_nan", "dead_gate2"):
        check(f"status '{status}' happened at least once", status in statuses)
    for pathology_class in (1, 2, 3):
        check(f"pathology class {pathology_class} logged at least once",
              pathology_class in classes)


def rounds_improve():
    print("Later rounds score better than round 1")
    with tempfile.TemporaryDirectory() as temporary:
        record = orchestrator.run(
            "balance upright and drive forward at the commanded speed",
            seed=1,
            out_dir=temporary,
            source="fixtures",
        )
        best = []
        for round_record in record["rounds"]:
            scores = [
                c["score"]["score"]
                for c in round_record["candidates"]
                if c["score"]
            ]
            best.append(max(scores))
        check(f"round 3 beats round 1 ({best[0]:.3f} -> {best[2]:.3f})",
              best[2] > best[0])


if __name__ == "__main__":
    gate1_paths()
    fixture_verdicts()
    prompt_is_clean()
    grader_sees_physics_only()
    full_run()
    rounds_improve()
    print(f"\n{PASSED} checks passed")
