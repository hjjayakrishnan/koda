# SPDX-License-Identifier: Apache-2.0
"""Gate 1: check a machine-written reward function before spending any compute.

In: the text of one candidate file.
Out: a verdict dict -- did it pass, and if not, exactly why.

Four things kill a candidate here, and each one maps to a `status` in the run
record:

  dead_compile   the file does not parse
  dead_gate1     wrong signature, a banned construct, the wrong return type, or
                 it raised when called
  dead_nan       a term came back NaN or infinity

No simulator, no GPU, a few milliseconds per candidate. The error text is kept
verbatim because it goes straight back to the model in the next round's prompt.
"""

import ast
import math
import traceback

from koda import state as state_mod

ALLOWED_IMPORTS = {"math"}

BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "breakpoint",
}

def _safe_import(name, *args, **kwargs):
    """`import math` has to work; nothing else may be imported at runtime."""
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed")
    return __import__(name, *args, **kwargs)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


def _verdict(status, error, term_names=None, reward_fn=None):
    return {
        "ok": status == "passed",
        "status": status,
        "error": error,
        "term_names": term_names or [],
        "reward_fn": reward_fn,
    }


def _check_source(tree):
    """Static checks on the parsed file. Returns an error string, or None."""
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{alias.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to '{node.attr}' is not allowed"
        elif isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            found = node

    if found is None:
        return "the file does not define a function called compute_reward"

    args = found.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    if len(positional) != 2:
        return (
            "compute_reward must take exactly two positional arguments "
            f"(state, action); this one takes {len(positional)}: "
            f"{', '.join(positional) or 'none'}"
        )
    return None


def _test_states():
    """The states every candidate is called with. Ordinary, then awkward."""
    cases = [("standing still", state_mod.zero_state(0.0))]

    trajectory = state_mod.fake_trajectory(400, 0.6)
    cases.append(("mid-trajectory", trajectory[120]))

    fallen = state_mod.fake_trajectory(400, 0.6, falls_at=100)
    cases.append(("fallen over", fallen[300]))

    # Zero commanded speed: anything dividing by the command blows up here.
    cases.append(("commanded speed of zero", state_mod.zero_state(0.0)))

    big = state_mod.zero_state(0.9)
    big["pitch"] = 1.2
    big["pitch_rate"] = 40.0
    big["ground_velocity"] = 3.0
    for joint in state_mod.WHEEL_JOINTS:
        big[f"{joint}_torque"] = 1.7
        big[f"{joint}_velocity"] = 50.0
    for joint in state_mod.LEG_JOINTS:
        big[f"{joint}_torque"] = 16.0
    cases.append(("every number at its limit", big))

    return cases


def check(source: str, candidate_id: str = "candidate") -> dict:
    """Run Gate 1 on one candidate's source text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _verdict("dead_compile", traceback.format_exc(limit=0).strip())

    problem = _check_source(tree)
    if problem:
        return _verdict("dead_gate1", problem)

    namespace = {"__builtins__": SAFE_BUILTINS, "math": math}
    try:
        exec(compile(tree, f"<{candidate_id}>", "exec"), namespace)
    except Exception:
        return _verdict("dead_gate1", "the file raised while being loaded:\n"
                        + traceback.format_exc())

    reward_fn = namespace.get("compute_reward")
    if not callable(reward_fn):
        return _verdict("dead_gate1", "compute_reward is not callable")

    term_names = None
    for label, test_state in _test_states():
        action = state_mod.fake_action(test_state)
        try:
            result = reward_fn(test_state, action)
        except Exception:
            return _verdict(
                "dead_gate1",
                f"compute_reward raised when the robot was {label}:\n"
                + traceback.format_exc(),
            )

        if not isinstance(result, dict):
            return _verdict(
                "dead_gate1",
                f"compute_reward returned {type(result).__name__}, "
                f"not a dict, when the robot was {label}",
            )
        if not result:
            return _verdict("dead_gate1", "compute_reward returned an empty dict")

        for key, value in result.items():
            if not isinstance(key, str):
                return _verdict(
                    "dead_gate1",
                    f"term name {key!r} is a {type(key).__name__}, not a string",
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return _verdict(
                    "dead_gate1",
                    f"term '{key}' returned {type(value).__name__}, not a number, "
                    f"when the robot was {label}",
                )
            if not math.isfinite(value):
                return _verdict(
                    "dead_nan",
                    f"term '{key}' returned {value} when the robot was {label}",
                )

        names = sorted(result)
        if term_names is None:
            term_names = names
        elif names != term_names:
            return _verdict(
                "dead_gate1",
                f"the term names changed between calls: {term_names} then {names}. "
                "compute_reward must always return the same set of terms",
            )

    return _verdict("passed", None, term_names, reward_fn)
