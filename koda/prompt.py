# SPDX-License-Identifier: Apache-2.0
"""Build the text that is sent to the model.

The prompt has four parts:

  1. the hole: the environment's own `reward = 0.0` line, quoted from its source
  2. the goal: one sentence from the command line
  3. the contract: the function signature and the state fields it may read
  4. from round 2 on: the best reward so far, what the grader measured about it,
     and the exact error message of every candidate that died

Part 4 is the whole point of the loop. Rounds 2 and 3 are shaped by numbers our
own grader produced, which exist in no training data anywhere.

The assembled text goes through `koda.sanitize` before it leaves this machine.
"""

import hashlib
from pathlib import Path

from koda import state as state_mod

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def env_excerpt() -> str:
    """The environment source showing the reward hole.

    Stage B reads it from the installed package. Stage A reads the saved copy in
    `fixtures/env_excerpt.txt`. Leading comment lines are our own attribution
    header and are not part of the prompt.
    """
    try:
        import upkie.envs.upkie_env as upkie_env  # noqa: PLC0415

        source = Path(upkie_env.__file__).read_text().splitlines()
        return "\n".join(source[218:242])
    except Exception:
        text = (FIXTURES / "env_excerpt.txt").read_text().splitlines()
        body = [line for line in text if not line.startswith("#")]
        return "\n".join(body).strip("\n")


def field_block() -> str:
    """The state and action fields, one per line, with units."""
    lines = ["state -- a dict of floats:"]
    for key, unit, meaning in state_mod.field_table():
        lines.append(f"    {key:<24} {unit:<8} {meaning}")
    lines.append("")
    lines.append("action -- a dict of floats:")
    for key, unit, meaning in state_mod.ACTION_FIELDS:
        lines.append(f"    {key:<24} {unit:<8} {meaning}")
    return "\n".join(lines)


CONTRACT = '''Write a Python file that defines exactly this function:

    def compute_reward(state: dict, action: dict) -> dict:
        ...

It is called once per control step. It returns a dict mapping your own term names
to plain floats. The caller adds the values together to get the step's reward, so
a term you want to discourage must return a negative number.

Rules:
  * return a dict of str -> float, and nothing else
  * every value must be a finite number, never NaN and never infinity
  * use only the Python standard library; `math` is available, nothing is imported
    for you
  * do not read files, open sockets, or run other programs
  * pick as many or as few terms as you think the job needs, and name them for
    what they measure
'''


def build(
    goal: str,
    candidates: int,
    best_reward_source: str | None = None,
    reflection: str | None = None,
    errors: list | None = None,
) -> str:
    """Assemble the prompt. Not yet sanitized -- the caller does that."""
    parts = [
        "You are writing the reward function for a robot. Its simulation "
        "environment ships without one: the maintainers wrote reward functions, "
        "removed them, and left the decision to whoever writes the agent.",
        "",
        "This is the environment's step function, verbatim, from "
        "`upkie/envs/upkie_env.py`, class `UpkieEnv`:",
        "",
        "```python",
        env_excerpt(),
        "```",
        "",
        f"The robot has two legs, each ending in a driven wheel, and six actuated "
        f"joints: {', '.join(state_mod.JOINTS)}. The wheels keep it balanced.",
        "",
        f"The goal, in one sentence: {goal}",
        "",
        field_block(),
        "",
        CONTRACT,
    ]

    if best_reward_source:
        parts += [
            "",
            "The best reward function from the previous round was this:",
            "",
            "```python",
            best_reward_source.strip(),
            "```",
        ]
    if reflection:
        parts += ["", "What happened when it was trained and measured:", "", reflection.strip()]
    if errors:
        parts += ["", "These candidates from the previous round failed. The error is exact:"]
        for candidate_id, status, error in errors:
            parts += ["", f"{candidate_id} ({status}):", "```", error.strip(), "```"]

    parts += [
        "",
        f"Write {candidates} different reward functions. Separate them with a line "
        "containing only `### candidate`. Each one must be a complete file defining "
        "compute_reward. Try genuinely different ideas, not the same function with "
        "different numbers.",
    ]
    return "\n".join(parts)


def sha(text: str) -> str:
    """Short sha of the prompt, recorded on every round."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
