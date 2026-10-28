# SPDX-License-Identifier: Apache-2.0
"""Where candidate reward functions come from.

Stage A: `from_fixtures` reads hand-written files out of `fixtures/roundN/`.
Same six candidates every run, so a bug in anything downstream is reproducible.

Stage B: a `from_nemotron` function goes here, posting the prompt to Nebius Token
Factory and splitting the reply on the `### candidate` separator. It returns the
same list of dicts, so nothing else changes.
"""

import os
import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

SEPARATOR = "### candidate"


def _load_env(path=".env"):
    """Read .env without a dependency. Values already in the environment win."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def split_reply(text: str, round_number: int, count: int) -> list:
    """Split one model reply into candidates on the `### candidate` separator.

    Tolerant on purpose: the model may number them, fence them in markdown, or
    emit fewer than asked for. Whatever comes back is passed to Gate 1 as-is --
    this function does not repair code, because a candidate that does not
    compile is a real result and its traceback goes back to the model.
    """
    chunks = [c for c in re.split(rf"^{re.escape(SEPARATOR)}.*$", text,
                                 flags=re.MULTILINE) if c.strip()]
    if len(chunks) <= 1:  # separator missing -- fall back to fenced blocks
        fenced = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
        chunks = fenced if fenced else [text]

    out = []
    for index, chunk in enumerate(chunks[:count], start=1):
        source = chunk.strip()
        if source.startswith("```"):
            source = re.sub(r"^```(?:python)?\n|```$", "", source).strip()
        out.append(
            {
                "candidate_id": f"r{round_number}c{index}",
                "source_path": f"<nemotron round {round_number} candidate {index}>",
                "source": source,
            }
        )
    return out


def from_nemotron(prompt_text: str, round_number: int, count: int,
                  save_raw: Path | None = None) -> tuple:
    """Ask Nemotron on Nebius Token Factory for `count` reward functions.

    Returns (candidates, meta). `meta` carries the token counts so the bill can
    be published, and the raw reply is saved when `save_raw` is given -- one
    real reply replayed offline costs nothing and keeps a search reproducible.
    """
    from openai import OpenAI  # noqa: PLC0415

    _load_env()
    client = OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL",
                                "https://api.tokenfactory.nebius.com/v1"),
    )
    model = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=1.0,  # independent samples, per Eureka
        max_tokens=8000,
    )
    text = response.choices[0].message.content or ""

    if save_raw is not None:
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(text)

    usage = getattr(response, "usage", None)
    meta = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "finish_reason": response.choices[0].finish_reason,
    }
    return split_reply(text, round_number, count), meta


def from_fixtures(round_number: int, count: int) -> list:
    """Read one round's candidates off disk, in filename order."""
    folder = FIXTURES / f"round{round_number}"
    if not folder.is_dir():
        raise FileNotFoundError(f"no fixtures for round {round_number}: {folder}")

    files = sorted(p for p in folder.glob("*.py"))
    if len(files) < count:
        raise FileNotFoundError(
            f"round {round_number} needs {count} candidates, {folder} has {len(files)}"
        )

    out = []
    for index, path in enumerate(files[:count], start=1):
        out.append(
            {
                "candidate_id": f"r{round_number}c{index}",
                "source_path": str(path.relative_to(FIXTURES.parent)),
                "source": path.read_text(),
            }
        )
    return out


