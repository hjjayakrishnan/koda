# SPDX-License-Identifier: Apache-2.0
"""Where candidate reward functions come from.

Stage A: `from_fixtures` reads hand-written files out of `fixtures/roundN/`.
Same six candidates every run, so a bug in anything downstream is reproducible.

Stage B: a `from_nemotron` function goes here, posting the prompt to Nebius Token
Factory and splitting the reply on the `### candidate` separator. It returns the
same list of dicts, so nothing else changes.
"""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


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


def split_reply(text: str) -> list:
    """Split a model reply on the `### candidate` separator. Used in Stage B."""
    chunks = [chunk.strip() for chunk in text.split("### candidate")]
    return [chunk for chunk in chunks if chunk]
