# SPDX-License-Identifier: Apache-2.0
"""Turn the run record into one static HTML page.

In:  the run record dict (the same thing as out/run.json).
Out: out/report.html, one file, no JavaScript, no network, nothing to install.

It is generated at the end rather than served live, because a static page still
renders when a run dies halfway through and a live dashboard does not.
"""

import html
import json
from pathlib import Path

STYLE = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0 auto;
       max-width: 60rem; padding: 2rem 1.25rem 6rem; line-height: 1.5; }
h1 { margin-bottom: 0.25rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #8884; padding-bottom: 0.3rem; }
h3 { margin-top: 1.75rem; }
.sub { opacity: 0.7; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.92rem; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #8883; }
th { font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.wrap { overflow-x: auto; }
pre { background: #8881; padding: 0.75rem; border-radius: 6px; overflow-x: auto;
      font-size: 0.85rem; }
code { font-size: 0.9em; }
.tag { display: inline-block; padding: 0.05rem 0.4rem; border-radius: 4px;
       font-size: 0.8rem; background: #8882; }
.dead { background: #d9534f33; }
.win { background: #5cb85c33; }
.note { border-left: 3px solid #8886; padding: 0.5rem 0.9rem; background: #8881;
        border-radius: 0 6px 6px 0; margin: 1rem 0; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem; }
.kv dt { opacity: 0.7; }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
"""

MEASUREMENT_COLUMNS = (
    ("survival_frac", "stayed up"),
    ("mean_speed_error", "speed error"),
    ("displacement_frac", "ground covered"),
    ("heading_error_deg", "worst drift"),
    ("peak_torque_frac_wheel", "wheel torque"),
    ("peak_torque_frac_leg", "leg torque"),
)


def _e(text):
    return html.escape(str(text))


def _candidate_rows(round_record):
    rows = []
    for candidate in round_record["candidates"]:
        trainer_result = candidate["trainer"]
        score = candidate["score"]
        is_winner = candidate["candidate_id"] == round_record["winner"]
        css = "win" if is_winner else ("dead" if score is None else "")

        cells = [
            f"<td><code>{_e(candidate['candidate_id'])}</code></td>",
            f"<td>{_e(Path(candidate['source_path']).name)}</td>",
            f"<td>{_e(trainer_result['status'])}</td>",
        ]
        if score is None:
            cells.append(
                f'<td class="num">--</td>'
                + '<td class="num">--</td>' * len(MEASUREMENT_COLUMNS)
                + "<td>--</td>"
            )
        else:
            cells.append(f'<td class="num">{score["score"]:.3f}</td>')
            for key, _label in MEASUREMENT_COLUMNS:
                value = score["measurements"][key]
                cells.append(f'<td class="num">{value:.3f}</td>')
            cells.append(f"<td>{_e(', '.join(score['violations']) or 'none')}</td>")

        rows.append(f'<tr class="{css}">' + "".join(cells) + "</tr>")
    return "\n".join(rows)


def _term_table(trainer_result):
    terms = trainer_result["terms"]
    if not terms:
        return ""
    head = "".join(
        f'<th class="num">{int(c * 100)}%</th>' for c in terms["checkpoints"]
    )
    body = []
    for key, values in sorted(terms["values"].items()):
        cells = "".join(f'<td class="num">{v:+.3f}</td>' for v in values)
        body.append(f"<tr><td><code>{_e(key)}</code></td>{cells}</tr>")
    total = "".join(f'<td class="num">{v:+.3f}</td>' for v in terms["total"])
    body.append(f"<tr><td><em>sum</em></td>{total}</tr>")
    return (
        '<div class="wrap"><table><tr><th>reward term</th>'
        + head
        + "</tr>"
        + "\n".join(body)
        + "</table></div>"
    )


def _errors(round_record):
    blocks = []
    for candidate in round_record["candidates"]:
        error = candidate["trainer"]["error"]
        if not error:
            continue
        blocks.append(
            "<details><summary><code>"
            + _e(candidate["candidate_id"])
            + "</code> — "
            + _e(candidate["trainer"]["status"])
            + " (this exact text goes back to the model next round)</summary>"
            + f"<pre>{_e(error.strip())}</pre></details>"
        )
    return "\n".join(blocks)


def build(record: dict) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Koda run — {_e(record['search_id'])}</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>Koda run report</h1>",
        f'<p class="sub">{_e(record["goal"])}</p>',
    ]

    config = record["config"]
    if config["trainer"] == "stub" or config["scorer"] == "stub":
        which = []
        if config["trainer"] == "stub":
            which.append("the trainer is the stub, so every physical number on this "
                         "page is made up and no simulator ran")
        if config["scorer"] == "stub":
            which.append("the grader is the stand-in, not the project's graded "
                         "scorer, so no score here counts")
        parts.append(
            '<div class="note"><strong>This is not a scored run.</strong> '
            + _e("; ".join(which))
            + ". The point of this run is that the plumbing works.</div>"
        )

    parts += [
        "<h2>The run</h2>",
        '<div class="kv">',
        f"<dt>robot</dt><dd>{_e(record['robot'])}</dd>",
        f"<dt>environment</dt><dd>{_e(record['env'])}</dd>",
        f"<dt>search id</dt><dd>{_e(record['search_id'])}</dd>",
        f"<dt>rounds</dt><dd>{config['rounds']}</dd>",
        f"<dt>candidates per round</dt><dd>{config['candidates']}</dd>",
        f"<dt>evaluation episodes</dt><dd>{config['episodes_per_candidate']} per candidate</dd>",
        f"<dt>candidates came from</dt><dd>{_e(config['candidate_source'])}</dd>",
        f"<dt>trainer</dt><dd>{_e(config['trainer'])}</dd>",
        f"<dt>grader</dt><dd>{_e(config['scorer'])}</dd>",
        f"<dt>compute</dt><dd>{record['bill']['compute_seconds']:.0f} s</dd>",
        f"<dt>cost</dt><dd>${record['bill']['usd_compute'] + record['bill']['usd_tokens']:.2f} "
        f"— {_e(record['bill']['note'])}</dd>",
        "</div>",
    ]

    baseline = record.get("human_baseline")
    parts.append("<h2>The entrant's own reward</h2>")
    if baseline:
        score = baseline["score"]
        parts += [
            '<div class="kv">',
            f"<dt>file</dt><dd><code>{_e(baseline['source_path'])}</code></dd>",
            f"<dt>status</dt><dd>{_e(baseline['status'])}</dd>",
            f"<dt>score</dt><dd>{'--' if score is None else format(score, '.3f')}</dd>",
            "</div>",
            f'<div class="note">{_e(baseline["note"])}</div>',
        ]
    else:
        parts.append(
            '<div class="note">No <code>human_baseline.py</code> in this run. It is '
            "the comparator every other number is read against, and it has to be "
            "written and frozen before any scored search.</div>"
        )

    for round_record in record["rounds"]:
        parts.append(f"<h2>Round {round_record['round']}</h2>")
        parts.append(
            '<div class="kv">'
            f"<dt>prompt</dt><dd><code>{_e(round_record['prompt_path'])}</code> "
            f"(sha {_e(round_record['prompt_sha'])})</dd>"
            f"<dt>survived Gate 1 and training</dt>"
            f"<dd>{round_record['survivors']} of "
            f"{len(round_record['candidates'])}</dd>"
            f"<dt>winner</dt><dd><code>{_e(round_record['winner'] or 'none')}</code></dd>"
            "</div>"
        )

        header = (
            "<tr><th>id</th><th>candidate</th><th>status</th>"
            '<th class="num">score</th>'
            + "".join(f'<th class="num">{label}</th>' for _k, label in MEASUREMENT_COLUMNS)
            + "<th>limits exceeded</th></tr>"
        )
        parts.append(
            '<div class="wrap"><table>'
            + header
            + _candidate_rows(round_record)
            + "</table></div>"
        )
        parts.append(
            '<p class="sub">Wheel and leg torque are fractions of the motors\' rated '
            "limits, so anything above 1.000 is over the limit.</p>"
        )

        errors = _errors(round_record)
        if errors:
            parts.append("<h3>What the dead candidates said</h3>")
            parts.append(errors)

        winner_record = next(
            (
                c
                for c in round_record["candidates"]
                if c["candidate_id"] == round_record["winner"]
            ),
            None,
        )
        if winner_record:
            parts.append("<h3>The winner's reward terms, over training</h3>")
            parts.append(_term_table(winner_record["trainer"]))
        if round_record["reflection"]:
            parts.append("<h3>What went back to the model</h3>")
            parts.append(f"<pre>{_e(round_record['reflection'])}</pre>")

    parts.append("<h2>Pathology catalogue</h2>")
    if record["pathologies"]:
        rows = "".join(
            "<tr>"
            f"<td class=\"num\">{p['class']}</td>"
            f"<td class=\"num\">{p['round']}</td>"
            f"<td><code>{_e(p['candidate_id'])}</code></td>"
            f"<td>{_e(p['label'])}</td>"
            f"<td>{_e(p['detail'])}</td>"
            "</tr>"
            for p in record["pathologies"]
        )
        parts.append(
            '<div class="wrap"><table><tr><th class="num">class</th>'
            '<th class="num">round</th><th>candidate</th><th>what happened</th>'
            "<th>detail</th></tr>" + rows + "</table></div>"
        )
    else:
        parts.append("<p>Nothing died and no limit was exceeded in this run.</p>")

    parts.append(
        '<div class="note">Classes 1 and 2 — did not compile, went non-finite — come '
        "straight out of the run record. Class 3 here is a candidate that scored on "
        "the task and still pushed a motor past its rated limit. Class 4, a candidate "
        "that scores well and is visibly wrong, is tagged by hand from the evaluation "
        "video, so it is absent from any run with no video.</div>"
    )

    parts.append("</body></html>")
    return "\n".join(parts)


def write(record: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(record))
    return path


if __name__ == "__main__":
    import sys

    run_record = json.loads(Path(sys.argv[1]).read_text())
    print(write(run_record, sys.argv[2] if len(sys.argv) > 2 else "out/report.html"))
