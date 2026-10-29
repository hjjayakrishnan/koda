# Results

Committed evidence, not working files. `out/` is gitignored because it holds
policies and scratch; this directory is the record a reader can check.

## `search_11/` — the first full search, 2026-10-28

X=1, seed 11. 6 candidates x 3 rounds = 18 training runs. 3 Nemotron calls on
Nebius Token Factory. 73.5 minutes wall clock, 6.57 compute-hours across 6
parallel workers. 5,078 prompt + 8,846 completion tokens.

    run.json                       every candidate, traceback, score, winner,
                                   reflection, pathologies, bill
    prompts/roundN.txt             the exact prompt sent, after sanitising
    prompts/roundN_reply.txt       the raw reply, unedited
    candidates/roundN/*.py         all 18 reward functions as the model wrote
                                   them, including the one that does not compile

Best score by round: 0.2281 -> 0.3209 -> 0.3106.

## `mpc_baseline_*.json` — the comparison

`upkie.controllers.MPCBalancer`, Upkie's own hand-engineered controller, over
the same 20 evaluation episodes and the same grader. Score 0.5348. Never fell;
saturated its wheel motors, which counts as a violation and is reported as
measured rather than exempted.

## Reproducing

    ./.venv/bin/python -m koda.cli run --seeds 11 --source nemotron \
        --trainer pybullet --jobs 6 --steps 500000

Prompts are saved, so a search can also be replayed offline from its own
record with `--source fixtures` pointed at `candidates/`.
