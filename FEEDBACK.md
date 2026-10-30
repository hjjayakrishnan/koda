# Feedback on Nebius Token Factory and NVIDIA Nemotron

Required submission feedback. Everything below is from building Koda over 12
days, with the numbers from our own run records rather than impressions.

**What we used:** `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory,
14 calls, 22,685 prompt and 49,187 completion tokens. No Nebius GPUs — all
training ran on a 16-thread laptop CPU, which is itself a finding we report.

---

## What worked

**The OpenAI-compatible endpoint meant there was no integration step.** Our
entire client is 20 lines and uses the stock `openai` package:

```python
client = OpenAI(api_key=os.environ["NEBIUS_API_KEY"],
                base_url="https://api.tokenfactory.nebius.com/v1")
```

We budgeted a day for provider integration and spent about twenty minutes. For a
hackathon that is the single highest-leverage thing you offer.

**14 calls, zero failures.** No timeouts, no rate limits, no retries needed, no
5xx. Latency was roughly 30–60 s for a 4,000-token completion, consistently. We
never wrote retry logic and never needed it — which, on a deadline, we noticed.

**`usage` is returned and accurate.** We publish an exact token bill because the
response carries `prompt_tokens` and `completion_tokens` per call and they added
up. Several winning entries in past Nebius events published their bill; your API
makes that trivial rather than an estimate.

**Nemotron writes real, non-trivial reward code.** Across 69 generated
candidates it produced working Python 93% of the time, inventing its own term
names and shapes — `metabolic_cost`, `mechanical_power`, `lyap_pen`,
`rate_smooth`, `upright_gauss`. One candidate wrote a `jerk_penalty` with
nothing in the prompt suggesting one, which matters to us because our grader is
explicitly forbidden from measuring jerk so the model could not have been
copying a hint.

---

## What needs improvement

**1. Bracket errors in multi-sample completions, concentrated in the last
samples.** We ask for 6 reward functions in one completion, separated by a
marker. 5 of 69 candidates failed to compile — 7.2%. Every one was a
bracket-matching error, and every one was in position 5 or 6 of 6:

```
r1c5   for joint in ('hip', 'knee', 'wheel':          unclosed paren
r2c6   }                                              unmatched brace
r3c6   sum(state.get(f'left_{j}_torque', '') ] )      ] closing a (
r3c6   # placeholder - fix below                      left in the output
```

The code is otherwise coherent; it is specifically delimiter matching that
degrades deeper into a long completion. The last line is the clearest signal —
the model emitted a self-acknowledged placeholder rather than finishing.

**Concretely useful to us would be:** guidance in the model card on whether
N-samples-per-completion is recommended at all versus N separate calls, and at
what completion length quality starts to fall off. We used `max_tokens=8000` and
hit no hard truncation, so this is not a length cap — it is quality decay we
could not have predicted from the docs.

**2. No pricing visible in the response or the console during the event.** Our
`run.json` records `usd_tokens: 0.0` because we had no per-token rate to apply —
we publish token counts and state plainly that no cost was recorded. If the
response carried a cost field, or the console exposed a per-model rate, every
entrant could publish a real currency figure instead of a token count. You are
one field away from making honest cost reporting the default in this community.

**3. Nemotron does not know it is being measured by a hidden grader, and says
so.** Several candidates wrote comments addressed to a human reviewer
("placeholder — fix below", "tune this"). In an automated loop there is no human
to read them and the file simply fails. A line in the model card about
system-prompt patterns for fully-automated code generation — where no follow-up
turn exists — would have saved us two dead candidates.

---

## Onboarding

**Fast, with one stumble.** Getting a key and making the first call took under
twenty minutes from a cold start, which is genuinely good.

The stumble: finding the correct `base_url`. We first tried a URL pattern from
another provider's docs and got an opaque auth error rather than a 404, which
sent us looking at the key instead of the URL. A wrong-host error that says
"this is not a Token Factory endpoint" would have saved fifteen minutes.

The `$25` hackathon credit with code `NEBIUS-DEVPOST-GLOBAL26` was more than
enough — we used 14 calls across 12 days and never approached the limit. Credit
anxiety is a real thing at hackathons and not having it changed how freely we
experimented.

---

## What we would build next

**The honest answer: we would rent CPU, not GPU, and that surprised us.**

Our bottleneck was never inference. It was that PyBullet is CPU-bound and we
could only train 6 reward candidates at a time on 16 threads. Eureka, the method
we are reproducing, runs 400 candidates per robot; we ran 18. The gap is
entirely compute for *training*, not for *generation*.

So what we would want from Nebius is a **CPU-heavy batch tier for simulation
fan-out** — 64+ cores, no GPU, cheap, for exactly this shape of workload:
hundreds of short independent physics rollouts. Everything we read about Nebius
AI Cloud is GPU-framed, and for a PyBullet or MuJoCo workload a GPU is the wrong
thing to rent. We suspect we are not the only robotics entry that discovered
this at 2 a.m.

**Second: a documented pattern for "LLM proposes, long simulation disposes".**
Our loop is 3 model calls and 6 CPU-hours — the inference is a rounding error
against the compute. Serverless Jobs looked right for the fan-out but every
example we found was GPU/inference-shaped. A worked example of a cheap
orchestrator waiting on a long parallel non-GPU job would have been the single
most useful page for this project.

---

## One thing we would tell another entrant

Publish the bill and the negative result. Our headline is that a hand-written
controller beat the model-designed reward by 2.3x, and the token counts above
include 2 calls wasted on our own bug. Both facts are in our README. The Token
Factory `usage` field made the first one a two-line function rather than a
spreadsheet.
