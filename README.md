# llm-evalkit

An evaluation harness for structured LLM outputs, built around one constraint:
**you write down what you expect before you run anything, and the harness will
not let you change your mind afterwards.**

Most eval code is written after the first results are in. That is not
dishonesty, it is gravity — a tolerance gets widened because 0.85 "is basically
right", a prompt gets a clarifying sentence because the model "obviously
misunderstood", a case gets dropped because it "wasn't testing what we meant".
Each edit is defensible on its own and the aggregate is a number that measures
the person who tuned it. This harness makes those edits impossible to make
quietly.

---

## 1. Pre-registration

Before a single request is sent, `predict` writes `predictions.json`: for every
case, a boolean `predicted_pass` and a free-text `predicted_notes` saying why.
The suite's `predicted_pass_rate` is derived from those booleans.

`run`, `grade` and `report` all refuse to do anything unless that file exists
and still verifies. In particular **`run` checks the lock before it sends the
first request**, so there is no window in which you can look at outputs and then
decide what you predicted.

Two things follow from this that are easy to miss:

- **Predicting a failure is a real move.** A suite where every prediction is
  "pass" is not a prediction, it is an assumption. This suite predicts two cases
  will fail and says why in the registration.
- **A miss in either direction is a miss.** Predicted-pass/observed-fail and
  predicted-fail/observed-pass both mean the prediction was wrong. The second is
  the one you are otherwise tempted to bank quietly as a win, so the report
  labels it as a miss just as loudly.

`predict` is write-once. There is no `--force` and no `--overwrite`. Re-
registering means deleting `predictions.json` on purpose, which leaves a visible
hole in your git history rather than a silent amendment.

## 2. The lock

`predictions.json` carries a `lock` field: a sha256 over

1. the canonicalised predictions block, and
2. **the source text of `graders.py` and `cases.py`**.

The second half is the part that matters. Locking only the predictions would
catch you editing a boolean and miss you loosening `numeric_within("total",
1284.50, 0.01)` to a tolerance of `50.0`, or rewording a prompt until the model
gets it, or swapping `exact_match` for something laxer. Those are the same act
as editing the prediction — moving the goalposts after kickoff — so they break
the same hash.

Flipping one `predicted_pass` in a copy of the registration and grading against
it produces, verbatim:

```
$ python -m evalkit --predictions tampered.json grade
error: predictions not locked: hash mismatch. Recorded 47a8685cae60fe9e...,
computed 77bf6f72792b9735.... Either the predictions or the text of graders.py /
cases.py changed after registration. There is no --force; re-register
deliberately or restore the files.
$ echo $?
1
```

CI enforces this on every push: the `test` job runs `run`/`grade`/`report`
against the **committed** `predictions.json` and never calls `predict`. A pull
request that touches a grader, a tolerance or a prompt without re-registering
turns the build red. That is the lock working, not a broken build.

Cases live in Python rather than YAML for exactly this reason — a data file
would need its own parallel hashing rule and would invite everyone to treat
"just a data edit" as not really an edit. It is really an edit.

## 3. Deterministic grading

Every grader is a pure function of the model's output:

| grader | passes when |
| --- | --- |
| `exact_match(field, expected)` | the parsed field equals `expected` |
| `numeric_within(field, expected, tol)` | the parsed field is within `tol` |
| `field_set_equal(field, expected)` | the field holds exactly that set, order ignored |
| `schema_valid(expect_valid=…)` | the output did (or deliberately did not) parse |
| `tool_call_matches(name, args_predicate)` | the model called `name` with satisfying arguments |

**No model grades a model here.** There is no LLM-judge path, and there is no
flag that turns one on. A judge model has its own error rate, its own
non-determinism and its own prompt sensitivity, so grading with one converts a
measurement into a comparison between two unmeasured things — and the judge's
failures correlate with the candidate's, which is the worst possible property
for a measuring instrument.

Every `Grade` carries a populated `reason` **on a pass as well as a failure**. A
report where only failures explain themselves is one you cannot audit for false
passes.

Bad model output is never an exception. `parse_into` returns
`Parsed(value=…) | ParseFailure(raw=…, errors=[…])`, and a `ParseFailure` is
carried all the way to the report as an ordinary graded outcome. One case in the
suite — `refusal_is_not_json` — has a `ParseFailure` as its *correct* result, so
that path stays a first-class outcome rather than an untested hole.

The tool-calling case is graded on the call alone. `tool_call_matches` reads
`response.tool_calls` and never touches prose, so a model that makes the right
call and then narrates something wrong still passes, and one that describes the
right call beautifully without making it still fails.

## 4. Non-determinism

Each case runs `k` times (default 3). All `k` outputs are kept and all `k` are
graded.

- **A case passes only if every one of its `k` outputs passes.** Two out of three
  is not a pass.
- Nothing is majority-voted, no canonical answer is selected, no score is
  averaged. "Right once out of three" and "right three times out of three" are
  different facts about a model, and collapsing them destroys the more
  interesting one.
- `agreement_rate` reports how often the model agreed with itself, and is
  reported *alongside* the pass rate rather than folded into it.

Retries are part of this picture and are never swallowed. Every attempt is
recorded in `RawResponse.attempts` with its outcome, duration and error, and the
run file keeps the whole chain. The Anthropic SDK's own retry loop is switched
off (`max_retries=0`) so those records are complete rather than describing only
the retries that leaked through.

**Known limitation, stated plainly:** `agreement_rate` compares the *whole*
parsed object plus tool calls, so two outputs that agree on the graded field but
differ in an ungraded one count as disagreement. In the live run below,
`sentiment_clear_positive` scores 0.67 while passing 3/3 — the label is
`positive` every time and only the confidence moves, 0.97 against 0.98. This is
the strict reading and it is the one implemented; if you want agreement on the
graded outcome alone, that is a different metric and this code does not
compute it.

## 5. Results

### Live: `claude-sonnet-5`, 2026-09-01

Run via the `live` workflow, k=3, graded against the pre-registered
`predictions.json` committed before any of it happened. The run file and the
generated report are committed under
[`results/`](results/2026-09-01-claude-sonnet-5.run.json).

| | |
| --- | --- |
| Model | `claude-sonnet-5` via `AnthropicProvider` |
| Run started | 2026-09-01T16:49:34Z |
| Cases | 16, each run k=3 |
| Predicted pass rate | 87.5% |
| **Observed pass rate** | **62.5%** |
| **Delta** | **−25.0 percentage points** |
| **Misses** | **6 of 16** |
| Cases with agreement < 1.0 | 7 of 16 |

**The predictions were wrong about six cases out of sixteen, and the suite came
in twenty-five points below what was registered.** That is the result. No case,
prompt, tolerance or prediction has been touched since — the lock over
`graders.py` and `cases.py` is the same one written before the run, and it still
verifies.

#### The six misses

**`sentiment_ambiguous_hedge` — predicted FAIL, observed pass, agreement 1.00.**
Registered reasoning: *"The review is neutral overall but ends on a concrete
complaint. Models weight the last clause heavily and we expect 'negative'."* The
model returned `neutral` three times out of three. The prediction was wrong in
the flattering direction, which is exactly the kind a report is tempted to bank
quietly as a win. It is counted as a miss.

**`invoice_total`, `invoice_currency`, `invoice_line_items` — all predicted
pass, all observed FAIL** (agreement 0.67, 1.00, 0.67). Three misses with **one
root cause**: the model returned `line_items` as objects —
`{"description": "Consulting", "amount": 1000.00, "currency": "USD"}` — where
the schema declares `list[str]`. The whole `Invoice` object therefore failed to
validate, and every field read failed with *"output did not parse, so field
'total' could not be read"*.

The arithmetic was right. `1284.50` appears in all three outputs of all three
cases. Two of these three cases do not even grade `line_items`. They failed
anyway, because a schema is all-or-nothing and a field you are not grading can
still sink the parse. Whether that is a finding about the model or about the
case design is a fair question — and it is the kind of question you can only ask
honestly *after* the fact if you wrote the prediction down *before* it.

**`topics_tags` — predicted pass, observed FAIL, agreement 0.33.** The model's
tags were semantically right and lexically different on every single run:

```
[1] cache eviction, distributed key-value stores, skewed workloads, tail latency
[2] cache eviction, distributed systems, key-value stores, tail latency
[3] cache eviction, key-value stores, load skew, tail latency
```

`field_set_equal` wanted exactly `caching, distributed-systems, latency` and
matched not one of them by string. A grader that demands exact tokens from an
open vocabulary is a brittle grader; the registration did not anticipate that,
so it is a miss.

**`sentiment_unstable` — predicted pass, observed FAIL, agreement 1.00.**
Registered reasoning: *"A backhanded compliment that resolves positive on
balance. We predict the model agrees, and predict it does so consistently."*
Half right, and the wrong half. The model *was* perfectly consistent — it said
`neutral` three times out of three. The stability prediction held; the label
prediction did not.

#### What was not a miss

`measurement_unit_literal` was predicted to fail and failed, so it is not a
miss. It is worth a line anyway: it returned `"meters per second"` once and
`"metres per second"` twice. A spelling flip inside a single case, across three
identical requests.

#### Agreement, and why there is still no voting

Seven of sixteen cases disagreed with themselves across three identical
requests. What varied was mostly *below* the graded surface — `confidence` 0.97
against 0.98, tag wording, three genuinely different prose paragraphs for the
refusal case. `agreement_rate` compares the whole parsed object, so it sees all
of that; the graders do not.

**Being precise about what this run does and does not show:** every one of the
sixteen cases was **unanimous on the graded outcome** — 3/3 or 0/3, no splits.
So majority voting would have produced the same 62.5% on this run. It would not
have flattered the score.

That is not an argument for voting. Voting would have discarded all seven
agreement figures, and a case that genuinely splits 2/1 is indistinguishable
from a unanimous one once you have voted it. The fixture run below contains
exactly such a case, and under voting it reads as a clean pass. The runner does
not vote because the split is information, not noise to be resolved.

Every case answered on the first attempt: 48 attempts across 48 outputs, no
retries, no timeouts.

### Fixture self-test: `StubProvider`

These are **not** a measurement of any model. They come from `StubProvider`
replaying hand-authored fixtures in `fixtures/stub/`, and they are kept because
they exercise paths the live run does not — a scripted 529 retry, and a case
that genuinely splits across k. This is what CI runs on every push.

| | |
| --- | --- |
| Predicted pass rate | 87.5% |
| Observed pass rate | 75.0% |
| Delta | −12.5 percentage points |
| Misses | 2 of 16 |
| Cases with agreement < 1.0 | 1 |

`invoice_line_items` misses because the fixture adds a `Total` summary row.
`sentiment_unstable` misses at agreement 0.33 with outputs
`[positive, negative, positive]` — **2 of 3 passing.** Under majority voting
that case records as a clean pass and its instability vanishes from the report
entirely. It is the case the live run did not happen to produce, and the reason
the no-voting rule is written the way it is.

`sentiment_retry_then_ok` shows 6 attempts across 3 outputs: its fixture scripts
a 529 on the first attempt of every call, so the retry path is exercised on
every default run and the attempt records are never empty.

Regenerate either with `python -m evalkit report`; the markdown lands in
`runs/report.md` and the machine-readable form in `runs/results.json`. `runs/`
is gitignored — the one published live run under `results/` is a deliberate
exception.

## 6. Install and use

Requires Python ≥ 3.11. Dependencies are pinned in `requirements-lock.txt`,
which is what CI installs and what the results above were produced under.

The normal route is to install the package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Then the CLI is available as a module:

```bash
python -m evalkit predict   # pre-register. write-once.
python -m evalkit run       # k runs per case; verifies the lock first
python -m evalkit grade     # writes runs/results.json
python -m evalkit report    # writes runs/report.md, prints it
```

### Running without installing

The src layout means `python -m evalkit` cannot find the package from a bare
checkout, so the no-install route needs `src` on the path explicitly:

```bash
.venv/bin/python -m pip install -r requirements-lock.txt   # dependencies only
export PYTHONPATH=src
python -m evalkit predict                                  # etc., as above
```

`pytest` needs no such prefix — `pythonpath = ["src"]` in `pyproject.toml`
handles it.

### What has and has not been executed

The build backend is `hatchling`, and packaging is exercised rather than merely
declared. The `build` job in CI runs `hatchling build`, asserts that the wheel
carries every module found under `src/evalkit/` at `evalkit/` plus the LICENSE
— the expected set is derived from the source tree, so a module that fails to
ship turns the job red — then installs the wheel and runs the CLI from a
directory containing no source tree, so the import can only be satisfied by the
installed package.

Two things are worth stating precisely, because they are the difference between
"tested" and "configured":

- **`pip install .` has never been executed anywhere in this repository's
  history.** What CI executes is `pip install dist/*.whl` after building. The
  two differ only in whether pip drives the build step, but they are not the
  same command and only the second one has been run.
- **The installed route is exercised in CI, not locally.** The development
  environment here runs from source; the wheel is built locally but never
  installed locally.

Against the real API:

```bash
export ANTHROPIC_API_KEY=...        # read from the environment only, never a file
python -m evalkit run --provider anthropic --model claude-sonnet-5 -k 3
```

`AnthropicProvider` raises at construction if `ANTHROPIC_API_KEY` is absent, with
a message naming the variable. It never opens a file looking for a key.

### Exit codes

| code | meaning |
| --- | --- |
| `0` | clean |
| `1` | lock or usage error — missing/invalid registration, no run to grade |
| `2` | provider failure — the harness worked, the model or its transport did not |

Misses do **not** make the exit code non-zero. Failing a build on a miss would
create pressure to predict conservatively, which is the bias pre-registration
exists to remove.

### Development

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/pytest --cov=evalkit --cov-fail-under=100
```

The test suite runs against `StubProvider` only and needs no API key. Coverage
is gated at 100% of `src/evalkit` on **line** coverage (branch coverage is not
measured), with three standard exclusions declared in `pyproject.toml`:
`pragma: no cover`, `if TYPE_CHECKING:`, bare `...` Protocol bodies, and the
`if __name__ == "__main__":` guard.

### A note on the pinned SDK

This is written against `anthropic==1.2.0` and two details of that version are
load-bearing:

- `messages.create` no longer accepts `temperature`, so this harness does not
  pass one and makes no determinism claim on that basis. Sampling is whatever
  the API defaults to.
- The client retries twice by default. That is disabled here so that the
  recorded attempt chain is the complete one.

Content blocks are read by their `type` discriminator rather than by position,
so unknown future block types are ignored rather than misread as text.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Pigfox LLC.
