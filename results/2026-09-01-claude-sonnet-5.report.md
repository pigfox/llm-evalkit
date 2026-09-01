# llm-evalkit results

- Model: `claude-sonnet-5` via `AnthropicProvider`
- Run started: 2026-09-01T16:49:34+00:00
- Cases: 16, each run k=3 times
- A case passes only if all k outputs pass. Nothing is majority-voted.

## Misses (6)

Cases where the observed outcome disagreed with the pre-registered prediction, in either direction.

### `sentiment_ambiguous_hedge` — predicted FAIL, observed pass

- Registered reasoning: PREDICTED FAIL. The review is neutral overall but ends on a concrete complaint. Models weight the last clause heavily and we expect 'negative'.
- Agreement rate: 1.00
- Output 1: pass — label == 'neutral'
- Output 2: pass — label == 'neutral'
- Output 3: pass — label == 'neutral'

### `invoice_total` — predicted pass, observed FAIL

- Registered reasoning: Three-term addition; well inside what current models do reliably.
- Agreement rate: 0.67
- Output 1: FAIL — output did not parse, so field 'total' could not be read
- Output 2: FAIL — output did not parse, so field 'total' could not be read
- Output 3: FAIL — output did not parse, so field 'total' could not be read

### `invoice_currency` — predicted pass, observed FAIL

- Registered reasoning: Currency is stated three times and the schema constrains it to three characters.
- Agreement rate: 1.00
- Output 1: FAIL — output did not parse, so field 'currency' could not be read
- Output 2: FAIL — output did not parse, so field 'currency' could not be read
- Output 3: FAIL — output did not parse, so field 'currency' could not be read

### `invoice_line_items` — predicted pass, observed FAIL

- Registered reasoning: Three services are named explicitly and 'only the billed services' rules out a summary row.
- Agreement rate: 0.67
- Output 1: FAIL — output did not parse, so field 'line_items' could not be read
- Output 2: FAIL — output did not parse, so field 'line_items' could not be read
- Output 3: FAIL — output did not parse, so field 'line_items' could not be read

### `topics_tags` — predicted pass, observed FAIL

- Registered reasoning: The three concepts are named in the abstract and the tag vocabulary is constrained by the prompt's 'up to four'.
- Agreement rate: 0.33
- Output 1: FAIL — tags set mismatch; missing ['caching', 'distributed-systems', 'latency'], unexpected ['cache eviction', 'distributed key-value stores', 'skewed workloads', 'tail latency']
- Output 2: FAIL — tags set mismatch; missing ['caching', 'distributed-systems', 'latency'], unexpected ['cache eviction', 'distributed systems', 'key-value stores', 'tail latency']
- Output 3: FAIL — tags set mismatch; missing ['caching', 'distributed-systems', 'latency'], unexpected ['cache eviction', 'key-value stores', 'load skew', 'tail latency']

### `sentiment_unstable` — predicted pass, observed FAIL

- Registered reasoning: A backhanded compliment that resolves positive on balance. We predict the model agrees, and predict it does so consistently.
- Agreement rate: 1.00
- Output 1: FAIL — label was 'neutral', expected 'positive'
- Output 2: FAIL — label was 'neutral', expected 'positive'
- Output 3: FAIL — label was 'neutral', expected 'positive'

## Summary

| case | predicted | observed | miss | agreement | attempts |
| --- | --- | --- | --- | --- | --- |
| `sentiment_clear_positive` | pass | pass |  | 0.67 | 3 |
| `sentiment_ambiguous_hedge` | FAIL | pass | yes | 1.00 | 3 |
| `sentiment_confidence_range` | pass | pass |  | 1.00 | 3 |
| `person_name` | pass | pass |  | 1.00 | 3 |
| `person_age` | pass | pass |  | 1.00 | 3 |
| `person_null_email` | pass | pass |  | 1.00 | 3 |
| `invoice_total` | pass | FAIL | yes | 0.67 | 3 |
| `invoice_currency` | pass | FAIL | yes | 1.00 | 3 |
| `invoice_line_items` | pass | FAIL | yes | 0.67 | 3 |
| `topics_tags` | pass | FAIL | yes | 0.33 | 3 |
| `refusal_is_not_json` | pass | pass |  | 0.33 | 3 |
| `measurement_valid_schema` | pass | pass |  | 1.00 | 3 |
| `measurement_unit_literal` | FAIL | FAIL |  | 0.67 | 3 |
| `weather_tool_call` | pass | pass |  | 1.00 | 3 |
| `sentiment_retry_then_ok` | pass | pass |  | 0.67 | 3 |
| `sentiment_unstable` | pass | FAIL | yes | 1.00 | 3 |

## Pass rate

- Predicted: 87.5%
- Observed: 62.5%
- Delta: -25.0 percentage points
- Misses: 6 of 16

## Cases with agreement below 1.0 (7)

- `sentiment_clear_positive` — agreement 0.67; the model did not agree with itself across k=3 runs.
- `invoice_total` — agreement 0.67; the model did not agree with itself across k=3 runs.
- `invoice_line_items` — agreement 0.67; the model did not agree with itself across k=3 runs.
- `topics_tags` — agreement 0.33; the model did not agree with itself across k=3 runs.
- `refusal_is_not_json` — agreement 0.33; the model did not agree with itself across k=3 runs.
- `measurement_unit_literal` — agreement 0.67; the model did not agree with itself across k=3 runs.
- `sentiment_retry_then_ok` — agreement 0.67; the model did not agree with itself across k=3 runs.
