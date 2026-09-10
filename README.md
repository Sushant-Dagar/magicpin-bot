# Vera Bot — magicpin AI Challenge Submission

## Approach

A 4-context LLM composer (`compose(category, merchant, trigger, customer)` in `composer.py`)
dispatched by `trigger.kind` — each of the 24 known trigger kinds gets its own short
guidance block telling the LLM exactly what "good" looks like for that trigger (lead with
the metric, cite the source, name the affected customer count, etc.), on top of a single
shared system prompt that encodes the challenge's own scoring rubric: mandatory source
citations for research/compliance triggers, numbers must trace back to the context (no
fabrication), the merchant's/customer's real first name, exactly one CTA in the final
sentence, category-correct vocabulary, and — where the data supports it — an actual
judgment call rather than a template fill (e.g. telling a restaurant to skip a Saturday
IPL promo because Saturday matches historically depress covers).

`main.py` implements the 5-endpoint HTTP contract with idempotent `(scope, context_id,
version)` context storage, and persists state to disk so a free-tier host restart mid-test
doesn't wipe context. `conversation_handlers.py` handles multi-turn replies: deterministic
auto-reply detection (tracked at the merchant level so it survives the judge issuing a new
`conversation_id` each turn), explicit intent-transition routing (a merchant saying "let's
do it" skips straight to action mode, never back to qualifying questions), hostile-message
handling that both ends the conversation *and* suppresses all future proactive sends to
that merchant, and deterministic slot-booking for customer-facing conversations (the
customer's own stated day/time is honored exactly, not re-asked).

## Key correctness fixes made against the actual spec

Two bugs in an earlier version actively fought the rubric: (1) a self-imposed ~315-char
hard truncation on every message, when the spec explicitly states there is no length cap
and the highest-scoring case studies run 250–450 characters; and (2) no guard against URLs
in the body, which the spec treats as an automatic hard fail (-3). Both are fixed — bodies
are now only capped by a generous sanity ceiling against a runaway LLM, and every body path
runs through a URL-stripping guard before it ships.

## Tradeoffs

- **LLM-first, rule-based fallback.** If the LLM call fails or times out, `compose()` falls
  back to a deterministic composer that harvests real facts straight from the trigger
  payload and merchant record — lower ceiling on Engagement Compulsion, but it guarantees
  trigger coverage stays at 100% and nothing fabricates data, even under a rate limit.
- **No retrieval/embeddings.** With only ~5 digest items per category, the composer passes
  the trigger's referenced digest item (or the top 3) directly in the prompt rather than
  building a vector index — simpler and just as accurate at this dataset size.
- **Provider-agnostic but single-provider-at-a-time.** `LLM_PROVIDER` env var switches
  between OpenAI/Anthropic/Groq; no automatic multi-provider failover, to keep the 30s
  latency budget predictable (each call has an 18s timeout, leaving headroom to fail into
  the fallback and still return before the judge's 30s cutoff).

## What additional context would have helped most

A larger `patient_content_library` / reusable content bank per category would let the bot
credibly say "I've drafted X" and attach something real, rather than promising to draft it
in a follow-up turn. Real historical A/B data on which compulsion levers actually convert
for each category (rather than inferring from the case studies) would sharpen the
per-trigger-kind guidance further.
