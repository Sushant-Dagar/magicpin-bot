# Vera bot — fix history

## v2.0 fixes (against the actual challenge ZIP — challenge-brief.md, challenge-testing-brief.md,
## examples/case-studies.md, examples/api-call-examples.md, judge_simulator.py)

1. **Removed a fictitious hard body-length cap.** Both `composer.py`'s system prompt
   ("HARD LIMIT 300 chars, fails schema validation over 320") and `main.py`'s `/v1/tick`
   handler (`body_text[:320]`) were silently truncating every message. The actual spec
   (testing-brief, failure-mode F.3) says explicitly: *"No hard body-length cap. Messages
   are judged on quality, specificity, and relevance."* The top-scoring case studies run
   250-450 characters. Replaced with a 900-char sanity ceiling that only guards against a
   genuinely runaway LLM response — never trims a normal message.
2. **Added a URL guard.** Failure-mode F.4: any URL in `body` is an automatic hard fail,
   -3 penalty ("Meta would reject it"). Nothing previously prevented this. Added
   `strip_urls()` in `composer.py`, applied to every body-producing path in both
   `composer.py` and `conversation_handlers.py`, plus a defensive re-check in `main.py`'s
   `/v1/tick`.
3. **Rewrote the system prompt** around the literal "cross-case patterns" checklist from
   `examples/case-studies.md` §"Cross-case patterns the judge looks for": mandatory source
   citation on research/compliance triggers (uncited = capped at 7), numbers must have
   visible provenance (no unexplained figures), owner/customer first name mandatory
   (generic "Hi" loses a merchant-fit point), exactly one CTA landing in the last sentence,
   category-correct vocabulary use, and rewarding an actual judgment call (e.g. "skip the
   promo, the data says it'll underperform") over pure template-filling.
4. **Merchant-level hostile suppression.** The phase-4 replay spec's own reference
   rationale for a hostile exit says "suppressing all triggers for this merchant for 30
   days" — the bot only closed that one conversation. Added a persistent
   `hostile_merchants` set; `/v1/tick` now skips any trigger for a merchant who has gone
   hostile in this test run.
5. **LLM call timeouts.** Neither the OpenAI, Anthropic, nor Groq client calls had an
   explicit timeout, risking blowing the judge's 30s-per-call budget on a slow provider.
   Added an 18s timeout (env `LLM_TIMEOUT_SECONDS`) so a stuck call fails fast into the
   deterministic fallback with time to spare.
6. **Empty-body guard after URL-stripping.** If stripping a URL left an empty body (edge
   case), `compose()`/`respond()` now fall back rather than ship an empty body (which is
   itself a -2 malformed-response penalty).

---

# Arpit's bot — v1.2 fixes (against magicpin judge feedback)

1. **320-char hard limit** — `smart_trim()` caps every body (tick + reply) at 315
   chars, preserving the first sentence and the CTA. LLM prompt updated to warn.
2. **/v1/reply split by from_role** — customer conversations get deterministic
   handling BEFORE the LLM: slot pick honors the customer's own stated day/time
   ("Yes please book me for Wed 5 Nov, 6pm" → books exactly that), numbered
   1/2 picks, booking confirmation as the merchant, opt-out ends immediately.
3. **Grounded merchant follow-ups** — "need help / we have an old X" style
   messages route to the question branch; the LLM prompt now instructs it to
   reference the merchant's exact detail, and the no-LLM fallback mirrors it
   deterministically ("Since you're on old D-speed film unit…").
4. **LLM-failure resilience** — compose() failures no longer skip triggers;
   the upgraded rule-based fallback harvests payload facts + merchant stats,
   so Trigger Coverage stays 25/25 even fully rate-limited.
5. **Schema** — `customer_id` omitted when null (was sent as null on all
   merchant-facing actions).
6. **Context contract** — same-version re-push is now an idempotent accept;
   stale version returns HTTP 409; invalid scope returns HTTP 400.
7. **State persistence** — contexts/conversations/suppression keys saved to
   `bot_state.json` after every mutating call and restored on startup, so a
   Render free-tier restart mid-evaluation no longer wipes state.

## SECURITY — do this immediately
The git repo inside the original zip contains `.env` with a real API key
committed to history. If that repo is public: rotate the key NOW and remove
.env from git (`git rm --cached .env`, add to .gitignore, force-push or
recreate the repo). Deploy keys belong in Render's Environment tab, not git.

## Test results (local, LLM simulated DOWN — worst case)
11/11: versioning (409/400/idempotent), 25/25 coverage, no null customer_id,
max body 314, customer slot pick, grounded follow-up, auto-reply end,
hostile end.
