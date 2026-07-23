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
