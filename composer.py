"""
Vera Message Composer

LLM-powered composer that takes the 4 contexts and produces a high-scoring
WhatsApp message. Dispatches by trigger kind for best results.
"""
from __future__ import annotations
import json
import os
import re
import time
from typing import Optional

import threading
from collections import deque

# --- Token-budget guard -----------------------------------------------------
# Groq's free tier for openai/gpt-oss-120b is 8,000 tokens/minute (confirmed from the
# account's own dashboard) -- a hard ceiling no amount of concurrency/retry tuning can
# get around. Rather than fire a call that's very likely to 429 and waste the attempt,
# track a rolling 60s window of estimated token usage and skip straight to the safe
# deterministic fallback when the budget for this window is already spent.
TPM_BUDGET = int(os.getenv("LLM_TPM_BUDGET", "7000"))  # a little under the real 8000 as margin
_token_usage_window: deque = deque()  # (timestamp, estimated_tokens)
_token_lock = threading.Lock()

def _estimate_tokens(system: str, user: str) -> int:
    # ~4 chars/token is the standard rough estimate; add a buffer for the model's own
    # (often substantial, since gpt-oss-120b is a reasoning model) output + reasoning tokens.
    return (len(system) + len(user)) // 4 + 400

def _reserve_token_budget(estimated: int) -> bool:
    """Returns True and reserves the tokens if there's room in the current 60s window,
    False if this call should be skipped to avoid a near-certain 429."""
    now = time.time()
    with _token_lock:
        while _token_usage_window and _token_usage_window[0][0] < now - 60:
            _token_usage_window.popleft()
        used = sum(t for _, t in _token_usage_window)
        if used + estimated > TPM_BUDGET:
            return False
        _token_usage_window.append((now, estimated))
        return True
# -----------------------------------------------------------------------------


# Both /v1/tick and /v1/reply have a 30s budget from the judge. compose() can now make up
# to 2 LLM calls (initial + one retry after a rejected fabrication), so each call's timeout
# must leave room for both to fit under budget with margin for parsing/network overhead.
# /v1/tick also now runs multiple triggers' compose() calls CONCURRENTLY (see main.py),
# so the worst case for the whole tick is bounded by the SLOWEST single trigger's two
# calls, not the sum across all triggers -- but that single worst case still needs to
# fit comfortably under 30s including all other tick-handler overhead.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))

# Reuse one client per provider instead of constructing a new one (with its own
# connection pool) on every single call. On a memory-constrained free-tier host,
# creating a fresh client per call under concurrent load was a real contributor to
# memory pressure severe enough to crash the whole process (observed: /v1/healthz
# itself started timing out and the service's uptime reset, meaning it had been
# killed and restarted by the host).
_client_cache: dict = {}

def _get_client(provider: str):
    if provider in _client_cache:
        return _client_cache[provider]
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""), timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,  # SDK default is 2 automatic retries -- each with its own
            # full timeout, silently turning a "10s" budget into ~30s. Disabled: our own
            # fallback logic already handles failures; we don't need the SDK retrying
            # underneath us and blowing the tick deadline as a result.
        )
    elif provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment / .env file")
        client = OpenAI(
            api_key=api_key, base_url="https://api.groq.com/openai/v1",
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,  # same reasoning as above -- this was very likely the real
            # cause of compose() calls observed taking 20-37s despite a 10s timeout.
        )
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=LLM_TIMEOUT_SECONDS,
                         max_retries=0)
    _client_cache[provider] = client
    return client


# LLM client (supports OpenAI + Anthropic + Groq)
def _llm_complete(system: str, user: str, temperature: float = 0.0) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        client = _get_client(provider)
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        resp = client.messages.create(
            model=model, max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    elif provider == "groq":
        client = _get_client(provider)
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        # Groq doesn't support response_format=json_object for all models,
        # so we ask for JSON in the prompt and parse manually.
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    else:  # default openai
        client = _get_client(provider)
        model = os.getenv("LLM_MODEL", "gpt-4o")
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


# Prompt templates per trigger kind
# NOTE: kept deliberately terse -- your Groq account is capped at 8K tokens/minute for
# this model, so every token here is budget taken away from actually being able to make
# more than a handful of calls per minute. Verbosity here has a real, measured cost.
SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. Compose WhatsApp messages for Indian merchants/customers.
Return ONLY this JSON: {"body":"...","cta":"...","send_as":"...","suppression_key":"...","rationale":"..."}

Rules:
1. body: no preamble/re-intro. No hard length cap but don't pad -- ~250-450 chars is the sweet spot.
2. cta: one of binary_yes_no | binary_confirm_cancel | open_ended | multi_choice_slot | none
3. send_as: "vera" (merchant-facing) or "merchant_on_behalf" (customer-facing)
4. suppression_key: copy from trigger
5. rationale: 1-2 sentences, must match what body actually does
6. NEVER fabricate: every number/date/name/claim must trace to the context given. Derived math (e.g. "22 of 240") must be arithmetically consistent.
7. Never use category voice_taboo words.
8. Never include a URL -- automatic hard fail.
9. Always use the real first name from context -- never a bare "Hi".
10. Anchor on >=1 verifiable number/date/citation. Research/compliance triggers MUST cite source.
11. Hindi-English code-mix OK/encouraged for hi/hi-en merchants.
12. Exactly ONE cta, in the LAST sentence. No stacked asks.
13. Prefer "Service @ Rs price" over generic "X% off".
14. Customer-facing: honor language/slot-preference/consent exactly; never invent customer detail.
15. Emoji: 1 max, category-fitting, never for pharmacies.
16. Use category vocab_allowed naturally where it fits.
17. Add real judgment (e.g. "skip this promo, data says it underperforms") over template-filling.
18. One ask only -- not three.
19. Category offer_catalog/vocab_allowed = STYLE REFERENCE ONLY, not proof this merchant has it. Only
claim a specific service/class/offer if it's in THIS merchant's own active `offers` -- especially for
customer-facing "we've added X" claims.
"""

COMPOSE_PROMPT = """=== CONTEXT ===

CATEGORY ({slug}):
Voice tone: {voice_tone}
Vocab taboos: {taboos}
Active offers (catalog): {offer_catalog}
Peer stats: {peer_stats}
Digest (latest): {digest}
Seasonal beats: {seasonal_beats}
Trend signals: {trend_signals}

MERCHANT:
ID: {merchant_id}
Name: {merchant_name}
Owner: {owner_name}
City: {city}, Locality: {locality}
Verified: {verified}
Languages: {languages}
Subscription: {subscription}
Performance (30d): views={views}, calls={calls}, CTR={ctr} (peer median CTR={peer_ctr})
CTR vs peer: {ctr_vs_peer}
Active offers: {active_offers}
Signals: {signals}
Recent conversation: {convo_history}
Customer aggregate: {customer_aggregate}
Review themes: {review_themes}

TRIGGER:
ID: {trigger_id}
Kind: {trigger_kind}
Source: {trigger_source}
Urgency: {urgency}/5
Payload: {trigger_payload}
Suppression key: {suppression_key}
Expires: {expires_at}

CUSTOMER (if present):
{customer_block}

=== TASK ===
Compose the best possible Vera message for this (merchant, trigger) combination.
Trigger kind = "{trigger_kind}" — specific guidance:
{kind_guidance}

Return JSON only. No markdown, no extra keys.
"""

KIND_GUIDANCE = {
    "research_digest": (
        "Lead with the specific research finding (numbers + source). Reference which patient segment "
        "in THIS merchant's roster it applies to. End with a low-friction offer to draft/pull content for them."
    ),
    "regulation_change": (
        "Lead with the regulatory deadline and what changes. Tell them EXACTLY what action they need to take "
        "before the deadline. Use urgency — compliance failure has real consequences."
    ),
    "cde_opportunity": (
        "Lead with the CDE credit count and cost. Who is the speaker or topic? What's the tangible value "
        "for their practice? Single yes/no CTA."
    ),
    "perf_dip": (
        "Name the exact metric and the % drop. Offer a diagnosis AND a concrete next step. "
        "Don't just describe the problem — give them something actionable right now."
    ),
    "perf_spike": (
        "Celebrate the spike with the exact number. Credit a likely driver if visible. "
        "Ask them to capitalize on the momentum — a specific action that extends it."
    ),
    "milestone_reached": (
        "Acknowledge the exact milestone number. Frame it as social proof. "
        "Suggest one action that turns the milestone into forward momentum."
    ),
    "dormant_with_vera": (
        "Re-engage without guilt. Lead with a new piece of value relevant to their category right now. "
        "Don't mention the dormancy — just give them a reason to re-engage."
    ),
    "review_theme_emerged": (
        "Name the theme and the occurrence count. Offer to draft a response template or fix. "
        "Position it as 'I noticed' — reciprocity, not accusation."
    ),
    "competitor_opened": (
        "Name the competitor + distance + their offer. Reframe as an opportunity. "
        "Suggest a specific counter-move anchored in THIS merchant's strengths."
    ),
    "festival_upcoming": (
        "Name the festival + days until. Suggest a specific service+price campaign relevant to "
        "their category. End with offer to draft the GBP post or WhatsApp blast."
    ),
    "ipl_match_today": (
        "Name the match + venue + time. Use the seasonal data (weeknight vs weekend pattern). "
        "Recommend the smart play — which existing offer to push, or not to push."
    ),
    "renewal_due": (
        "Be direct — X days left. Show what they'd lose (profile paused, visibility drop). "
        "Single confirm CTA. Don't beg — frame as their business interest."
    ),
    "curious_ask_due": (
        "Ask the merchant ONE specific question about their business right now. "
        "Offer to turn their answer into a ready-made artifact (post, reply template, etc.)."
    ),
    "winback_eligible": (
        "Lead with what they've missed since expiry (specific metric). "
        "Make re-subscribing feel effortless — one confirm CTA."
    ),
    "active_planning_intent": (
        "The merchant already said yes — DO NOT ask qualifying questions. "
        "Deliver the concrete plan/artifact they asked for. End with a confirm/execute CTA."
    ),
    "seasonal_perf_dip": (
        "Normalize the dip with the peer data range. Reframe as the right time for retention focus. "
        "Give one specific retention action they can take this week."
    ),
    "gbp_unverified": (
        "Name the specific uplift % they'd get from verifying. "
        "Tell them exactly how to verify (postcard or phone call). Single CTA."
    ),
    "recall_due": (
        "Customer-facing. Name the service + how long since last visit. "
        "Offer specific slots matching their preference. Real price from the catalog."
    ),
    "customer_lapsed_soft": (
        "Customer-facing. No-shame, warm re-engagement. Name a new/relevant service or offer. "
        "Single commitment — no obligation framing removes the friction."
    ),
    "customer_lapsed_hard": (
        "Customer-facing. Acknowledge the gap without guilt. Give a specific new reason to return "
        "(new class, new offer, new capability). No-commitment trial."
    ),
    "trial_followup": (
        "Customer-facing. Reference their trial experience. "
        "Give one specific next session slot with the price. Single yes CTA."
    ),
    "chronic_refill_due": (
        "Customer-facing. List the molecules + runout date. Show price + savings with applicable offers. "
        "Free delivery if applicable. Reply CONFIRM CTA."
    ),
    "appointment_tomorrow": (
        "Customer-facing. Confirm date + time + service. "
        "Add one small value-add reminder (what to bring, how to prepare)."
    ),
    "supply_alert": (
        "Urgent compliance — name exact batch numbers + molecule. "
        "Tell them how many of THEIR customers are affected (from aggregate). "
        "Offer to draft the customer notification + replacement workflow."
    ),
    "category_seasonal": (
        "Name 2-3 specific demand shifts with numbers. "
        "Suggest ONE concrete shelf or service action for each. Quick wins only."
    ),
    "wedding_package_followup": (
        "Customer-facing. Reference the trial they did. Days to wedding count. "
        "Suggest the next program with price + specific slot. Single booking CTA."
    ),
}


def _build_customer_block(customer: Optional[dict]) -> str:
    if not customer:
        return "None — this is a merchant-facing message."
    idn = customer.get("identity", {})
    rel = customer.get("relationship", {})
    return (
        f"Name: {idn.get('name', '?')}\n"
        f"Language pref: {idn.get('language_pref', 'en')}\n"
        f"Age band: {idn.get('age_band', '?')}\n"
        f"State: {customer.get('state', '?')}\n"
        f"Last visit: {rel.get('last_visit', '?')}, Total visits: {rel.get('visits_total', '?')}\n"
        f"Services: {rel.get('services_received', [])}\n"
        f"Preferences: {customer.get('preferences', {})}\n"
        f"Consent scope: {customer.get('consent', {}).get('scope', [])}"
    )


# The challenge spec is explicit: "No hard body-length cap. Messages are judged on
# quality, specificity, and relevance." (challenge-testing-brief §Failure-mode F.3).
# This is only a sanity ceiling against a runaway/looping LLM output, NOT a target —
# the real winning examples in case-studies.md run 250-450 chars freely.
SANITY_CEILING = 900

def smart_trim(body: str, limit: int = SANITY_CEILING) -> str:
    """Only trims pathologically long output; never touches normal-length messages."""
    if len(body) <= limit:
        return body
    sents = re.split(r"(?<=[.!?]) +", body.strip())
    if len(sents) <= 2:
        return body[: limit - 1].rstrip() + "…"
    first, last = sents[0], sents[-1]
    kept = []
    for s in sents[1:-1]:
        if len(" ".join([first] + kept + [s, last])) <= limit:
            kept.append(s)
        else:
            break
    out = " ".join([first] + kept + [last])
    if len(out) > limit:
        first = first[: limit - len(last) - 3].rstrip() + "…"
        out = first + " " + last
    return out


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

def strip_urls(body: str) -> str:
    """Hard requirement (F.4): any URL in body is an automatic -3 fail. Belt-and-suspenders
    against the LLM slipping one in — strip it rather than let the message ship broken."""
    if not body:
        return body
    return re.sub(r"\s{2,}", " ", _URL_RE.sub("", body)).strip()


def _context_blob(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Flattened lowercase text of everything actually provided, for fabrication checks."""
    parts = [
        json.dumps(category, ensure_ascii=False),
        json.dumps(merchant, ensure_ascii=False),
        json.dumps(trigger, ensure_ascii=False),
    ]
    if customer:
        parts.append(json.dumps(customer, ensure_ascii=False))
    return " ".join(parts).lower()


# Citation-style clauses the LLM tends to invent when it wants to sound authoritative:
# "— <Proper Noun ...>, <Month> <Year>" or "circular NNN/YYYY". If the distinctive proper
# noun in a citation doesn't appear anywhere in the actual context we gave the model, the
# citation -- and therefore likely the claim it's attached to -- is fabricated. This is a
# real, observed failure mode (GPT-OSS-120B via Groq invented a "GST Council circular
# 224/2026" and a "Zomato partner update, Apr 2026" that exist nowhere in the pushed
# context), not a hypothetical edge case.
_CITATION_RE = re.compile(
    r"[—–-]\s*([A-Z][A-Za-z&.]+(?:\s+[A-Z][A-Za-z&.]+){0,4}"
    r"(?:\s*,?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})?)"
)
_REGULATION_RE = re.compile(r"\bcircular\s+[\w/.\-]+\b", re.IGNORECASE)

def _has_fabricated_citation(body: str, context_blob: str) -> Optional[str]:
    """Returns the offending fragment if body cites something untraceable to context."""
    reg_match = _REGULATION_RE.search(body)
    if reg_match and "circular" not in context_blob:
        return reg_match.group()
    for match in _CITATION_RE.finditer(body):
        citation_text = match.group(1)
        tokens = [
            t for t in re.findall(r"[A-Za-z]+", citation_text)
            if t.lower() not in ("jan", "feb", "mar", "apr", "may", "jun", "jul",
                                  "aug", "sep", "oct", "nov", "dec")
            and len(t) > 2
        ]
        if not tokens:
            continue
        if not any(t.lower() in context_blob for t in tokens):
            return citation_text
    return None


_NEW_ADDITION_RE = re.compile(
    r"(?:we'?ve\s+(?:\w+\s+){0,2}(?:added|introduced|launched|started|got)|"
    r"we\s+now\s+(?:\w+\s+){0,2}(?:have|offer)|"
    r"since\s+then\s+we'?ve\s+(?:\w+\s+){0,2}(?:added|introduced)|"
    r"new (?:class|service|instructor|trainer|program|batch|session)es?|"
    r"introduced a new|hum ne|naya(?:\s+\w+)?\s+shuru)"
    r"\s+((?:(?!\bplease\b|\breply\b|\band\b\s)[\w'/,]+\s*){1,8})",
    re.IGNORECASE,
)
_STOPWORDS = {"a", "an", "the", "new", "we", "you", "your", "for", "to", "and", "at",
              "with", "plus", "led", "by", "class", "classes", "service", "services",
              "of", "our", "is", "are", "this", "that", "perfect", "still", "certified",
              "just", "now", "recently"}

def _has_unconfirmed_merchant_service(body: str, category: dict, merchant: dict, is_customer_facing: bool) -> Optional[str]:
    """For customer-facing messages: flag any claim that the merchant added/introduced
    something new, OR any category-vocabulary "service word" (yoga, HIIT, aligner, etc.)
    used as if it's specifically this merchant's, unless verifiable in the merchant's OWN
    data (offers, signals, conversation history, review themes) or the customer's own
    service history. Three real observed failures fixed progressively: a category-catalog
    item presented as merchant-confirmed; a free-form invented class matching no catalog
    item; and the same invented class rephrased with an inserted word ("we've JUST added")
    that broke a too-rigid phrase regex. Given LLM phrasing varies endlessly, this also
    checks the underlying vocabulary noun directly, not just the surrounding verb phrase,
    as a second line of defense."""
    if not is_customer_facing:
        return None
    norm = lambda s: re.sub(r"[-–—]", " ", s).lower()
    body_norm = norm(body)

    merchant_own_blob = norm(" ".join([
        json.dumps(merchant.get("offers", []), ensure_ascii=False),
        json.dumps(merchant.get("signals", []), ensure_ascii=False),
        json.dumps(merchant.get("conversation_history", []), ensure_ascii=False),
        json.dumps(merchant.get("review_themes", []), ensure_ascii=False),
    ]))

    # 1. Exact category-catalog item presented as merchant's own
    active_offer_titles = norm(" ".join(
        o.get("title", "") for o in merchant.get("offers", []) if o.get("status") == "active"
    ))
    for item in category.get("offer_catalog", []):
        title_key = norm(re.sub(r"@.*|₹.*", "", item.get("title", "")).strip())
        if len(title_key) >= 6 and title_key in body_norm and title_key not in active_offer_titles:
            return item.get("title")

    # 2. "We added/introduced X" pattern (tolerant of inserted words like "just"/"recently")
    for match in _NEW_ADDITION_RE.finditer(body):
        noun_phrase = match.group(1)
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", noun_phrase)
                  if t.lower() not in _STOPWORDS and len(t) > 2]
        if tokens and not any(t in merchant_own_blob for t in tokens):
            return match.group(0).strip()

    # 3. Second line of defense: any category vocab_allowed "service word" (yoga, HIIT,
    # aligner, whitening, ...) appearing in the body but nowhere in the merchant's own
    # data. Catches phrasings the verb-pattern regex above still misses.
    vocab_words = [v.lower() for v in category.get("voice", {}).get("vocab_allowed", [])
                   if len(v) > 3]
    for word in vocab_words:
        if word in body_norm and word not in merchant_own_blob:
            return f"category vocabulary term '{word}' used as if merchant-specific"

    return None


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    allow_retry: bool = True,
) -> dict:
    """
    Main composition entry point.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale

    allow_retry: if a fabrication is caught, whether to give the LLM one corrected
    attempt (better quality) or fall straight to the safe deterministic fallback
    (bounded to exactly 1 LLM call, much lower worst-case latency). /v1/tick processes
    multiple triggers per call under a hard wall-clock budget -- with a reasoning model
    (visible "reasoning" tokens generated before every answer), two sequential calls per
    trigger multiplied across several triggers can blow the budget even with concurrency.
    main.py passes allow_retry=False for tick; the retry path stays available for
    lower-volume/single-message use.
    """
    slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest = category.get("digest", [])
    seasonal_beats = category.get("seasonal_beats", [])
    trend_signals = category.get("trend_signals", [])

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    convo = merchant.get("conversation_history", [])
    convo_summary = [
        f"[{t.get('from','?')} @ {t.get('ts','')[:10]}]: {t.get('body','')[:80]}"
        for t in convo[-1:]
    ] if convo else ["(no recent conversation)"]

    peer_ctr = peer_stats.get("avg_ctr", 0.03)
    merchant_ctr = perf.get("ctr", 0)
    ctr_delta = round((merchant_ctr - peer_ctr) / peer_ctr * 100, 0) if peer_ctr else 0
    ctr_vs_peer = (
        f"BELOW peer by {abs(ctr_delta):.0f}%" if ctr_delta < -5 else
        f"ABOVE peer by {ctr_delta:.0f}%" if ctr_delta > 5 else
        "AT peer median"
    )

    trigger_kind = trigger.get("kind", "")
    kind_guidance = KIND_GUIDANCE.get(trigger_kind, "Compose a relevant, specific, compelling message.")

    # Find the relevant digest item if trigger references one
    top_item_id = trigger.get("payload", {}).get("top_item_id") or trigger.get("payload", {}).get("digest_item_id")
    relevant_digest = []
    if top_item_id:
        relevant_digest = [d for d in digest if d.get("id") == top_item_id]
    if not relevant_digest:
        relevant_digest = digest[:1]  # only the single top item -- token budget is tight

    prompt = COMPOSE_PROMPT.format(
        slug=slug,
        voice_tone=voice.get("tone", ""),
        taboos=voice.get("vocab_taboo", [])[:5],
        offer_catalog=[o["title"] for o in category.get("offer_catalog", [])[:4]],
        peer_stats={k: v for k, v in peer_stats.items() if k in
                    ("avg_ctr", "avg_rating", "avg_review_count", "avg_views_30d", "avg_calls_30d")},
        digest=relevant_digest,
        seasonal_beats=seasonal_beats[:2],
        trend_signals=trend_signals[:2],
        merchant_id=merchant.get("merchant_id", ""),
        merchant_name=identity.get("name", ""),
        owner_name=identity.get("owner_first_name", ""),
        city=identity.get("city", ""),
        locality=identity.get("locality", ""),
        verified=identity.get("verified", False),
        languages=identity.get("languages", ["en"]),
        subscription=merchant.get("subscription", {}),
        views=perf.get("views", 0),
        calls=perf.get("calls", 0),
        ctr=perf.get("ctr", 0),
        peer_ctr=peer_ctr,
        ctr_vs_peer=ctr_vs_peer,
        active_offers=active_offers or ["(none)"],
        signals=signals,
        convo_history=convo_summary,
        customer_aggregate=merchant.get("customer_aggregate", {}),
        review_themes=merchant.get("review_themes", [])[:2],
        trigger_id=trigger.get("id", ""),
        trigger_kind=trigger_kind,
        trigger_source=trigger.get("source", ""),
        urgency=trigger.get("urgency", 1),
        trigger_payload=json.dumps(trigger.get("payload", {}), ensure_ascii=False),
        suppression_key=trigger.get("suppression_key", ""),
        expires_at=trigger.get("expires_at", ""),
        customer_block=_build_customer_block(customer),
        kind_guidance=kind_guidance,
    )

    def _try_llm_compose(extra_instruction: str = "") -> tuple:
        """One LLM attempt. Returns (parsed dict or None, error string or None)."""
        full_prompt = prompt + (f"\n\n{extra_instruction}" if extra_instruction else "")
        estimated = _estimate_tokens(SYSTEM_PROMPT, full_prompt)
        try:
            has_budget = _reserve_token_budget(estimated)
        except Exception:
            has_budget = True  # fail OPEN -- a bug in the guard itself must never be able
            # to take down real composition (this exact failure mode was observed: an
            # unrelated NameError in this guard silently zeroed out every single action
            # for a full test run before this fix).
        if not has_budget:
            return None, (
                f"token budget guard: ~{estimated} tokens needed but the rolling 60s window "
                f"is already near the {TPM_BUDGET} TPM cap -- skipping to avoid a near-certain "
                f"429 and preserve remaining budget for other in-flight requests"
            )
        try:
            raw = _llm_complete(SYSTEM_PROMPT, full_prompt, temperature=0.0)
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw.strip())
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                raw = m.group()
            return json.loads(raw), None
        except Exception as e:
            return None, str(e)

    result, err = _try_llm_compose()
    if result is None:
        result = _fallback_compose(category, merchant, trigger, customer)
        result["rationale"] += f" [LLM error: {err}]"

    # Ensure suppression_key is always set
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger_kind}")

    # Ensure send_as is correct
    if customer and not result.get("send_as"):
        result["send_as"] = "merchant_on_behalf"
    elif not result.get("send_as"):
        result["send_as"] = "vera"

    result["body"] = strip_urls(smart_trim(result.get("body", "")))

    # Anti-fabrication guardrail: if the LLM cited something untraceable to the actual
    # context (invented regulation, invented partner program, invented service), the
    # message fails the challenge's core "never fabricate" rule regardless of how
    # polished it reads. Rather than dropping straight to the plain deterministic
    # fallback (safe but reads like a data dump — costs Category Fit / Engagement),
    # give the LLM one retry with explicit feedback about exactly what was fabricated.
    # Only fall back to the mechanical version if the retry ALSO fails the check.
    blob = _context_blob(category, merchant, trigger, customer)

    def _check(body: str) -> tuple:
        bad_citation = _has_fabricated_citation(body, blob)
        bad_service = _has_unconfirmed_merchant_service(
            body, category, merchant, is_customer_facing=bool(customer)
        )
        return bad_citation, bad_service

    bad_citation, bad_service = _check(result["body"])

    if bad_citation or bad_service:
        offending = bad_citation or bad_service

        if allow_retry:
            retry_instruction = (
                f"Your previous attempt included this unverifiable claim: \"{offending}\". "
                "It does not appear anywhere in the context provided above. Rewrite the "
                "message using ONLY facts, offers, and services that are literally present "
                "in the category/merchant/trigger/customer context given. Do not invent any "
                "new class, service, instructor, or citation. Return JSON only."
            )
            retry_result, retry_err = _try_llm_compose(retry_instruction)
            if retry_result is not None:
                if not retry_result.get("suppression_key"):
                    retry_result["suppression_key"] = result["suppression_key"]
                if customer and not retry_result.get("send_as"):
                    retry_result["send_as"] = "merchant_on_behalf"
                elif not retry_result.get("send_as"):
                    retry_result["send_as"] = "vera"
                retry_result["body"] = strip_urls(smart_trim(retry_result.get("body", "")))
                retry_bad_citation, retry_bad_service = _check(retry_result["body"])
                if not (retry_bad_citation or retry_bad_service) and retry_result["body"].strip():
                    retry_result["rationale"] = (
                        retry_result.get("rationale", "")
                        + f" [Retried after first attempt was rejected for: '{offending}']"
                    )
                    return retry_result
            retry_note = f"; retry error: {retry_err}" if retry_err else "; retry still fabricated"
        else:
            # Retry disabled (e.g. from /v1/tick, where a second sequential LLM call per
            # trigger risks blowing the batch's wall-clock budget) -- go straight to the
            # safe, bounded-to-one-call fallback.
            retry_note = "; retry disabled for this call path (latency budget)"

        # Retry either disabled, failed outright, or still fabricated -- use the safe fallback.
        fallback = _fallback_compose(category, merchant, trigger, customer)
        reason = (
            f"unverifiable citation '{bad_citation}'" if bad_citation
            else f"claimed unconfirmed merchant service '{bad_service}' (only in category catalog, not merchant's own active offers)"
        )
        fallback["rationale"] += (
            f" [LLM output rejected: {reason}, not traceable to pushed context{retry_note}]"
        )
        return fallback

    if not result["body"].strip():
        # URL-stripping or a degenerate LLM output left nothing usable — fall back rather
        # than ship an empty body (empty body = malformed, -2 penalty).
        result = _fallback_compose(category, merchant, trigger, customer)
    return result


def _harvest_facts(payload: dict, limit: int = 4) -> list:
    """Pull message-worthy facts (numbers, dates, short labels) from any payload."""
    if not payload or payload.get("placeholder") is True:
        # Thin/placeholder trigger (real gap in the dataset: ~75% of expanded triggers
        # are just {"placeholder": true, "metric_or_topic": kind}) -- there is nothing
        # genuinely informative to harvest here. Returning [] forces the caller to lean
        # on merchant/category-level facts instead of echoing the placeholder's own
        # field name back as if it were a real detail (e.g. "metric or topic: X").
        return []
    out = []
    def walk(o, prefix="", depth=0):
        if depth > 4 or len(out) > limit * 3:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                label = k.replace("_", " ")
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    if "pct" in k or (isinstance(v, float) and -1 <= v <= 1):
                        out.append((2, f"{label} {round(v * 100)}%"))
                    else:
                        out.append((1, f"{label}: {v:,}" if isinstance(v, int) else f"{label}: {v}"))
                elif isinstance(v, str):
                    if re.match(r"\d{4}-\d{2}-\d{2}", v):
                        out.append((1, f"{label}: {v[:10]}"))
                    elif 2 < len(v) <= 60 and "id" not in k.lower():
                        out.append((3, f"{label}: {v.replace('_', ' ')}"))
                elif isinstance(v, (dict, list)):
                    walk(v, label, depth + 1)
        elif isinstance(o, list):
            strs = [x for x in o if isinstance(x, str)][:3]
            if strs and prefix:
                out.append((2, f"{prefix}: {', '.join(s.replace('_', ' ') for s in strs)}"))
            for it in o[:3]:
                if isinstance(it, dict):
                    walk(it, prefix, depth + 1)
    walk(payload)
    out.sort(key=lambda t: t[0])
    seen, facts = set(), []
    for _, f in out:
        key = f.split(":")[0]
        if key not in seen:
            seen.add(key)
            facts.append(f)
        if len(facts) >= limit:
            break
    return facts


def _fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    """Rule-based fallback when LLM fails — packed with verifiable specifics so
    Specificity/Merchant Fit hold up even without the LLM."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "there")
    if category.get("slug") == "dentists" and identity.get("owner_first_name"):
        name = f"Dr. {identity['owner_first_name']}"
    kind = trigger.get("kind", "update").replace("_", " ")
    payload = trigger.get("payload", {})
    facts = _harvest_facts(payload)
    perf = merchant.get("performance", {})
    loc = identity.get("locality") or identity.get("city", "")
    stat = ""
    if perf.get("views") and perf.get("calls"):
        stat = f" For context, your profile pulled {perf['views']} views and {perf['calls']} calls in 30 days."

    if customer:
        cident = customer.get("identity", {})
        cname = (cident.get("name") or "there").split(" (")[0]
        mname = identity.get("name", "us")
        fact_txt = ("; ".join(facts[:2]) + ". ") if facts else ""
        body = (f"Hi {cname}, {mname} here. {fact_txt}"
                "Reply YES and we'll book your slot this week.")
        return {"body": smart_trim(body), "cta": "binary_yes_no",
                "send_as": "merchant_on_behalf",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": "Fallback: customer message with payload facts, single CTA."}

    # Merchant-facing. If the trigger payload was a real, populated one, lead with those
    # specifics. If it was a thin placeholder (facts == []), lean on merchant-level
    # context instead of a vague "I've lined up the next step" non-answer, and phrase the
    # kind naturally rather than echoing the raw snake_case trigger kind.
    kind_natural = kind  # already de-snake-cased above
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])

    if facts:
        body = f"{name}, heads up on {kind_natural}" + (f" for your {loc} listing" if loc else "") + "."
        body += " " + "; ".join(facts) + "."
        body += stat
        body += " Want me to put together next steps for this?"
    else:
        # No usable trigger-level facts -- ground the message in real merchant data instead
        # of the trigger payload, and ask a concrete, specific question rather than a
        # generic "reply YES to know more".
        body = f"{name}, a {kind_natural} update for your {loc or category.get('slug', 'business')} listing."
        if stat:
            body += stat
        elif active_offers:
            body += f" Your active offer right now is {active_offers[0]}."
        elif signals:
            body += f" Noting: {signals[0].replace('_', ' ')}."
        body += " Want me to look into how this affects your listing and suggest a next step?"

    return {"body": smart_trim(body), "cta": "binary_yes_no" if facts else "open_ended",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": (
                f"Fallback for '{kind}': payload facts + merchant anchors, no fabrication."
                if facts else
                f"Fallback for '{kind}': trigger payload was a thin placeholder, grounded in "
                f"merchant-level data instead rather than echoing the placeholder's own field names."
            )}
