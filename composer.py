"""
Vera Message Composer

LLM-powered composer that takes the 4 contexts and produces a high-scoring
WhatsApp message. Dispatches by trigger kind for best results.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

# LLM client (supports OpenAI + Anthropic)
def _llm_complete(system: str, user: str, temperature: float = 0.0) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        resp = client.messages.create(
            model=model, max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    elif provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment / .env file")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
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
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
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
SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages that
engage Indian merchants or their customers. You ALWAYS return valid JSON with these exact keys:
{"body": "...", "cta": "...", "send_as": "...", "suppression_key": "...", "rationale": "..."}

RULES (violating any costs heavy scoring penalty):
1. body — the WhatsApp message text. Concise, no preamble ("Hope you're well" etc.). HARD LIMIT 300 characters — bodies over 320 chars FAIL schema validation.
2. cta — one of: "binary_yes_no", "binary_confirm_cancel", "open_ended", "multi_choice_slot", "none"
3. send_as — "vera" for merchant-facing, "merchant_on_behalf" for customer-facing
4. suppression_key — copy from the trigger's suppression_key
5. rationale — 1-2 sentences explaining the compulsion lever used and why this message fits
6. NEVER fabricate data not present in the context JSON provided
7. NEVER use taboo words from the category voice profile
8. ALWAYS use the merchant/customer's actual name
9. ALWAYS anchor on at least one specific number, date, or source citation from the contexts
10. Hindi-English code-mix is ENCOURAGED for hi/hi-en merchants and customers
11. The last sentence should be the CTA — buried CTAs lose points
12. "X% off" generic discounts score LOWER than "Service @ ₹price" specifics
13. For customer-facing messages: honor language preference, preferred slot times, relationship state
14. For research/compliance triggers: include the source citation at end (e.g. — JIDA Oct 2026 p.14)
15. Emoji: 1 max, only if it fits the category (🦷 dental, 💇 salon, 🏋️ gym — not for pharmacies)
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


MAX_BODY = 315

def smart_trim(body: str, limit: int = MAX_BODY) -> str:
    """Trim at sentence boundaries; always keep first sentence + final CTA sentence."""
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


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Main composition entry point.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale
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
        f"[{t.get('from','?')} @ {t.get('ts','')}]: {t.get('body','')[:120]}"
        for t in convo[-3:]
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
        relevant_digest = digest[:3]  # top 3 by default

    prompt = COMPOSE_PROMPT.format(
        slug=slug,
        voice_tone=voice.get("tone", ""),
        taboos=voice.get("vocab_taboo", []),
        offer_catalog=[o["title"] for o in category.get("offer_catalog", [])[:6]],
        peer_stats={k: v for k, v in peer_stats.items() if k in
                    ("avg_ctr", "avg_rating", "avg_review_count", "avg_views_30d", "avg_calls_30d")},
        digest=relevant_digest,
        seasonal_beats=seasonal_beats,
        trend_signals=trend_signals[:3],
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
        review_themes=merchant.get("review_themes", []),
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

    try:
        raw = _llm_complete(SYSTEM_PROMPT, prompt, temperature=0.0)
        # strip markdown fences if any
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())
        # extract first JSON object if there's surrounding text
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            raw = m.group()
        result = json.loads(raw)
    except Exception as e:
        # fallback — rule-based message
        result = _fallback_compose(category, merchant, trigger, customer)
        result["rationale"] += f" [LLM error: {e}]"

    # Ensure suppression_key is always set
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger_kind}")

    # Ensure send_as is correct
    if customer and not result.get("send_as"):
        result["send_as"] = "merchant_on_behalf"
    elif not result.get("send_as"):
        result["send_as"] = "vera"

    result["body"] = smart_trim(result.get("body", ""))
    return result


def _harvest_facts(payload: dict, limit: int = 4) -> list:
    """Pull message-worthy facts (numbers, dates, short labels) from any payload."""
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

    body = f"{name}, flagging {kind}" + (f" for your {loc} listing" if loc else "") + " right now."
    if facts:
        body += " Key details: " + "; ".join(facts) + "."
    body += stat
    body += " I've lined up the recommended next step — reply YES and I'll walk you through it."
    return {"body": smart_trim(body), "cta": "binary_yes_no", "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": f"Fallback for '{kind}': payload facts + merchant anchors, no fabrication."}
