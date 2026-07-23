"""
Multi-turn conversation handler for Vera.
Handles: auto-reply detection, intent transitions, graceful exits, hostile messages.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from composer import _llm_complete, SYSTEM_PROMPT, smart_trim

# Auto-reply detection
AUTO_REPLY_PATTERNS = [
    r"thank you for contact",
    r"automated (reply|response|message)",
    r"i am (away|unavailable|out of office)",
    r"i.*ll (get back|respond|reply) (to you )?(shortly|soon|asap)",
    r"(business|working) hours",
    r"aapki (jaankari|madad)",
    r"aapka (sandesh|message) (hamare|hamari) team",
    r"main ek automated",
    r"yeh ek swachalit",
    r"out of office",
    r"will respond.*within.*hour",
    r"currently (unavailable|busy)",
    r"this is an automatic",
    r"auto.reply",
    r"do not reply to this",
    r"noreply",
]

HOSTILE_PATTERNS = [
    r"\bstop\b.*\bmessag",
    r"\bstop (messaging|sending|contacting)\b",
    r"not interested",
    r"\bspam\b",
    r"remove (me|my number)",
    r"don'?t (message|contact|bother|call) (me|us)",
    r"\bblock\b",
    r"band karo",
    r"mat bhejo",
    r"mujhe nahi chahiye",
    r"bother",
    r"useless",
    r"waste of time",
    r"irritating",
    r"annoying",
]

INTENT_POSITIVE_PATTERNS = [
    r"\b(yes|ok|okay|sure|alright|haan|ha)\b",
    r"let'?s do it",
    r"go ahead",
    r"sounds good",
    r"what'?s next",
    r"please (proceed|continue|do it|send)",
    r"kar do",
    r"karo",
    r"theek hai",
    r"bahut acha",
    r"bilkul",
    r"zaroor",
    r"i want to (join|start|proceed|sign up)",
    r"join (karna|chahta|chahti)",
    r"sign (me )?up",
    r"let'?s (proceed|go|start|do)",
    r"confirm",
    r"approved",
]


def is_auto_reply(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in AUTO_REPLY_PATTERNS)


def is_hostile(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in HOSTILE_PATTERNS)


def is_positive_intent(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in INTENT_POSITIVE_PATTERNS)


# Conversation state helpers
def get_auto_reply_count(turns: list[dict]) -> int:
    """Count consecutive auto-replies from the end of turn list."""
    count = 0
    for turn in reversed(turns):
        if turn.get("from") == "merchant" and is_auto_reply(turn.get("msg", "")):
            count += 1
        else:
            break
    return count


DATETIME_RE = re.compile(
    r"\b(?:(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+)?"
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,9},?\s+)?"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.IGNORECASE)


def pick_slot(msg: str, slots: list) -> Optional[str]:
    """Numbered pick, offered-slot match (>=2 tokens), or the customer's own day/time."""
    m = re.match(r"^\s*([12])\s*[.)]?\s*$", msg.strip())
    if m and slots:
        idx = int(m.group(1)) - 1
        if idx < len(slots):
            return slots[idx]
    low = msg.lower()
    tm = DATETIME_RE.search(msg)
    own = None
    if tm:
        prefix = re.search(
            r"((mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+(\d{1,2}\s+[a-z]{3,9},?\s+)?)$",
            msg[:tm.start()], re.IGNORECASE)
        own = ((prefix.group(1) if prefix else "") + tm.group(0)).strip().rstrip(",")
    for s in slots:
        toks = [t.strip(",.") for t in re.split(r"\s+", str(s).lower()) if len(t.strip(",.")) > 2]
        if toks and sum(1 for t in toks if t in low) >= 2:
            return s
    return own


def get_last_bot_body(turns: list[dict]) -> str:
    for turn in reversed(turns):
        if turn.get("from") == "bot":
            return turn.get("msg", "")
    return ""


# ---------------------------------------------------------------------------
# Reply composer prompt
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant in an ongoing WhatsApp conversation.
You ALWAYS return valid JSON with keys: {"action": "send"|"wait"|"end", "body": "...", "cta": "...", "rationale": "..."}

RULES:
- "action": "send" — compose the next reply
- "action": "wait" — back off; add "wait_seconds" key (int)
- "action": "end" — close the conversation gracefully
- body required only for "send"; omit for "wait"/"end"
- body: concise WhatsApp reply, max ~200 chars
- NO preamble, NO re-introduction
- Honor the conversation thread — answer what was asked
- If merchant said YES/confirmed → switch to ACTION mode, deliver the artifact
- If merchant gave a curveball (off-topic question) → politely decline, redirect
- Match merchant's language from the turn history
- Return JSON only, no markdown
"""

REPLY_PROMPT = """=== CONVERSATION HISTORY ===
{history}

=== LATEST MESSAGE (Turn {turn_number}) ===
From: {from_role}
Message: "{message}"

=== MERCHANT CONTEXT (brief) ===
Merchant: {merchant_name}, {city}
Category: {category_slug}
Active offers: {active_offers}
Trigger that started this: {trigger_kind}

=== TASK ===
This is turn {turn_number}. The merchant/customer just said: "{message}"

Situation: {situation}

Compose the ideal next move (send/wait/end).
If sending: deliver value, advance the conversation, honor their intent.
Return JSON only."""


def respond(
    state: dict,
    merchant_message: str,
    merchant: Optional[dict] = None,
    category: Optional[dict] = None,
) -> dict:
    """
    Given conversation state + the merchant's latest message → produce reply.
    state = {"conversation_id": str, "turns": [...], "trigger_kind": str, ...}
    """
    turns = state.get("turns", [])
    turn_number = state.get("turn_number", len(turns) + 1)

    # hard-rule checks first

    # 0. CUSTOMER-FACING conversation → deterministic slot handling BEFORE the LLM.
    #    We speak AS the merchant to the customer; merchant-campaign talk is wrong here.
    if state.get("from_role") == "customer" or state.get("customer_id"):
        if is_hostile(merchant_message):
            return {"action": "end",
                    "rationale": "Customer opt-out. Ending immediately and suppressing."}
        slots = state.get("slots") or []
        picked = pick_slot(merchant_message, slots)
        m_identity = (merchant or {}).get("identity", {})
        mname = m_identity.get("name", "us")
        if picked or is_positive_intent(merchant_message):
            slot_txt = picked or (slots[0] if slots else "your preferred time")
            state["booked"] = True
            body = (f"Perfect — you're booked for {slot_txt} at {mname}. "
                    "We'll send a reminder before your visit; reply here anytime to reschedule. "
                    "See you soon!")
            return {"action": "send", "body": smart_trim(body), "cta": "none",
                    "rationale": "Customer confirmed — booking locked with their stated slot."}
        # anything else from a customer: keep it warm and slot-focused, no LLM needed
        body = ("Sure — " + (f"open slots: {' or '.join(slots[:2])}. Reply 1"
                + (" or 2" if len(slots) > 1 else "") + " to confirm."
                if slots else "just reply with a day and time that suits you and we'll book it."))
        return {"action": "send", "body": smart_trim(body), "cta": "multi_choice_slot",
                "rationale": "Customer conversation — low-friction slot nudge."}

    # 1. Hostile message → end immediately
    if is_hostile(merchant_message):
        return {
            "action": "end",
            "rationale": "Merchant expressed frustration/opt-out. Closing conversation and suppressing.",
        }

    # 2. Auto-reply detection
    # main.py already incremented merchant_auto_reply_counts and wrote it into
    # state["auto_reply_count"] before calling respond(). Just read it here.
    if is_auto_reply(merchant_message):
        new_count = state.get("auto_reply_count", 1)  # already incremented by main.py
        if new_count >= 3:
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {new_count} times in a row. Owner not at phone. Closing.",
            }
        elif new_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Second consecutive auto-reply. Backing off 24 hours.",
            }
        else:
            mname = merchant.get("identity", {}).get("name", "") if merchant else ""
            return {
                "action": "send",
                "body": f"Looks like an auto-reply 😊 When {mname or 'the owner'} sees this, just reply YES to continue.",
                "cta": "binary_yes_no",
                "rationale": "Detected first auto-reply; one prompt to flag it for the owner.",
            }
    else:
        state["auto_reply_count"] = 0

    # 3. Positive intent transition → switch to action mode
    if is_positive_intent(merchant_message) and turn_number <= 4:
        state["intent_confirmed"] = True

    # --- LLM-based reply for everything else ---
    situation = _classify_situation(merchant_message, state)

    # Build history summary
    history_lines = []
    for t in turns[-6:]:
        role = "VERA" if t.get("from") == "bot" else "MERCHANT"
        history_lines.append(f"[{role}]: {t.get('msg', '')[:150]}")
    history_str = "\n".join(history_lines) if history_lines else "(conversation start)"

    m_identity = merchant.get("identity", {}) if merchant else {}
    active_offers = [o["title"] for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]

    prompt = REPLY_PROMPT.format(
        history=history_str,
        turn_number=turn_number,
        from_role=state.get("from_role", "merchant"),
        message=merchant_message,
        merchant_name=m_identity.get("name", ""),
        city=m_identity.get("city", ""),
        category_slug=category.get("slug", "") if category else "",
        active_offers=active_offers or ["(none)"],
        trigger_kind=state.get("trigger_kind", ""),
        situation=situation,
    )

    try:
        raw = _llm_complete(REPLY_SYSTEM, prompt, temperature=0.0)
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())
        result = json.loads(raw)

        # Safety: if body is the same as the last bot message, nudge differently
        if result.get("action") == "send":
            last = get_last_bot_body(turns)
            if last and result.get("body", "").strip() == last.strip():
                result["body"] = result["body"] + " — want me to go ahead?"
            result["body"] = smart_trim(result.get("body", ""))

        return result

    except Exception as e:
        # Grounded deterministic fallback — mirror their specifics, never generic.
        if turn_number >= 5:
            return {"action": "end", "rationale": f"Max turns reached. [err: {e}]"}
        m_identity = (merchant or {}).get("identity", {})
        name = m_identity.get("owner_first_name") or m_identity.get("name", "")
        if (category or {}).get("slug") == "dentists" and m_identity.get("owner_first_name"):
            name = f"Dr. {m_identity['owner_first_name']}"
        own = re.search(r"\b(?:we|i)\s+(?:have|use|run|are on|got)\s+(an?\s+)?([^.!?\n]{3,60})",
                        merchant_message, re.IGNORECASE)
        mirror = f" Since you're on {own.group(2).strip().rstrip(',')}, that's the first thing to tackle." if own else ""
        body = (f"On it{', ' + name if name else ''}.{mirror} "
                "I've prepared the step-by-step checklist tailored to your setup — "
                "reply YES and I'll send it now.")
        return {
            "action": "send",
            "body": smart_trim(body),
            "cta": "binary_yes_no",
            "rationale": f"Grounded fallback (mirrors merchant specifics). [LLM error: {e}]",
        }


def _classify_situation(message: str, state: dict) -> str:
    """Return a plain-text situation label to guide the LLM."""
    if is_positive_intent(message):
        return (
            "INTENT TRANSITION — merchant has CONFIRMED/ACCEPTED. "
            "Switch to action mode immediately. Do NOT ask qualifying questions. "
            "Deliver the artifact or next concrete step."
        )
    msg_lower = message.lower()
    off_topic_signals = ["gst", "tax", "legal", "police", "government", "loan", "bank", "insurance"]
    if any(s in msg_lower for s in off_topic_signals):
        return (
            "OFF-TOPIC request. Politely decline (it's outside Vera's scope), "
            "then redirect back to the original conversation topic."
        )
    question_signals = ["?", "kya", "kaise", "when", "how", "what", "why", "kitna", "kaisa",
                        "need help", "help me", "help us", "want to", "need to", "looking to",
                        "we have", "we use", "we run", "guide me", "madad"]
    if any(s in msg_lower for s in question_signals):
        own = re.search(r"\b(?:we|i)\s+(?:have|use|run|are on|got)\s+(an?\s+)?([^.!?\n]{3,60})",
                        message, re.IGNORECASE)
        mirror = (f' They told you: "{own.group(2).strip()}" — reference this EXACT detail '
                  'in your answer.') if own else ""
        return ("QUESTION / HELP REQUEST from merchant. Answer with SPECIFIC facts from the "
                "trigger/category context (deadlines, sources, numbers) — generic replies score "
                "poorly." + mirror + " End with one concrete next step.")
    negative_signals = ["no", "nahi", "nope", "not now", "later", "baad mein", "abhi nahi"]
    if any(s in msg_lower for s in negative_signals):
        return (
            "SOFT DECLINE. Acknowledge gracefully, offer to come back later or "
            "offer a lower-friction alternative. Don't push hard."
        )
    return "CONTINUING conversation. Advance naturally — answer, deliver, or invite next step."
