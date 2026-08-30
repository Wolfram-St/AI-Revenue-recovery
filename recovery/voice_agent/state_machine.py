"""Finite state dialogue automaton and dynamic concession matrix for Hinglish recovery.

Implements a bounded, non-hallucinating conversational state machine:
1. **Deterministic Dialogue Graph**:
   ``GREETING`` → ``DIAGNOSE`` → ``OFFER_PAYMENT_LINK`` → ``NEGOTIATE_CONCESSION`` → ``CAPTURE_PTP`` → ``SUCCESS_TERMINAL``
   with explicit terminal exits for ``OPT_OUT_STOP`` and ``DISPUTE_ESCALATE``.
2. **Dynamic Concession Matrix**:
   Calculates allowable waivers (5%-8%) or split installment options bounded by
   transaction amount and recovery probability.
3. **Bilingual Hinglish Dialogue Synthesis**:
   Generates natural speech prompts for Voice calls and formatted WhatsApp messages.
4. **Direct PTP Registration**:
   Automatically records negotiated payment dates as active Promise-to-Pay records.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from recovery.ptp_tracker import (
    PTPChannel,
    PromiseToPay,
    register_promise,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DialogueState(str, Enum):
    """Closed set of conversational recovery states."""
    GREETING = "GREETING"
    DIAGNOSE = "DIAGNOSE"
    OFFER_PAYMENT_LINK = "OFFER_PAYMENT_LINK"
    NEGOTIATE_CONCESSION = "NEGOTIATE_CONCESSION"
    CAPTURE_PTP = "CAPTURE_PTP"
    SUCCESS_TERMINAL = "SUCCESS_TERMINAL"
    OPT_OUT_STOP = "OPT_OUT_STOP"
    DISPUTE_ESCALATE = "DISPUTE_ESCALATE"


class CustomerIntent(str, Enum):
    """Recognized customer dialogue intents."""
    CONFIRM_ID = "CONFIRM_ID"
    TECH_ERROR = "TECH_ERROR"
    LACK_OF_FUNDS = "LACK_OF_FUNDS"
    REQUEST_DISCOUNT = "REQUEST_DISCOUNT"
    ACCEPT_OFFER = "ACCEPT_OFFER"
    PROMISE_TO_PAY = "PROMISE_TO_PAY"
    REFUSE_OUTREACH = "REFUSE_OUTREACH"
    DISPUTE_CHARGE = "DISPUTE_CHARGE"


class ConcessionType(str, Enum):
    """Types of authorized recovery concessions."""
    NONE = "NONE"
    WAIVER_PERCENT = "WAIVER_PERCENT"
    SPLIT_EMI = "SPLIT_EMI"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConcessionOffer:
    """Bounded financial concession calculated by the concession matrix.

    Attributes
    ----------
    concession_type:
        Type of concession (None, percentage waiver, or split EMI).
    waiver_pct:
        Discount percentage (e.g. 5.0 for 5%).
    waiver_inr:
        Calculated discount amount in INR.
    net_payable_inr:
        Net balance payable after concession.
    split_installments:
        Number of installment payments (default 1).
    terms_explanation:
        Human-readable explanation of terms for customer.
    """

    concession_type: ConcessionType
    waiver_pct: float
    waiver_inr: float
    net_payable_inr: float
    split_installments: int = 1
    terms_explanation: str = ""


@dataclass(frozen=True)
class DialogueTurn:
    """Output turn produced by the dialogue state machine.

    Attributes
    ----------
    state:
        The current dialogue state reached.
    agent_speech_hinglish:
        Natural Hinglish speech script for Voice / TTS.
    agent_whatsapp_text:
        Formatted WhatsApp message text.
    allowed_intents:
        List of valid intents expected from the customer in this state.
    active_concession:
        Active concession offer if applicable.
    ptp_created:
        Generated PromiseToPay record if a commitment was captured.
    is_terminal:
        Whether the dialogue session has concluded.
    """

    state: DialogueState
    agent_speech_hinglish: str
    agent_whatsapp_text: str
    allowed_intents: tuple[CustomerIntent, ...]
    active_concession: ConcessionOffer | None = None
    ptp_created: PromiseToPay | None = None
    is_terminal: bool = False


@dataclass
class DialogueSession:
    """Stateful dialogue session context."""

    session_id: str
    case_id: str
    customer_id: str
    customer_name: str
    amount_inr: float
    current_state: DialogueState = DialogueState.GREETING
    recovery_probability: float = 0.50
    channel: PTPChannel = PTPChannel.VOICE_AGENT
    active_concession: ConcessionOffer | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dynamic Concession Matrix
# ---------------------------------------------------------------------------

def calculate_concession_matrix(
    amount_inr: float,
    recovery_prob: float,
) -> ConcessionOffer:
    """Calculate bounded negotiation concession based on amount and recovery probability.

    Bounds:
    - Tier 1 (< ₹2,000): 0% waiver. Full amount payable.
    - Tier 2 (₹2,000 - ₹10,000): If recovery_prob < 0.60, max 5% waiver capped at ₹300.
    - Tier 3 (> ₹10,000): If recovery_prob < 0.70, max 8% waiver capped at ₹800 OR 3-month split.
    """
    if amount_inr < 2000.0:
        return ConcessionOffer(
            concession_type=ConcessionType.NONE,
            waiver_pct=0.0,
            waiver_inr=0.0,
            net_payable_inr=round(amount_inr, 2),
            split_installments=1,
            terms_explanation="Instant 1-click payment via UPI QR or Card.",
        )

    if 2000.0 <= amount_inr <= 10000.0:
        if recovery_prob < 0.60:
            waiver_inr = min(amount_inr * 0.05, 300.0)
            waiver_pct = round((waiver_inr / amount_inr) * 100, 1)
            net_amt = round(amount_inr - waiver_inr, 2)
            return ConcessionOffer(
                concession_type=ConcessionType.WAIVER_PERCENT,
                waiver_pct=waiver_pct,
                waiver_inr=round(waiver_inr, 2),
                net_payable_inr=net_amt,
                split_installments=1,
                terms_explanation=f"Special {waiver_pct}% instant settlement waiver (₹{waiver_inr:.2f} off).",
            )
        return ConcessionOffer(
            concession_type=ConcessionType.NONE,
            waiver_pct=0.0,
            waiver_inr=0.0,
            net_payable_inr=round(amount_inr, 2),
            split_installments=1,
            terms_explanation="Standard 1-click payment link.",
        )

    # Tier 3 (> ₹10,000)
    if recovery_prob < 0.70:
        waiver_inr = min(amount_inr * 0.08, 800.0)
        waiver_pct = round((waiver_inr / amount_inr) * 100, 1)
        net_amt = round(amount_inr - waiver_inr, 2)
        return ConcessionOffer(
            concession_type=ConcessionType.SPLIT_EMI,
            waiver_pct=waiver_pct,
            waiver_inr=round(waiver_inr, 2),
            net_payable_inr=net_amt,
            split_installments=3,
            terms_explanation=(
                f"3-Month No-Cost EMI or instant {waiver_pct}% waiver (₹{waiver_inr:.2f} off). "
                f"Net payable: ₹{net_amt:.2f}"
            ),
        )

    return ConcessionOffer(
        concession_type=ConcessionType.NONE,
        waiver_pct=0.0,
        waiver_inr=0.0,
        net_payable_inr=round(amount_inr, 2),
        split_installments=1,
        terms_explanation="Full balance payment link.",
    )


# ---------------------------------------------------------------------------
# Dialogue State Machine Transition Engine
# ---------------------------------------------------------------------------

def transition_dialogue(
    session: DialogueSession,
    intent: CustomerIntent | None = None,
    payload: dict[str, Any] | None = None,
) -> DialogueTurn:
    """Transition dialogue state machine based on customer intent and session state."""
    payload = payload or {}
    name = session.customer_name
    amt = session.amount_inr

    # Initial Turn: GREETING
    if session.current_state == DialogueState.GREETING and intent is None:
        speech = (
            f"Namaste {name} ji! Main RecoverAI se baat kar raha hoon. "
            f"Aapka recent order payment ₹{amt:.2f} ka decline ho gaya tha. "
            "Kya main aapki identity confirm karne ke liye 1 minute baat kar sakta hoon?"
        )
        wa = (
            f"Namaste {name} ji, aapka recent payment of *₹{amt:.2f}* fail ho gaya tha. "
            "Kya aap transaction complete karna chahte hain?"
        )
        return DialogueTurn(
            state=DialogueState.GREETING,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(CustomerIntent.CONFIRM_ID, CustomerIntent.REFUSE_OUTREACH, CustomerIntent.DISPUTE_CHARGE),
            is_terminal=False,
        )

    # ── Universal Escape 1: Customer Refusal / Opt-Out ──────────────────
    if intent == CustomerIntent.REFUSE_OUTREACH:
        session.current_state = DialogueState.OPT_OUT_STOP
        speech = (
            f"Ji shukriya {name} ji. Humne aapka preference note kar liya hai aur future "
            "recovery communications stop kar diye hain. Have a nice day."
        )
        wa = (
            f"Thank you {name}. We have updated your preferences and stopped recovery outreach. "
            "For any queries, please visit our help center."
        )
        return DialogueTurn(
            state=DialogueState.OPT_OUT_STOP,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(),
            is_terminal=True,
        )

    # ── Universal Escape 2: Customer Dispute ─────────────────────────────
    if intent == CustomerIntent.DISPUTE_CHARGE:
        session.current_state = DialogueState.DISPUTE_ESCALATE
        speech = (
            f"Main samajh gaya {name} ji. Main aapka case hamari senior customer support "
            "team ko transfer kar raha hoon jo is dispute ko review karegi."
        )
        wa = (
            f"We have noted your dispute for transaction ₹{amt:.2f}. "
            "A senior support manager will review your account within 24 hours."
        )
        return DialogueTurn(
            state=DialogueState.DISPUTE_ESCALATE,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(),
            is_terminal=True,
        )

    # ── State: GREETING -> DIAGNOSE ──────────────────────────────────────
    if session.current_state == DialogueState.GREETING and intent == CustomerIntent.CONFIRM_ID:
        session.current_state = DialogueState.DIAGNOSE
        speech = (
            f"Shukriya {name} ji. Payment fail hone ka reason kya bank network issue tha, "
            "ya aap payment method change karna chahte hain?"
        )
        wa = (
            "Payment failure diagnosis: Was it due to a bank OTP/network issue, "
            "or would you like to use another payment mode (UPI / Card)?"
        )
        return DialogueTurn(
            state=DialogueState.DIAGNOSE,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(
                CustomerIntent.TECH_ERROR,
                CustomerIntent.LACK_OF_FUNDS,
                CustomerIntent.REQUEST_DISCOUNT,
                CustomerIntent.PROMISE_TO_PAY,
            ),
            is_terminal=False,
        )

    # ── State: DIAGNOSE -> OFFER_PAYMENT_LINK (Tech Issue) ───────────────
    if session.current_state == DialogueState.DIAGNOSE and intent == CustomerIntent.TECH_ERROR:
        session.current_state = DialogueState.OFFER_PAYMENT_LINK
        speech = (
            f"Theek hai {name} ji! Main aapke registered mobile par direct 1-click UPI QR link "
            f"bhej raha hoon taaki payment turant ho sake. Kya main abhi bhej doon?"
        )
        wa = (
            f"Here is your secure 1-click recovery link for ₹{amt:.2f}: "
            "https://pay.recoverai.io/link/quick_pay\nClick to complete via GPay / PhonePe / Paytm."
        )
        return DialogueTurn(
            state=DialogueState.OFFER_PAYMENT_LINK,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(CustomerIntent.ACCEPT_OFFER, CustomerIntent.PROMISE_TO_PAY),
            is_terminal=False,
        )

    # ── State: DIAGNOSE / REQUEST_DISCOUNT -> NEGOTIATE_CONCESSION ────────
    if intent in (CustomerIntent.LACK_OF_FUNDS, CustomerIntent.REQUEST_DISCOUNT):
        session.current_state = DialogueState.NEGOTIATE_CONCESSION
        concession = calculate_concession_matrix(amt, session.recovery_probability)
        session.active_concession = concession

        if concession.concession_type == ConcessionType.WAIVER_PERCENT:
            speech = (
                f"{name} ji, hamare system ne aapke liye special {concession.waiver_pct}% "
                f"settlement discount approve kiya hai. Aapko sirf ₹{concession.net_payable_inr:.2f} "
                "pay karna hoga. Kya aap abhi ya 2 din ke andar pay kar sakte hain?"
            )
            wa = (
                f"Special concession approved: Pay *₹{concession.net_payable_inr:.2f}* "
                f"(₹{concession.waiver_inr:.2f} waiver applied). Link: https://pay.recoverai.io/concession"
            )
        elif concession.concession_type == ConcessionType.SPLIT_EMI:
            speech = (
                f"{name} ji, hum aapke ₹{amt:.2f} payment ko 3 aasaan No-Cost monthly installments "
                f"ya instant ₹{concession.waiver_inr:.2f} discount ke sath split kar sakte hain. "
                "Aap kaunsa option prefer karenge?"
            )
            wa = (
                f"Flexible payment options for ₹{amt:.2f}:\n"
                f"Option 1: 3x Monthly installments of ₹{round(concession.net_payable_inr/3, 2)}\n"
                f"Option 2: Instant waiver of ₹{concession.waiver_inr:.2f}\n"
                "Select on: https://pay.recoverai.io/split"
            )
        else:
            speech = (
                f"{name} ji, hum payment date ko extend kar sakte hain. "
                "Aap payment kab tak complete karne ka commitment kar sakte hain?"
            )
            wa = (
                f"We can schedule your payment date. When would you like to complete the ₹{amt:.2f} payment?"
            )

        return DialogueTurn(
            state=DialogueState.NEGOTIATE_CONCESSION,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(CustomerIntent.ACCEPT_OFFER, CustomerIntent.PROMISE_TO_PAY),
            active_concession=concession,
            is_terminal=False,
        )

    # ── State: PROMISE_TO_PAY -> CAPTURE_PTP ──────────────────────────────
    if intent == CustomerIntent.PROMISE_TO_PAY or (session.current_state in (DialogueState.OFFER_PAYMENT_LINK, DialogueState.NEGOTIATE_CONCESSION) and intent == CustomerIntent.ACCEPT_OFFER):
        session.current_state = DialogueState.SUCCESS_TERMINAL
        promise_date_str = payload.get("promised_date") or (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)).isoformat()
        concession_dict = session.active_concession.terms_explanation if session.active_concession else {}
        promised_amt = session.active_concession.net_payable_inr if session.active_concession else amt

        ptp = register_promise(
            recovery_case_id=session.case_id,
            customer_id=session.customer_id,
            promised_amount_inr=promised_amt,
            promised_date=promise_date_str,
            channel_source=session.channel,
            concession_applied={"details": concession_dict} if concession_dict else {},
            transcript_snippet=payload.get("transcript_snippet") or "Customer agreed to complete payment.",
        )

        speech = (
            f"Bahut shukriya {name} ji! Humne aapka commitment record kar liya hai. "
            f"Payment link aapke WhatsApp par bhej diya gaya hai. Have a wonderful day!"
        )
        wa = (
            f"Payment Commitment Confirmed: ₹{promised_amt:.2f} by {promise_date_str[:10]}.\n"
            "Complete anytime using: https://pay.recoverai.io/ptp\nThank you!"
        )
        return DialogueTurn(
            state=DialogueState.SUCCESS_TERMINAL,
            agent_speech_hinglish=speech,
            agent_whatsapp_text=wa,
            allowed_intents=(),
            active_concession=session.active_concession,
            ptp_created=ptp,
            is_terminal=True,
        )

    # Fallback default turn
    speech = f"{name} ji, kya aap payment process karna chahte hain?"
    wa = f"{name}, would you like to proceed with payment of ₹{amt:.2f}?"
    return DialogueTurn(
        state=session.current_state,
        agent_speech_hinglish=speech,
        agent_whatsapp_text=wa,
        allowed_intents=(CustomerIntent.CONFIRM_ID, CustomerIntent.REFUSE_OUTREACH),
        is_terminal=False,
    )
