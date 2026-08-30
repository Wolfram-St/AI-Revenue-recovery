"""Tests for Bounded Hinglish Voice & WhatsApp Recovery Agent.

Covers:
- Dynamic concession matrix calculations across transaction tiers
- Finite state dialogue automaton transitions
- Golden tech-issue recovery flow
- Negotiation and PTP commitment flow
- Universal customer opt-out and dispute escalations
"""

from __future__ import annotations

import pytest

from recovery.ptp_tracker import PTPChannel, PTPStatus
from recovery.voice_agent.state_machine import (
    ConcessionOffer,
    ConcessionType,
    CustomerIntent,
    DialogueSession,
    DialogueState,
    DialogueTurn,
    calculate_concession_matrix,
    transition_dialogue,
)


class TestConcessionMatrix:
    def test_tier_1_small_amount_no_waiver(self):
        # Under ₹2,000 -> 0% waiver
        offer = calculate_concession_matrix(amount_inr=1500.0, recovery_prob=0.30)
        assert offer.concession_type == ConcessionType.NONE
        assert offer.waiver_pct == 0.0
        assert offer.net_payable_inr == 1500.0

    def test_tier_2_medium_amount_waiver_applied(self):
        # ₹5,000 with recovery_prob < 0.60 -> 5% waiver (₹250)
        offer = calculate_concession_matrix(amount_inr=5000.0, recovery_prob=0.40)
        assert offer.concession_type == ConcessionType.WAIVER_PERCENT
        assert offer.waiver_pct == 5.0
        assert offer.waiver_inr == 250.0
        assert offer.net_payable_inr == 4750.0

    def test_tier_2_capped_waiver(self):
        # ₹9,000 with 5% = ₹450, capped at max ₹300
        offer = calculate_concession_matrix(amount_inr=9000.0, recovery_prob=0.45)
        assert offer.concession_type == ConcessionType.WAIVER_PERCENT
        assert offer.waiver_inr == 300.0
        assert offer.net_payable_inr == 8700.0

    def test_tier_3_high_value_split_emi(self):
        # ₹25,000 with recovery_prob < 0.70 -> max 8% waiver (capped at ₹800) or 3-month split
        offer = calculate_concession_matrix(amount_inr=25000.0, recovery_prob=0.55)
        assert offer.concession_type == ConcessionType.SPLIT_EMI
        assert offer.split_installments == 3
        assert offer.waiver_inr == 800.0
        assert offer.net_payable_inr == 24200.0


class TestDialogueStateTransitions:
    def _create_session(self, amount: float = 4500.0, prob: float = 0.40) -> DialogueSession:
        return DialogueSession(
            session_id="sess_123",
            case_id="case_turn_01",
            customer_id="cust_rahul",
            customer_name="Rahul",
            amount_inr=amount,
            recovery_probability=prob,
            channel=PTPChannel.VOICE_AGENT,
        )

    def test_initial_greeting_turn(self):
        sess = self._create_session()
        turn = transition_dialogue(sess)
        assert turn.state == DialogueState.GREETING
        assert "Namaste Rahul ji" in turn.agent_speech_hinglish
        assert CustomerIntent.CONFIRM_ID in turn.allowed_intents
        assert turn.is_terminal is False

    def test_golden_tech_error_flow(self):
        sess = self._create_session()
        # 1. Greet -> Confirm ID
        turn1 = transition_dialogue(sess, intent=CustomerIntent.CONFIRM_ID)
        assert turn1.state == DialogueState.DIAGNOSE
        assert "bank network issue" in turn1.agent_speech_hinglish

        # 2. Diagnose -> Tech Error
        turn2 = transition_dialogue(sess, intent=CustomerIntent.TECH_ERROR)
        assert turn2.state == DialogueState.OFFER_PAYMENT_LINK
        assert "1-click UPI QR link" in turn2.agent_speech_hinglish

        # 3. Accept Offer -> Terminal Success
        turn3 = transition_dialogue(sess, intent=CustomerIntent.ACCEPT_OFFER)
        assert turn3.state == DialogueState.SUCCESS_TERMINAL
        assert turn3.is_terminal is True
        assert turn3.ptp_created is not None
        assert turn3.ptp_created.status == PTPStatus.PTP_ACTIVE

    def test_funds_issue_and_concession_flow(self):
        sess = self._create_session(amount=5000.0, prob=0.40)
        # Advance to Diagnose
        transition_dialogue(sess, intent=CustomerIntent.CONFIRM_ID)

        # Diagnose -> Lack of funds triggers Concession Negotiation
        turn = transition_dialogue(sess, intent=CustomerIntent.LACK_OF_FUNDS)
        assert turn.state == DialogueState.NEGOTIATE_CONCESSION
        assert turn.active_concession is not None
        assert turn.active_concession.waiver_pct == 5.0
        assert "special 5.0% settlement discount" in turn.agent_speech_hinglish

        # Accept concession and promise to pay
        turn_final = transition_dialogue(
            sess,
            intent=CustomerIntent.PROMISE_TO_PAY,
            payload={"promised_date": "2026-09-08T18:00:00Z"},
        )
        assert turn_final.state == DialogueState.SUCCESS_TERMINAL
        assert turn_final.ptp_created.promised_amount_inr == 4750.0 # Concession applied!

    def test_opt_out_stop_flow(self):
        sess = self._create_session()
        turn = transition_dialogue(sess, intent=CustomerIntent.REFUSE_OUTREACH)
        assert turn.state == DialogueState.OPT_OUT_STOP
        assert turn.is_terminal is True
        assert "future recovery communications stop kar diye hain" in turn.agent_speech_hinglish

    def test_dispute_escalate_flow(self):
        sess = self._create_session()
        turn = transition_dialogue(sess, intent=CustomerIntent.DISPUTE_CHARGE)
        assert turn.state == DialogueState.DISPUTE_ESCALATE
        assert turn.is_terminal is True
        assert "senior customer support" in turn.agent_speech_hinglish
