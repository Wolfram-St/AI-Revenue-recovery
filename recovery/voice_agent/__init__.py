"""Hinglish Voice and WhatsApp conversational recovery agent package.

Provides a finite state dialogue automaton and dynamic concession matrix for
interactive payment recovery.
"""

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

__all__ = [
    "DialogueState",
    "CustomerIntent",
    "ConcessionType",
    "ConcessionOffer",
    "DialogueTurn",
    "DialogueSession",
    "calculate_concession_matrix",
    "transition_dialogue",
]
