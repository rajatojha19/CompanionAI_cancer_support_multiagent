"""
Cancer Support Companion Agent
A multi-agent system providing emotional support and educational assistance
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import textwrap

from services.gemini_service import GeminiClient
from models.session import UserSession
from agents.emotional_support_agent import EmotionalSupportAgent
from agents.educational_agent import EducationalAgent
from agents.question_organizer_agent import QuestionOrganizerAgent
from services.session_manager import SessionManager
from core.companion import CancerSupportCompanion


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("CancerSupportCompanion")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class QuestionOrganizationTool:
    """Custom tool for organizing medical questions"""

    def __init__(self, question_agent: QuestionOrganizerAgent):
        self.question_agent = question_agent

    def organize_medical_questions(self, user_concerns: str) -> str:
        concerns = self.question_agent.extract_concerns(user_concerns)
        return self.question_agent.generate_questions(concerns)


def demo_agent():
    print("=== Cancer Support Companion Demo ===\n")

    llm = GeminiClient(GEMINI_API_KEY)
    companion = CancerSupportCompanion(llm)

    welcome, session_id = companion.start_new_conversation("Alex")
    print("Agent:", welcome, "\n")

    print("User: I'm feeling really scared about my upcoming treatment")
    resp = companion.process_message(session_id, "I'm feeling really scared about my upcoming treatment")
    print("\nAgent:\n", textwrap.fill(resp, width=90), "\n")

    print("User: I have an appointment tomorrow, what should I ask my doctor?")
    resp = companion.process_message(session_id, "I have an appointment tomorrow, what should I ask my doctor?")
    print("\nAgent:\n", textwrap.fill(resp, width=90), "\n")

    print("User: Can you explain what chemotherapy is?")
    resp = companion.process_message(session_id, "Can you explain what chemotherapy is?")
    print("\nAgent:\n", textwrap.fill(resp, width=90), "\n")

    print("=== System Metrics ===")
    print(json.dumps(companion.get_metrics(), indent=2))

    history = companion.get_conversation_history(session_id)
    print(f"\n=== Session Info ===\nConversation has {len(history)} messages")


def interactive_chat():
    print("=== Cancer Support Companion – Interactive Chat ===")
    print("This tool offers emotional support and education, not medical advice.")
    print("Type 'quit' to end the conversation.\n")

    name = input("Before we start, what name would you like me to use for you? (press Enter to skip)\n> ").strip()
    if not name:
        name = "Friend"

    llm = GeminiClient(GEMINI_API_KEY)
    companion = CancerSupportCompanion(llm)
    welcome, session_id = companion.start_new_conversation(name)
    print("\nAgent:\n", textwrap.fill(welcome, width=90), "\n")

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"quit", "exit"}:
            print(
                "\nAgent:\nThank you for talking with me today. "
                "I hope you feel a little more supported.\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional.\n"
            )
            break

        if not user_msg:
            continue

        resp = companion.process_message(session_id, user_msg)
        print("\nAgent:\n", textwrap.fill(resp, width=90), "\n")


if __name__ == "__main__":
    demo_agent()
    print("\n\nNow entering interactive mode...\n")
    interactive_chat()
