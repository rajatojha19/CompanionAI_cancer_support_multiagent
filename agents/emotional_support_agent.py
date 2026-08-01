from typing import Optional
import random

from models.session import UserSession
from services.gemini_service import GeminiClient
from utils.logger import logger

class EmotionalSupportAgent:
    """Primary agent for emotional support and conversation"""

    def __init__(self, llm: Optional[GeminiClient] = None):
        self.llm = llm
        self.safety_disclaimer = (
            "⚠️ I am not a doctor and cannot provide medical advice. "
            "Please consult a qualified medical professional."
        )
        self.emotional_responses = {
            "scared": [
                "It's completely normal to feel scared right now. Many people feel this way when facing health challenges.",
                "I hear the fear in your words. Would you like to talk about what's making you feel most anxious?",
                "Feeling scared is a natural response. Remember to breathe and take things one step at a time.",
            ],
            "sad": [
                "I'm really sorry you're feeling sad. It's okay to have these feelings.",
                "This sounds really difficult. Would it help to talk about what's on your mind?",
                "Your feelings are valid. Many people find it helpful to express their sadness rather than keeping it inside.",
            ],
            "overwhelmed": [
                "It sounds like you're dealing with a lot right now. Let's break this down into smaller pieces.",
                "Feeling overwhelmed is common in situations like this. What's one small thing that might help right now?",
                "Take a moment to breathe. You don't have to solve everything at once.",
            ],
        }

    def detect_emotion(self, message: str) -> str:
        message_lower = message.lower()
        if any(word in message_lower for word in ["scared", "fear", "afraid", "terrified"]):
            return "scared"
        if any(word in message_lower for word in ["sad", "depressed", "hopeless", "crying"]):
            return "sad"
        if any(word in message_lower for word in ["overwhelmed", "too much", "cant handle", "stressed"]):
            return "overwhelmed"
        return "neutral"

    def provide_support(self, user_message: str, session: UserSession) -> str:
        emotion = self.detect_emotion(user_message)
        logger.info(f"EmotionalSupportAgent: Detected emotion '{emotion}'")

        if self.llm and self.llm.active:
            prompt = f"""
User name: {session.user_name or 'Friend'}
Detected emotion: {emotion}
User message: "{user_message}"

Write a warm, supportive reply in 6–10 sentences.
- Acknowledge and normalise the emotion.
- Offer gentle, practical coping suggestions (e.g., breathing, journaling, talking to trusted people).
- Encourage reaching out to the medical team for concerns.
- Do NOT talk about medication, treatments, or diagnosis.
Remember to keep the language simple and human.
"""
            response = self.llm.generate(prompt)
        else:
            if emotion != "neutral":
                import random

                response = random.choice(self.emotional_responses[emotion])
            else:
                response = (
                    "I'm here to listen and support you. Could you tell me more about how you're feeling?"
                )
            response += f"\n\n{self.safety_disclaimer}"

        session.add_message("emotional_agent", response)
        return response