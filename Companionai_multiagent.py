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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CancerSupportCompanion")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_SAFETY_PROMPT = """
You are CompanionAI, a SAFE and EMPATHETIC assistant for people affected by cancer.
You MUST follow these rules:

- Do NOT diagnose.
- Do NOT recommend or compare treatments.
- Do NOT suggest drugs, dosages, or medical procedures.
- Do NOT interpret test results or symptoms.
- Do NOT predict survival, remission, or outcomes.

You MAY:
- Offer emotional support in a warm, human tone.
- Explain cancer-related concepts in simple, general terms.
- Help users prepare questions for their medical team.
- Encourage users to talk to doctors, nurses, and counsellors.

Always stay gentle, non-judgmental, and cautious.
Every reply MUST end with this exact sentence:

"⚠️ I am not a doctor and cannot provide medical advice. Please consult a qualified medical professional."
""".strip()


class GeminiClient:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.active = False
        self.model = None

        if not api_key:
            logger.info("GeminiClient: No API key found – running in stub mode.")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                system_instruction=SYSTEM_SAFETY_PROMPT,
            )
            self.active = True
            logger.info("GeminiClient: Gemini model initialised successfully.")
        except Exception as e:
            logger.warning(f"GeminiClient: Failed to initialise Gemini – {e}")
            self.active = False

    def generate(self, user_prompt: str) -> str:
        """
        Generate a response using Gemini.
        If Gemini isn't available, return a stub response.
        """
        if not self.active or self.model is None:
            logger.info("GeminiClient: Using stub response (no real LLM).")
            return (
                "[Gemini stub] " + user_prompt[:220]
                + " ... (this is a demo stub response without real LLM output)\n\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional."
            )

        try:
            response = self.model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_output_tokens": 600,
                },
            )
            text = response.text or ""
            if "⚠️ I am not a doctor" not in text:
                text += (
                    "\n\n⚠️ I am not a doctor and cannot provide medical advice. "
                    "Please consult a qualified medical professional."
                )
            return text
        except Exception as e:
            logger.warning(f"GeminiClient: Error while calling Gemini – {e}")
            return (
                "I'm having trouble generating a detailed response right now, "
                "but I’m still here to support you and listen.\n\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional."
            )


@dataclass
class UserSession:
    """Session management for user conversations"""

    session_id: str
    user_name: str
    created_at: datetime
    conversation_history: List[Dict]
    user_preferences: Dict

    def add_message(self, role: str, content: str):
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
            }
        )
        logger.info(f"Session {self.session_id}: Added {role} message")


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


class QuestionOrganizerAgent:
    """Agent that helps organize questions for medical teams"""

    def __init__(self):
        self.question_categories = {
            "treatment": "Questions about treatment options",
            "symptoms": "Questions about symptoms and side effects",
            "lifestyle": "Questions about daily life and activities",
            "follow_up": "Questions about next steps and monitoring",
        }

    def extract_concerns(self, user_message: str) -> List[str]:
        concerns: List[str] = []
        patterns = {
            "treatment": r"(treatment|therapy|medication|chemo|chemotherapy|radiation)",
            "symptoms": r"(pain|tired|fatigue|nausea|sleep|appetite|vomit)",
            "lifestyle": r"(work|exercise|diet|food|family|daily|routine)",
            "follow_up": r"(next|follow|appointment|test|scan|results)",
        }

        for category, pattern in patterns.items():
            if re.search(pattern, user_message.lower()):
                concerns.append(category)

        logger.info(f"QuestionOrganizerAgent: Extracted concerns {concerns}")
        return concerns

    def generate_questions(self, concerns: List[str]) -> str:
        question_templates = {
            "treatment": [
                "What are the goals of this treatment?",
                "What are the potential side effects I should watch for?",
                "How will we know if the treatment is working?",
                "Are there alternative treatment options we should consider?",
            ],
            "symptoms": [
                "Is this symptom something I should be concerned about?",
                "What can I do to manage this symptom at home?",
                "When should I contact you about this symptom?",
                "Could this symptom be related to my treatment?",
            ],
            "lifestyle": [
                "What kind of daily activities are safe for me right now?",
                "Are there dietary changes that might help me?",
                "How can I manage my energy levels throughout the day?",
                "What support is available for my family and caregivers?",
            ],
            "follow_up": [
                "When is our next appointment?",
                "What tests will we do next?",
                "What changes should I report before our next visit?",
                "Who should I contact if I have questions between appointments?",
            ],
        }

        questions: List[str] = []
        for concern in concerns:
            if concern in question_templates:
                questions.extend(question_templates[concern][:2])

        if not questions:
            questions = [
                "Can you explain my diagnosis in terms I can understand?",
                "What are the next steps in my care plan?",
                "Who is the best person to contact with questions?",
                "What resources are available to help me cope emotionally?",
            ]

        response_lines = [
            "Based on what you've shared, here are some questions you might want to ask your medical team:",
            "",
        ]
        for i, q in enumerate(questions, 1):
            response_lines.append(f"{i}. {q}")

        response_lines.append(
            "\nRemember to write down your questions before appointments.\n\n"
            "⚠️ I am not a doctor and cannot provide medical advice. Please consult a qualified medical professional."
        )

        response = "\n".join(response_lines)
        logger.info(f"QuestionOrganizerAgent: Generated {len(questions)} questions")
        return response


class EducationalAgent:
    """Provides general educational information about cancer concepts"""

    def __init__(self, llm: Optional[GeminiClient] = None):
        self.llm = llm
        self.safety_disclaimer = (
            "⚠️ I am not a doctor and cannot provide medical advice. "
            "Please consult a qualified medical professional."
        )
        self.educational_topics = {
            "chemotherapy": "Chemotherapy uses medications to treat cancer. These medications work by targeting rapidly dividing cells.",
            "radiation": "Radiation therapy uses high-energy beams to target and damage cancer cells in specific areas.",
            "biopsy": "A biopsy is a procedure where a small sample of tissue is taken for examination under a microscope.",
            "remission": "Remission means there is no evidence of cancer after treatment. It can be partial or complete.",
            "side_effects": "Side effects are unintended effects of treatment that can vary from person to person.",
            "support_care": "Supportive care focuses on managing symptoms and improving quality of life during treatment.",
        }

    def explain_concept(self, concept: str) -> str:
        concept_lower = concept.lower()

        if self.llm and self.llm.active:
            prompt = f"""
User asked: "{concept}"

Explain the concept in clear, simple language
for a non-medical person. Stay high-level and general.
Do NOT provide instructions, dosages, or treatment decisions.
"""
            return self.llm.generate(prompt)

        for topic, explanation in self.educational_topics.items():
            if topic in concept_lower:
                logger.info(f"EducationalAgent: Explained concept '{topic}'")
                return (
                    f"Here's some general information about {topic}:\n\n"
                    f"{explanation}\n\n{self.safety_disclaimer}"
                )

        return (
            "I can provide general information about common cancer-related topics "
            "like chemotherapy, radiation, biopsies, and more. "
            "What specific concept would you like me to explain?\n\n"
            f"{self.safety_disclaimer}"
        )


class SessionManager:
    """Manages user sessions and memory"""

    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        logger.info("SessionManager: Initialized session management")

    def create_session(self, user_name: str) -> UserSession:
        session_id = f"session_{len(self.sessions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = UserSession(
            session_id=session_id,
            user_name=user_name,
            created_at=datetime.now(),
            conversation_history=[],
            user_preferences={},
        )
        self.sessions[session_id] = session
        logger.info(f"SessionManager: Created new session {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        return self.sessions.get(session_id)

    def save_session_state(self, session: UserSession):
        logger.info(f"SessionManager: Saved state for session {session.session_id}")


class CancerSupportCompanion:
    """Main multi-agent system coordinating all components"""

    def __init__(self, llm: Optional[GeminiClient] = None):
        self.session_manager = SessionManager()
        self.emotional_agent = EmotionalSupportAgent(llm)
        self.question_agent = QuestionOrganizerAgent()
        self.educational_agent = EducationalAgent(llm)

        self.metrics = {
            "sessions_created": 0,
            "messages_processed": 0,
            "emotional_support_given": 0,
            "questions_generated": 0,
            "concepts_explained": 0,
        }

        logger.info("CancerSupportCompanion: Multi-agent system initialized")

    def start_new_conversation(self, user_name: str):
        session = self.session_manager.create_session(user_name)
        self.metrics["sessions_created"] += 1

        welcome_message = f"""Hello {user_name}, I'm your Cancer Support Companion. I'm here to:

• Provide emotional support when you need someone to talk to
• Help you organize questions for your medical team
• Explain general cancer-related concepts in simple terms

You can type 'quit' at any time to end the conversation.

How are you feeling today?"""

        session.add_message("system", welcome_message)
        return welcome_message, session.session_id

    def process_message(self, session_id: str, user_message: str) -> str:
        session = self.session_manager.get_session(session_id)
        if not session:
            return "I'm sorry, I couldn't find your conversation. Let's start over."

        session.add_message("user", user_message)
        self.metrics["messages_processed"] += 1

        user_message_lower = user_message.lower()

        if any(word in user_message_lower for word in ["feel", "scared", "sad", "worried", "anxious", "overwhelmed"]):
            self.metrics["emotional_support_given"] += 1
            response = self.emotional_agent.provide_support(user_message, session)

        elif any(word in user_message_lower for word in ["ask", "question", "doctor", "appointment", "what to say"]):
            concerns = self.question_agent.extract_concerns(user_message)
            self.metrics["questions_generated"] += 1
            response = self.question_agent.generate_questions(concerns)

        elif any(
            phrase in user_message_lower
            for phrase in ["what is", "explain", "mean", "tell me about"]
        ):
            self.metrics["concepts_explained"] += 1
            response = self.educational_agent.explain_concept(user_message)

        else:
            self.metrics["emotional_support_given"] += 1
            response = self.emotional_agent.provide_support(user_message, session)

        self.session_manager.save_session_state(session)
        logger.info("CancerSupportCompanion: Processed message through multi-agent system")
        return response

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        session = self.session_manager.get_session(session_id)
        if session:
            return session.conversation_history
        return []

    def get_metrics(self) -> Dict:
        return self.metrics.copy()


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
