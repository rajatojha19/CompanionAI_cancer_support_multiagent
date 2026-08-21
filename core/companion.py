from typing import Optional, List, Dict

from config import GEMINI_API_KEY
from agents.emotional_support_agent import EmotionalSupportAgent
from agents.educational_agent import EducationalAgent
from agents.question_organizer_agent import QuestionOrganizerAgent
from services.session_manager import SessionManager
from services.gemini_service import GeminiClient
from services.speech_service import SpeechService
from utils.logger import logger


class CancerSupportCompanion:
    """Main multi-agent system coordinating all components."""

    def __init__(self, llm: Optional[GeminiClient] = None):

        if llm is None:
            llm = GeminiClient(GEMINI_API_KEY)

        self.session_manager = SessionManager()

        self.emotional_agent = EmotionalSupportAgent(llm)

        self.question_agent = QuestionOrganizerAgent()

        self.educational_agent = EducationalAgent(llm)

        self.speech_service = SpeechService(llm)

        # Stores which agent handled the latest message
        self.last_agent = None

        self.metrics = {
            "sessions_created": 0,
            "messages_processed": 0,
            "emotional_support_given": 0,
            "questions_generated": 0,
            "concepts_explained": 0,
        }

        logger.info(
            "CancerSupportCompanion: Multi-agent system initialized"
        )

    # ============================================================
    # Start New Conversation
    # ============================================================

    def start_new_conversation(self, user_name: str):

        session = self.session_manager.create_session(user_name)

        self.metrics["sessions_created"] += 1

        welcome_message = f"""
Hello {user_name}, I'm your Cancer Support Companion. I'm here to:

• Provide emotional support when you need someone to talk to
• Help you organize questions for your medical team
• Explain general cancer-related concepts in simple terms

You can type 'quit' at any time to end the conversation.

How are you feeling today?
"""

        session.add_message(
            "system",
            welcome_message
        )

        return (
            welcome_message,
            session.session_id
        )

    # ============================================================
    # Process User Message
    # ============================================================

    def process_message(
        self,
        session_id: str,
        user_message: str,
    ) -> str:
        """Route the user's message to the appropriate agent."""

        session = self.session_manager.get_session(session_id)

        if not session:
            return (
                "I'm sorry, I couldn't find your conversation. "
                "Let's start over."
            )

        # Save user message
        session.add_message("user", user_message)
        self.metrics["messages_processed"] += 1

        user_message_lower = user_message.lower().strip()

        # ========================================================
        # Educational phrases
        # ========================================================

        educational_phrases = [
            # English
            "what is",
            "what are",
            "what does",
            "explain",
            "tell me about",
            "how does",
            "how is",
            "how are",
            "what happens",
            "what causes",
            "why does",
            "why is",
            "meaning of",

            # Hinglish / Roman Hindi
            "kya hai",
            "kya hota",
            "kya hoti",
            "kya hote",
            "kaise",
            "kaise hota",
            "kaise hoti",
            "kaise hote",
            "kyun",
            "kyu",
            "samjhao",
            "samjha do",
            "samjha",
            "batao",
            "bata do",
            "btao",
            "bta do",
            "iske baare mein",
            "iske bare mein",
            "short mein",
            "short me",
            "brief mein",
            "brief me",
        ]

        # ========================================================
        # Common medical terms
        # ========================================================

        medical_terms = [
            "biopsy",
            "biospy",
            "biopsi",
            "cancer",
            "chemotherapy",
            "chemo",
            "radiation",
            "remission",
            "tumor",
            "tumour",
            "metastasis",
            "metastatic",
            "cancer cell",
            "cancer cells",
            "treatment",
            "side effect",
            "side effects",
            "scan",
            "mri",
            "ct scan",
            "pet scan",
            "pathology",
            "diagnosis",
        ]

        # ========================================================
        # Determine educational intent
        # ========================================================

        is_educational = any(
            phrase in user_message_lower
            for phrase in educational_phrases
        )

        mentions_medical_topic = any(
            term in user_message_lower
            for term in medical_terms
        )

        # ========================================================
        # Educational Agent
        # ========================================================

        if is_educational or (
            mentions_medical_topic
            and any(
                word in user_message_lower
                for word in [
                    "tell",
                    "explain",
                    "about",
                    "samj",
                    "bata",
                    "meaning",
                    "kya",
                    "kaise",
                    "what",
                    "how",
                ]
            )
        ):
            self.last_agent = "Educational Agent"

            self.metrics["concepts_explained"] += 1

            response = self.educational_agent.explain_concept(
                user_message
            )

        # ========================================================
        # Question Organizer Agent
        # ========================================================

        elif any(
            word in user_message_lower
            for word in [
                "ask",
                "question",
                "doctor",
                "appointment",
                "what to say",
            ]
        ):
            self.last_agent = "Question Organizer Agent"

            concerns = self.question_agent.extract_concerns(
                user_message
            )

            self.metrics["questions_generated"] += 1

            response = self.question_agent.generate_questions(
                concerns,
                user_message,
            )

        # ========================================================
        # Emotional Support Agent
        # ========================================================

        elif any(
            word in user_message_lower
            for word in [
                "feel",
                "scared",
                "sad",
                "worried",
                "anxious",
                "overwhelmed",
                "afraid",
                "stress",
                "stressed",
                "tension",
                "dar",
                "darr",
                "pareshan",
                "pareshaan",
                "ghabra",
                "ghabrahat",
                "dukhi",
                "udaas",
            ]
        ):
            self.last_agent = "Emotional Support Agent"

            self.metrics["emotional_support_given"] += 1

            response = self.emotional_agent.provide_support(
                user_message,
                session,
            )

        # ========================================================
        # Default → Educational Agent
        # ========================================================

        else:
            self.last_agent = "Educational Agent"

            self.metrics["concepts_explained"] += 1

            response = self.educational_agent.explain_concept(
                user_message
            )

        # ========================================================
        # Save session
        # ========================================================

        self.session_manager.save_session_state(session)

        logger.info(
            "CancerSupportCompanion: "
            f"Processed message through {self.last_agent}"
        )

        return response
    # ============================================================
    # Get Last Agent
    # ============================================================

    def get_last_agent(self) -> str:

        """Return the agent that handled the latest message."""

        return self.last_agent or "CompanionAI"

    # ============================================================
    # Get Conversation History
    # ============================================================

    def get_conversation_history(
        self,
        session_id: str,
    ) -> List[Dict]:

        session = self.session_manager.get_session(
            session_id
        )

        if session:
            return session.conversation_history

        return []

    # ============================================================
    # Get Metrics
    # ============================================================

    def get_metrics(self) -> Dict:

        """Return a copy of the current system metrics."""

        return self.metrics.copy()