from typing import Optional

from models.session import UserSession
from services.gemini_service import GeminiClient
from utils.logger import logger


class EmotionalSupportAgent:
    """Primary agent for emotional support and supportive conversation."""

    def __init__(
        self,
        llm: Optional[GeminiClient] = None,
    ):
        self.llm = llm

        self.safety_disclaimer = (
            "⚠️ I am not a doctor and cannot provide medical advice. "
            "Please consult a qualified medical professional."
        )

        self.emotional_responses = {
            "scared": [
                (
                    "Mujhe samajh aa raha hai ki ye situation scary lag sakti hai. "
                    "Abhi ek time par ek step lene ki koshish karo."
                ),
                (
                    "Darr feel karna aise situation mein natural hai. "
                    "Agar tum chaho to bata sakte ho ki sabse zyada kis baat ki tension hai."
                ),
            ],
            "sad": [
                (
                    "Mujhe afsos hai ki tum abhi sad feel kar rahe ho. "
                    "Apni feelings kisi trusted friend ya family member ke saath share karna helpful ho sakta hai."
                ),
                (
                    "Sad ya low feel karna difficult ho sakta hai. "
                    "Tumhe sab kuch ek saath handle karne ki zarurat nahi hai."
                ),
            ],
            "overwhelmed": [
                (
                    "Lagta hai abhi bahut saari cheezein ek saath chal rahi hain. "
                    "Sab kuch ek saath solve karne ke bajay ek chhoti cheez par focus karne ki koshish karo."
                ),
                (
                    "Overwhelmed feel hona understandable hai. "
                    "Thoda break lo, dheere-dheere breathe karo aur jo sabse important hai usse ek-ek karke handle karo."
                ),
            ],
            "anxious": [
                (
                    "Anxiety ke time par thoughts bahut overwhelming ho sakte hain. "
                    "Kuch slow, deep breaths lene aur kisi trusted person se baat karne ki koshish karo."
                ),
                (
                    "Agar tum anxious feel kar rahe ho, to abhi sirf next small step par focus karna helpful ho sakta hai."
                ),
            ],
        }

    # ============================================================
    # Emotion Detection
    # ============================================================

    def detect_emotion(
        self,
        message: str,
    ) -> str:
        """Detect the main emotion expressed by the user."""

        message_lower = message.lower()

        if any(
            word in message_lower
            for word in [
                "scared",
                "fear",
                "afraid",
                "terrified",
                "dar",
                "darr",
                "darna",
                "ghabra",
                "ghabrahat",
            ]
        ):
            return "scared"

        if any(
            word in message_lower
            for word in [
                "sad",
                "depressed",
                "hopeless",
                "crying",
                "dukhi",
                "udaas",
                "rona",
                "ro raha",
                "ro rahi",
            ]
        ):
            return "sad"

        if any(
            word in message_lower
            for word in [
                "overwhelmed",
                "too much",
                "can't handle",
                "cant handle",
                "stressed",
                "stress",
                "pareshan",
                "pareshaan",
            ]
        ):
            return "overwhelmed"

        if any(
            word in message_lower
            for word in [
                "anxious",
                "anxiety",
                "nervous",
                "tension",
                "anxious hoon",
                "tension ho rahi",
            ]
        ):
            return "anxious"

        return "neutral"

    # ============================================================
    # Language Detection
    # ============================================================

    def is_hinglish(
        self,
        message: str,
    ) -> bool:
        """Detect common Roman Hindi / Hinglish expressions."""

        message_lower = message.lower()

        hinglish_words = [
            "mujhe",
            "mera",
            "meri",
            "mere",
            "mujhko",
            "mujhse",
            "hai",
            "hoon",
            "hota",
            "hoti",
            "lag",
            "lagta",
            "lagti",
            "bahut",
            "zyada",
            "dar",
            "darr",
            "tension",
            "pareshan",
            "pareshaan",
            "ghabra",
            "ghabrahat",
            "kya",
            "kaise",
            "kyun",
            "kyu",
            "nahi",
            "nahin",
            "raha",
            "rahi",
            "sakta",
            "sakti",
            "chahta",
            "chahti",
        ]

        return any(
            word in message_lower.split()
            for word in hinglish_words
        )

    # ============================================================
    # Provide Support
    # ============================================================

    def provide_support(
        self,
        user_message: str,
        session: UserSession,
    ) -> str:
        """Generate a concise emotional-support response."""

        emotion = self.detect_emotion(user_message)

        logger.info(
            f"EmotionalSupportAgent: Detected emotion '{emotion}'"
        )

        # --------------------------------------------------------
        # Gemini response
        # --------------------------------------------------------

        if self.llm and self.llm.active:

            if self.is_hinglish(user_message):
                language_instruction = """
Respond in natural Hinglish using Roman/English letters.

Do NOT use Devanagari Hindi.

Example:
"Mujhe samajh aa raha hai ki ye situation difficult lag sakti hai.
Tumhe sab kuch ek saath handle karne ki zarurat nahi hai."
"""
            else:
                language_instruction = """
Respond in natural English.

If the user clearly writes in Hindi Devanagari,
respond in Hindi Devanagari.
"""

            prompt = f"""
You are the Emotional Support Agent of CompanionAI.

User name:
{session.user_name or "User"}

Detected emotion:
{emotion}

User message:
"{user_message}"

LANGUAGE:
{language_instruction}

RESPONSE LENGTH:
- Maximum 3-4 short sentences.
- Prefer 2-3 sentences.
- Keep the response concise.
- Do NOT write a long motivational paragraph.
- Do NOT repeat the user's question.
- Do NOT start with "Hello".
- Do NOT say "It is completely understandable" repeatedly.

SUPPORT STYLE:
- Acknowledge the user's feelings.
- Be warm, calm and respectful.
- Give at most one simple coping suggestion.
- Encourage talking to a trusted person when appropriate.
- If the concern is medical, remind them that their healthcare
  team is the right source for medical guidance.

DO NOT:
- Diagnose the user.
- Give medical advice.
- Recommend medicines or treatments.
- Interpret medical reports.
- Predict medical outcomes.
- Pretend to be a doctor or therapist.

Return only a concise supportive response.

After the response, add:

⚠️ I am not a doctor and cannot provide medical advice.
Please consult a qualified medical professional.
"""

            response = self.llm.generate(prompt)

            if response:
                session.add_message(
                    "emotional_agent",
                    response,
                )
                return response

        # --------------------------------------------------------
        # Local fallback
        # --------------------------------------------------------

        if emotion in self.emotional_responses:

            import random

            response = random.choice(
                self.emotional_responses[emotion]
            )

        else:

            if self.is_hinglish(user_message):
                response = (
                    "Main yahan tumhari baat sunne ke liye hoon. "
                    "Agar tum comfortable ho, to batao ki abhi "
                    "sabse zyada kis baat ki tension ho rahi hai."
                )

            else:
                response = (
                    "I'm here to listen and support you. "
                    "If you're comfortable, tell me what is "
                    "worrying you the most right now."
                )

        response += f"\n\n{self.safety_disclaimer}"

        session.add_message(
            "emotional_agent",
            response,
        )

        return response