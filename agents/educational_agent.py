from typing import Optional

from utils.logger import logger
from services.gemini_service import GeminiClient


class EducationalAgent:
    """Provides general educational information about cancer concepts."""

    def __init__(
        self,
        llm: Optional[GeminiClient] = None,
    ):
        self.llm = llm

        self.safety_disclaimer = (
            "⚠️ I am not a doctor and cannot provide medical advice. "
            "Please consult a qualified medical professional."
        )

        self.educational_topics = {
            "chemotherapy": (
                "Chemotherapy uses medications to treat cancer. "
                "These medications work by targeting rapidly dividing cells."
            ),
            "radiation": (
                "Radiation therapy uses high-energy beams to target "
                "and damage cancer cells in specific areas."
            ),
            "biopsy": (
                "A biopsy is a procedure where a small sample of "
                "tissue is taken for examination under a microscope."
            ),
            "remission": (
                "Remission means there is no evidence of cancer "
                "after treatment. It can be partial or complete."
            ),
            "side_effects": (
                "Side effects are unintended effects of treatment "
                "that can vary from person to person."
            ),
            "support_care": (
                "Supportive care focuses on managing symptoms "
                "and improving quality of life during treatment."
            ),
        }

    def explain_concept(
        self,
        concept: str,
    ) -> str:
        """Explain a cancer-related concept in the user's language."""

        concept_lower = concept.lower()

        # ========================================================
        # Gemini-powered response
        # ========================================================

        if self.llm and self.llm.active:

            prompt = f"""
You are the Educational Agent of CompanionAI.

User question:
"{concept}"

Your task is to answer the user's medical education question
in a SHORT, DIRECT and EASY-TO-UNDERSTAND way.

LANGUAGE:
- English → English
- Hindi → Hindi
- Hinglish / Roman Hindi → Hinglish / Roman Hindi
- Never convert Roman Hindi into English.
- Never convert Roman Hindi into Devanagari Hindi.

Examples:

User:
"biopsy kya hoti hai?"

Answer:
"Biopsy ek medical procedure hai jisme doctor body ke kisi
area se tissue ya cells ka chhota sample lete hain. Is sample
ko laboratory mein examine kiya jata hai taaki doctors
condition ko better samajh saken."

User:
"what is a biopsy?"

Answer:
"A biopsy is a medical procedure in which a small sample of
tissue or cells is taken from the body and examined in a
laboratory."

STRICT LENGTH:
- Maximum 3 sentences.
- Prefer 2 sentences.
- Maximum 60 words.
- Do NOT give emotional-support advice unless specifically asked.
- Do NOT start with "Hello".
- Do NOT say "It is completely understandable".
- Do NOT repeat the question.
- Answer the question immediately.

MEDICAL SAFETY:
- Give only general educational information.
- Do not diagnose.
- Do not interpret reports or scans.
- Do not recommend medicines or treatments.
- Do not provide personalized medical advice.

At the end, add this exact disclaimer:

"⚠️ I am not a doctor and cannot provide medical advice.
Please consult a qualified medical professional."

Return only the answer and disclaimer.
"""

            response = self.llm.generate(prompt)

            if response:
                return response

        # ========================================================
        # Local fallback
        # ========================================================

        for topic, explanation in self.educational_topics.items():

            if topic in concept_lower:

                logger.info(
                    f"EducationalAgent: Explained concept '{topic}'"
                )

                return (
                    f"Here's some general information about "
                    f"{topic}:\n\n"
                    f"{explanation}\n\n"
                    f"{self.safety_disclaimer}"
                )

        return (
            "I can provide general information about common "
            "cancer-related topics like chemotherapy, radiation, "
            "biopsies, and more. What specific concept would "
            "you like me to explain?\n\n"
            f"{self.safety_disclaimer}"
        )