from typing import Optional
from utils.logger import logger
from services.gemini_service import GeminiClient


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

    Answer the user's question directly.

    RESPONSE LENGTH:
    - Give a concise but complete answer.
- For a simple question, use 2-3 short sentences.
- For a normal question, use 3-4 short sentences.
- If the user explicitly asks for detailed information, use at most 5-6 sentences.
- Do not repeat information unnecessarily.
- Do not add emotional-support advice unless the user asks for it.

LANGUAGE:
- Reply in the same language as the user.
- English question → English.
- Hindi question → Hindi.
- Hinglish/Roman Hindi question → natural Hinglish/Roman Hindi.
- Do not unnecessarily switch languages.

MEDICAL SAFETY:
- Provide only general educational information.
- Do not diagnose the user.
- Do not interpret their symptoms, reports, scans, or test results.
- Do not recommend, compare, or select treatments.
- Do not suggest medicines, dosages, or medical procedures.
- Do not provide step-by-step instructions for medical procedures.
- Do not make predictions about survival, remission, or outcomes.
- Encourage the user to consult their healthcare professional when appropriate.

STYLE:
- Be clear, natural, and easy for a non-medical person to understand.
- Answer the actual question first.
- Avoid unnecessary introductions such as "Hello" or "It is completely normal..."
- Do not turn a simple educational question into an emotional-support response.
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
