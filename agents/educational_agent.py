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
