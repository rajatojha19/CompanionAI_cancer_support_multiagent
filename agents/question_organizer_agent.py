import re

from typing import List, Optional
from utils.logger import logger

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