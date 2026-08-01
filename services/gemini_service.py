import logging

logger = logging.getLogger("CancerSupportCompanion")

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

⚠️ I am not a doctor and cannot provide medical advice. Please consult a qualified medical professional.
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

        if not self.active or self.model is None:
            logger.info("GeminiClient: Using stub response (no real LLM).")
            return (
                "[Gemini stub] "
                + user_prompt[:220]
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
                "but I'm still here to support you and listen.\n\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional."
            )