from typing import Optional

from google import genai
from google.genai import types

from config import SYSTEM_SAFETY_PROMPT
from utils.logger import logger


class GeminiClient:
    """Client for optional Gemini API integration."""

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.active = False
        self.client = None

        if not api_key:
            logger.info(
                "GeminiClient: No API key found - running in stub mode."
            )
            return

        try:
            self.client = genai.Client(api_key=api_key)
            self.active = True

            logger.info(
                "GeminiClient: Gemini client initialized successfully."
            )

        except Exception as e:
            logger.warning(
                f"GeminiClient: Failed to initialize Gemini - {e}"
            )
            self.active = False

    def generate(self, user_prompt: str) -> str:

        if not self.active or self.client is None:
            logger.info(
                "GeminiClient: Using stub response (no real LLM)."
            )

            return (
                "[Gemini stub] "
                + user_prompt[:220]
                + " ... (this is a demo stub response without real LLM output)\n\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional."
            )

        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_SAFETY_PROMPT,
                    max_output_tokens=200,
                ),
            )

            text = response.text or ""

            if "⚠️ I am not a doctor" not in text:
                text += (
                    "\n\n⚠️ I am not a doctor and cannot provide medical advice. "
                    "Please consult a qualified medical professional."
                )

            return text

        except Exception as e:
            logger.warning(
                f"GeminiClient: Error while calling Gemini - {e}"
            )

            return (
                "I'm having trouble generating a detailed response right now, "
                "but I'm still here to support you and listen.\n\n"
                "⚠️ I am not a doctor and cannot provide medical advice. "
                "Please consult a qualified medical professional."
            )

    def generate_stream(self, user_prompt: str):
        """Generate a response progressively using Gemini streaming."""

        if not self.active or self.client is None:
            logger.info(
                "GeminiClient: Using stub response (no real LLM)."
            )

            yield (
                "[Gemini stub] "
                + user_prompt[:220]
                + " ... (demo stub response)\n\n"
            )
            return

        try:
            prompt = f"""
{SYSTEM_SAFETY_PROMPT}

User request:

{user_prompt}
"""

            response_stream = self.client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=300,
                ),
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.warning(
                f"GeminiClient: Streaming error - {e}"
            )

            yield (
                "I'm having trouble generating a detailed response right now, "
                "but I'm still here to support you and listen."
            )