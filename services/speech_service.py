from typing import Optional

from google.genai import types

from utils.logger import logger


class SpeechService:
    """Converts recorded speech into text using Gemini."""

    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def transcribe(self, audio_file) -> Optional[str]:
        """Transcribe a Streamlit recorded audio file."""

        if not audio_file:
            return None

        if not self.gemini_client.active:
            logger.warning(
                "SpeechService: Gemini is not active."
            )
            return None

        try:
            audio_bytes = audio_file.getvalue()

            response = self.gemini_client.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    (
                        "Transcribe the speech exactly as spoken. "
                        "Preserve the user's language. "
                        "If the speaker uses Hindi written in Roman letters "
                        "(Hinglish), keep it in Roman Hindi. "
                        "Return only the transcription, without explanation."
                    ),
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type="audio/wav",
                    ),
                ],
            )

            text = (response.text or "").strip()

            if text:
                logger.info(
                    "SpeechService: Audio transcribed successfully."
                )
                return text

            logger.warning(
                "SpeechService: Gemini returned an empty transcription."
            )
            return None

        except Exception as e:
            logger.warning(
                f"SpeechService: Transcription failed - {e}"
            )
            return None