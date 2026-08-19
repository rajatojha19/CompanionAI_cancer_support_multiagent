from typing import Optional
import io
import wave

from google.genai import types

from utils.logger import logger


class SpeechService:
    """Handles speech-to-text and text-to-speech using Gemini."""

    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    # ============================================================
    # Speech-to-Text
    # ============================================================

    def transcribe(self, audio_file) -> Optional[str]:
        """Transcribe recorded speech into text using Gemini."""

        if not audio_file:
            return None

        if not self.gemini_client.active:
            logger.warning(
                "SpeechService: Gemini is not active."
            )
            return None

        try:
            audio_bytes = audio_file.getvalue()

            response = (
                self.gemini_client
                .client
                .models
                .generate_content(
                    model="gemini-3.5-flash",
                    contents=[
                        (
                            "Transcribe the speech exactly as spoken. "
                            "Preserve the user's language. "
                            "If the speaker uses Hindi written in Roman "
                            "letters (Hinglish), keep it in Roman Hindi. "
                            "Return only the transcription, without "
                            "explanation."
                        ),
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav",
                        ),
                    ],
                )
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

    # ============================================================
    # Text-to-Speech
    # ============================================================

    def text_to_speech(
        self,
        text: str
    ) -> Optional[bytes]:
        """Convert assistant text into WAV audio using Gemini TTS."""

        if not text or not text.strip():
            return None

        if not self.gemini_client.active:
            logger.warning(
                "SpeechService: Gemini is not active."
            )
            return None

        try:
            response = (
                self.gemini_client
                .client
                .models
                .generate_content(
                    model="gemini-3.1-flash-tts-preview",
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=(
                                    types.PrebuiltVoiceConfig(
                                        voice_name="Kore"
                                    )
                                )
                            )
                        ),
                    ),
                )
            )

            # ----------------------------------------------------
            # Safely extract generated audio
            # ----------------------------------------------------

            if not response.candidates:
                logger.warning(
                    "SpeechService: TTS returned no candidates."
                )
                return None

            candidate = response.candidates[0]

            if not candidate.content:
                logger.warning(
                    "SpeechService: TTS returned no content."
                )
                return None

            if not candidate.content.parts:
                logger.warning(
                    "SpeechService: TTS returned no parts."
                )
                return None

            audio_part = None

            for part in candidate.content.parts:

                if getattr(part, "inline_data", None):
                    audio_part = part.inline_data
                    break

            if audio_part is None:
                logger.warning(
                    "SpeechService: TTS returned no audio data."
                )
                return None

            audio_data = audio_part.data

            if not audio_data:
                logger.warning(
                    "SpeechService: TTS audio data is empty."
                )
                return None

            # ----------------------------------------------------
            # Gemini returns raw PCM audio.
            #
            # Convert PCM → WAV so Streamlit can play it.
            # Gemini TTS output:
            #   Channels   = 1
            #   Sample rate = 24000 Hz
            #   Sample width = 2 bytes / 16-bit
            # ----------------------------------------------------

            wav_buffer = io.BytesIO()

            with wave.open(wav_buffer, "wb") as wav_file:

                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)

            wav_audio = wav_buffer.getvalue()

            logger.info(
                "SpeechService: Text-to-speech generated successfully."
            )

            return wav_audio

        except Exception as e:
            logger.warning(
                f"SpeechService: Text-to-speech failed - {e}"
            )
            return None