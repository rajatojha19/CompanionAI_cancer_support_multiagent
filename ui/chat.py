"""
CompanionAI - Chat UI

Chat interface for the Cancer Support Companion multi-agent system.

Features:
- CompanionAI header
- Safety notice
- User and assistant avatars
- Text input
- Voice input
- Multi-agent responses
- Read Aloud for assistant responses
"""

import streamlit as st

from core.companion import CancerSupportCompanion


def show_chat_page():
    """Display the CompanionAI chat interface."""

    # ============================================================
    # Initialize session state
    # ============================================================

    if "companion" not in st.session_state:
        st.session_state.companion = CancerSupportCompanion()

    if "session_id" not in st.session_state:
        (
            _,
            st.session_state.session_id,
        ) = (
            st.session_state.companion
            .start_new_conversation("User")
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Store generated audio so it survives Streamlit reruns
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}

    # ============================================================
    # CompanionAI Header
    # ============================================================

    st.title("🎗️ CompanionAI")

    st.subheader(
        "Cancer Support & Education"
    )

    st.caption(
        "For emotional support and general cancer-related "
        "education — not a substitute for professional "
        "medical advice."
    )

    st.divider()

    # ============================================================
    # Conversation History
    # ============================================================

    for message_id, message in enumerate(
        st.session_state.messages
    ):

        # --------------------------------------------------------
        # User Message
        # --------------------------------------------------------

        if message["role"] == "user":

            with st.chat_message(
                "user",
                avatar="🧑",
            ):

                st.write(
                    message["content"]
                )

        # --------------------------------------------------------
        # Assistant Message
        # --------------------------------------------------------

        else:

            with st.chat_message(
                "assistant",
                avatar="🎗️",
            ):

                # Show agent name
                if message.get("agent"):

                    st.caption(
                        message["agent"]
                    )

                # Show response
                st.write(
                    message["content"]
                )

                # ------------------------------------------------
                # Read Aloud
                # ------------------------------------------------

                audio_key = (
                    f"audio_{message_id}"
                )

                if st.button(
                    "🔊 Read aloud",
                    key=f"read_aloud_{message_id}",
                ):

                    with st.spinner(
                        "🔊 Generating audio..."
                    ):

                        audio = (
                            st.session_state
                            .companion
                            .speech_service
                            .text_to_speech(
                                message["content"]
                            )
                        )

                    if audio:

                        st.session_state.audio_cache[
                            audio_key
                        ] = audio

                    else:

                        st.warning(
                            "Sorry, I couldn't generate audio."
                        )

                # ------------------------------------------------
                # Keep audio visible after rerun
                # ------------------------------------------------

                if (
                    audio_key
                    in st.session_state.audio_cache
                ):

                    st.audio(
                        st.session_state.audio_cache[
                            audio_key
                        ],
                        format="audio/wav",
                    )

    # ============================================================
    # Text + Voice Input
    # ============================================================

    chat_input = st.chat_input(
        "Type or record your message...",
        accept_audio=True,
        audio_sample_rate=16000,
    )

    if not chat_input:
        return

    # ============================================================
    # Get Typed Message
    # ============================================================

    user_message = chat_input.text

    # ============================================================
    # Voice Input
    # ============================================================

    if (
        not user_message
        and chat_input.audio
    ):

        with st.spinner(
            "🎙️ Transcribing your voice..."
        ):

            user_message = (
                st.session_state
                .companion
                .speech_service
                .transcribe(
                    chat_input.audio
                )
            )

        if not user_message:

            st.error(
                "Sorry, I couldn't understand the recording. "
                "Please try again."
            )

            return

    # ============================================================
    # Validate Message
    # ============================================================

    if not user_message:
        return

    # ============================================================
    # Save User Message
    # ============================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # ============================================================
    # Display User Message
    # ============================================================

    with st.chat_message(
        "user",
        avatar="🧑",
    ):

        st.write(
            user_message
        )

    # ============================================================
    # Generate Assistant Response
    # ============================================================

    with st.chat_message(
        "assistant",
        avatar="🎗️",
    ):

        with st.spinner(
            "Thinking..."
        ):

            assistant_response = (
                st.session_state
                .companion
                .process_message(
                    st.session_state.session_id,
                    user_message,
                )
            )

        # Get agent that handled the request
        agent_name = (
            st.session_state
            .companion
            .get_last_agent()
        )

        # Display agent name
        st.caption(
            agent_name
        )

        # Display response
        st.write(
            assistant_response
        )
        
    # --------------------------------------------------------
    # Read Aloud for the newly generated response
    # --------------------------------------------------------

    audio_key = (
        f"audio_new_{len(st.session_state.messages)}"
    )

    if st.button(
        "🔊 Read aloud",
        key=f"read_aloud_new_{len(st.session_state.messages)}",
    ):

        with st.spinner(
            "🔊 Generating audio..."
        ):

            audio = (
                st.session_state
                .companion
                .speech_service
                .text_to_speech(
                    assistant_response
                )
            )

        if audio:

            st.session_state.audio_cache[
                audio_key
            ] = audio

        else:

            st.warning(
                "Sorry, I couldn't generate audio."
            )

    # Show cached audio after rerun
    if audio_key in st.session_state.audio_cache:

        st.audio(
            st.session_state.audio_cache[
                audio_key
            ],
            format="audio/wav",
        )

    # ============================================================
    # Save Assistant Response
    # ============================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
            "agent": agent_name,
        }
    )

