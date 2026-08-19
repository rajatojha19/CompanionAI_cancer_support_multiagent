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
            st.session_state
            .companion
            .start_new_conversation("User")
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    for message in st.session_state.messages:

        # Choose avatar based on message role
        if message["role"] == "assistant":
            avatar = "🎗️"
        else:
            avatar = "🧑"

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):

            # Show agent name for assistant responses
            if (
                message["role"] == "assistant"
                and message.get("agent")
            ):

                st.caption(
                    message["agent"]
                )

            # Display message
            st.write(
                message["content"]
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
    # Get typed message
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

        # Get the agent that handled the request
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