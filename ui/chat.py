import streamlit as st

from core.companion import CancerSupportCompanion


# Custom CSS
st.markdown(
    """
    <style>

    .companion-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.08);
    }

    .companion-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .companion-logo {
        font-size: 38px;
        line-height: 1;
    }

    .companion-title {
        font-size: 28px;
        font-weight: 700;
        line-height: 1.1;
    }

    .companion-subtitle {
        font-size: 14px;
        opacity: 0.7;
        margin-top: 4px;
    }

    .online-status {
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: 14px;
        opacity: 0.8;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
    }

    .safety-banner {
        padding: 10px 14px;
        margin-bottom: 20px;
        border-radius: 10px;
        font-size: 13px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: rgba(128, 128, 128, 0.06);
    }

    .agent-badge {
        display: inline-block;
        padding: 4px 10px;
        margin-bottom: 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    @media (max-width: 600px) {

        .companion-header {
            padding: 14px;
        }

        .companion-logo {
            font-size: 30px;
        }

        .companion-title {
            font-size: 22px;
        }

        .online-status {
            font-size: 12px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def show_chat_page():
    """Display chat interface."""

    # Initialize session
    if "companion" not in st.session_state:
        st.session_state.companion = CancerSupportCompanion()

    if "session_id" not in st.session_state:
        _, st.session_state.session_id = (
            st.session_state.companion.start_new_conversation("User")
        )

    # Display header
    st.html(
        """
    <div class="companion-header">

        <div class="companion-brand">

                <div class="companion-logo">
                    🎗️
                </div>

                <div>
                    <div class="companion-title">
                        CompanionAI
                    </div>

                    <div class="companion-subtitle">
                        Cancer Support & Education
                    </div>
                </div>

            </div>

            <div class="online-status">
                <span class="status-dot"></span>
                Online
            </div>

    </div>
    """,
        
    )

    # Safety notice
    st.html(
        """
    <div class="safety-banner">
        🛡️ <strong>CompanionAI provides educational and emotional support.</strong>
        It is not a substitute for professional medical advice.
    </div>
    """,
        
    )

    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            if message["role"] == "assistant":
                agent_name = message.get(
                    "agent",
                    "CompanionAI",
                )

                st.markdown(
                    f"""
                    <div class="agent-badge">
                        🤖 {agent_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(message["content"])

    # Text and voice input
    chat_input = st.chat_input(
        "Type or record your message...",
        accept_audio=True,
        audio_sample_rate=16000,
    )

    if chat_input:

        # Get typed message first
        user_message = chat_input.text

        # Transcribe voice input when no text is provided
        if not user_message and chat_input.audio:

            with st.spinner("🎙️ Transcribing your voice..."):
                user_message = (
                    st.session_state.companion
                    .speech_service
                    .transcribe(chat_input.audio)
                )

            if not user_message:
                st.error(
                    "Sorry, I couldn't understand the recording. "
                    "Please try again."
                )
                return

        # Ignore empty input
        if not user_message:
            return

        # Add user message to history
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Display user message
        with st.chat_message("user"):
            st.write(user_message)

        # Process user message through the multi-agent system
        assistant_response = (
            st.session_state.companion.process_message(
                st.session_state.session_id,
                user_message,
            )
        )

        # Get the agent selected by the backend
        agent_name = (
            st.session_state.companion.get_last_agent()
        )

        # Add assistant response to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "agent": agent_name,
            }
        )

        # Display assistant response
        with st.chat_message("assistant"):

            st.html(
                f"""
                <div class="agent-badge">
                    🤖 {agent_name}
                </div>
                """,
            )

            st.write(assistant_response)