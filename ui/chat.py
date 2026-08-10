import streamlit as st

from core.companion import CancerSupportCompanion


def show_chat_page():
    """Display chat interface."""

    if "companion" not in st.session_state:
        st.session_state.companion = CancerSupportCompanion()

    if "session_id" not in st.session_state:
        _, st.session_state.session_id = (
            st.session_state.companion.start_new_conversation("User")
        )

    st.title("🤖 CompanionAI Chat")

    st.caption(
        "Your AI companion for emotional support and cancer education."
    )

    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_message = st.chat_input(
        "Type your message here..."
    )

    if user_message:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.write(user_message)

        # Process message through the real multi-agent system
        assistant_response = (
            st.session_state.companion.process_message(
                st.session_state.session_id,
                user_message
            )
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        with st.chat_message("assistant"):
            st.write(assistant_response)