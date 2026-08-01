import streamlit as st


def show_chat_page():
    """Display chat interface."""

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

        # Temporary response
        assistant_response = (
            "I am here to support you. "
            "This connection will be linked to the AI agents next."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        with st.chat_message("assistant"):
            st.write(assistant_response)