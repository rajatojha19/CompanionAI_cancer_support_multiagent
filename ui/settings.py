import streamlit as st


def show_settings():
    """Display CompanionAI settings."""

    st.title("⚙️ Settings")

    st.markdown("---")

    st.subheader("👤 User")

    user_name = st.text_input(
        "Name",
        value=st.session_state.get("user_name", "User"),
    )

    if st.button("💾 Save Name"):
        st.session_state["user_name"] = user_name
        st.success("Name saved successfully.")

    st.markdown("---")

    st.subheader("📊 Session Information")

    session_id = st.session_state.get("session_id")

    if session_id:
        st.write(f"**Session ID:** `{session_id}`")
    else:
        st.info("No active conversation.")

    message_count = len(
        st.session_state.get("messages", [])
    )

    st.write(f"**Messages:** {message_count}")

    st.markdown("---")

    st.subheader("🗑️ Conversation")

    if st.button("Clear Current Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("session_id", None)

        st.success("Current chat cleared.")

    st.markdown("---")

    st.subheader("⚠️ Medical Disclaimer")

    st.info(
        """
CompanionAI provides emotional support and general educational
information. It is not a substitute for professional medical advice,
diagnosis, or treatment.

Always consult a qualified healthcare professional regarding
medical decisions.
"""
    )