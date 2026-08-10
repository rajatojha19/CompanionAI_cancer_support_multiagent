import streamlit as st


def show_sidebar():
    """Display the CompanionAI sidebar."""

    st.sidebar.title("🤖 CompanionAI")

    if st.sidebar.button("➕ New Chat", use_container_width=True):
        st.session_state["page"] = "chat"
        st.session_state["messages"] = []
        st.session_state.pop("session_id", None)
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ❤️ Emotional Support")
    st.sidebar.markdown("### 📚 Cancer Education")
    st.sidebar.markdown("### 📝 Doctor Questions")

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📜 Chat History")

    if not st.session_state.get("messages"):
        st.sidebar.info("No previous conversations")

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⚙️ Settings")