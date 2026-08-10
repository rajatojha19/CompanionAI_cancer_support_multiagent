import streamlit as st

from core.companion import CancerSupportCompanion


def show_sidebar():
    """Display the CompanionAI sidebar."""

    st.sidebar.title("🤖 CompanionAI")

    # Make sure the backend companion exists
    if "companion" not in st.session_state:
        st.session_state.companion = CancerSupportCompanion()

    # New Chat
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

    # Chat History
    st.sidebar.markdown("### 📜 Chat History")

    sessions = st.session_state.companion.session_manager.get_all_sessions()

    if not sessions:
        st.sidebar.info("No previous conversations")
    else:
        for session_id, session in sessions.items():

            label = (
                f"💬 {session.user_name} — "
                f"{session.created_at.strftime('%d %b %Y, %H:%M')}"
            )

            if st.sidebar.button(
                label,
                key=f"history_{session_id}",
                use_container_width=True,
            ):
                st.session_state["session_id"] = session_id
                st.session_state["page"] = "chat"

                # Load selected conversation
                st.session_state["messages"] = []

                for message in session.conversation_history:
                    role = message["role"]

                    if role == "user":
                        display_role = "user"
                    else:
                        display_role = "assistant"

                    st.session_state["messages"].append(
                        {
                            "role": display_role,
                            "content": message["content"],
                        }
                    )

                st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state["page"] = "settings"
        st.rerun()