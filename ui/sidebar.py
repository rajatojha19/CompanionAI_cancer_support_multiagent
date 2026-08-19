import streamlit as st

from core.companion import CancerSupportCompanion


def show_sidebar():
    """Display the CompanionAI sidebar."""

    # ============================================================
    # Backend initialization
    # ============================================================

    if "companion" not in st.session_state:
        st.session_state.companion = CancerSupportCompanion()

    # ============================================================
    # Sidebar Header
    # ============================================================

    st.sidebar.title("🎗️ CompanionAI")
    st.sidebar.caption("Cancer Support & Education")

    st.sidebar.markdown("---")

    # ============================================================
    # Home
    # ============================================================

    if st.sidebar.button(
        "⌂  Home",
        use_container_width=True,
    ):
        st.session_state["page"] = "welcome"
        st.rerun()

    # ============================================================
    # New Chat
    # ============================================================

    if st.sidebar.button(
        "＋  New Chat",
        use_container_width=True,
    ):
        st.session_state["page"] = "chat"
        st.session_state["messages"] = []
        st.session_state.pop("session_id", None)
        st.rerun()

    st.sidebar.markdown("---")

    # ============================================================
    # Quick Access
    # ============================================================

    st.sidebar.markdown("### Quick Access")

    if st.sidebar.button(
        "💗  Emotional Support",
        use_container_width=True,
    ):
        st.session_state["page"] = "chat"
        st.session_state["suggested_topic"] = (
            "I'm feeling overwhelmed and would like some emotional support."
        )
        st.rerun()

    if st.sidebar.button(
        "📚  Cancer Education",
        use_container_width=True,
    ):
        st.session_state["page"] = "chat"
        st.session_state["suggested_topic"] = (
            "I would like to learn about a cancer-related topic."
        )
        st.rerun()

    if st.sidebar.button(
        "📝  Doctor Questions",
        use_container_width=True,
    ):
        st.session_state["page"] = "chat"
        st.session_state["suggested_topic"] = (
            "I want help preparing questions for my medical team."
        )
        st.rerun()

    st.sidebar.markdown("---")

    # ============================================================
    # Chat History
    # ============================================================

    st.sidebar.markdown("### 📜 Chat History")

    sessions = (
        st.session_state
        .companion
        .session_manager
        .get_all_sessions()
    )

    if not sessions:

        st.sidebar.caption(
            "No previous conversations."
        )

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

    # ============================================================
    # Settings
    # ============================================================

    if st.sidebar.button(
        "⚙️  Settings",
        use_container_width=True,
    ):
        st.session_state["page"] = "settings"
        st.rerun()