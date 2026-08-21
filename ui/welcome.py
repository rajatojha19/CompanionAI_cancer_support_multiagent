import streamlit as st

def show_welcome_page():
    """Display the CompanionAI welcome page."""

    # ============================================================
    # Header
    # ============================================================

    st.markdown(
        """
        # 🎗️ CompanionAI

        ### Support when you need it. Information when you seek it.

        CompanionAI is designed to provide emotional support and
        general cancer-related education in a simple, conversational way.
        """
    )

    st.divider()

    # ============================================================
    # Quick Actions
    # ============================================================

    st.subheader("How can I help you today?")

    st.write("Choose an option to get started.")

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------------
    # Emotional Support
    # ------------------------------------------------------------

    with col1:

        st.markdown("### 💗 Emotional Support")

        st.caption(
            "Talk about how you're feeling "
            "in a supportive environment."
        )

        if st.button(
            "Talk to CompanionAI",
            key="home_emotional",
            use_container_width=True,
        ):
            st.session_state["page"] = "chat"

            st.session_state["suggested_topic"] = (
                "I'm feeling overwhelmed and would like some emotional support."
            )

            st.rerun()

    # ------------------------------------------------------------
    # Cancer Education
    # ------------------------------------------------------------

    with col2:

        st.markdown("### 📚 Cancer Education")

        st.caption(
            "Learn about cancer-related concepts "
            "in simple language."
        )

        if st.button(
            "Explore Education",
            key="home_education",
            use_container_width=True,
        ):
            st.session_state["page"] = "chat"

            st.session_state["suggested_topic"] = (
                "I would like to learn about a cancer-related topic."
            )

            st.rerun()

    # ------------------------------------------------------------
    # Doctor Questions
    # ------------------------------------------------------------

    with col3:

        st.markdown("### 📝 Doctor Questions")

        st.caption(
            "Prepare organized questions "
            "for your medical team."
        )

        if st.button(
            "Prepare Questions",
            key="home_questions",
            use_container_width=True,
        ):
            st.session_state["page"] = "chat"

            st.session_state["suggested_topic"] = (
                "I want help preparing questions for my medical team."
            )

            st.rerun()

    # ============================================================
    # Start Conversation
    # ============================================================

    st.divider()

    if st.button(
        "🚀 Start a Conversation",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["page"] = "chat"
        st.session_state.pop("suggested_topic", None)
        st.rerun()

    # ============================================================
    # Safety Information
    # ============================================================

    st.divider()

    st.subheader("🛡️ Safe & Responsible")

    st.info(
        "CompanionAI provides emotional support and general "
        "cancer-related educational information."
    )

    st.caption(
        "CompanionAI is not a substitute for professional medical "
        "advice, diagnosis, or treatment. Always consult a qualified "
        "healthcare professional for medical decisions."
    )
    st.divider()

    st.caption(
        "Developed by Rajat Ojha | "
        "Powered by Google Gemini AI • Streamlit • Python"
    )