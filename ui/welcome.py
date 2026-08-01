import streamlit as st


def show_welcome_page():
    """Display the CompanionAI welcome page."""

    st.title("🤖 CompanionAI")

    st.subheader("Your trusted AI companion for cancer support and education")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
### ❤️ Emotional Support

Talk through your feelings in a safe and supportive environment.
"""
        )

    with col2:
        st.info(
            """
### 📚 Cancer Education

Learn about cancer treatments and general medical concepts.
"""
        )

    with col3:
        st.info(
            """
### 📝 Doctor Questions

Prepare organized questions before your next appointment.
"""
        )

    st.markdown("---")

    st.markdown(
        """
**Medical Disclaimer**

CompanionAI provides emotional support and general educational information.
It is **not** a substitute for professional medical advice, diagnosis, or treatment.
Always consult your healthcare provider regarding medical decisions.
"""
    )

    st.markdown("")

    if st.button("🚀 Start Conversation", use_container_width=True):
        st.session_state["page"] = "chat"
        st.rerun()
        