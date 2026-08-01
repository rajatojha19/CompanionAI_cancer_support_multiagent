import streamlit as st

st.set_page_config(
    page_title="CompanionAI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Sidebar ---------------- #

st.sidebar.title("🤖 CompanionAI")

st.sidebar.button("➕ New Chat")

st.sidebar.markdown("---")

st.sidebar.markdown("### ❤️ Emotional Support")
st.sidebar.markdown("### 📚 Cancer Education")
st.sidebar.markdown("### 📝 Doctor Questions")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📜 Chat History")
st.sidebar.info("No previous conversations")

st.sidebar.markdown("---")

st.sidebar.markdown("⚙️ Settings")

# ---------------- Main Page ---------------- #

st.title("🤖 CompanionAI")

st.subheader("Your AI-powered Cancer Support Companion")

st.write(
    """
Welcome to CompanionAI.

This assistant can help you with:

- ❤️ Emotional support
- 📚 General cancer education
- 📝 Organizing questions for your doctor

Click **New Chat** in the sidebar to begin.
"""
)