import streamlit as st

from ui.welcome import show_welcome_page
from ui.chat import show_chat_page

st.set_page_config(
    page_title="CompanionAI",
    page_icon="🤖",
    layout="wide"
)


if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

if st.session_state["page"] == "welcome":
    show_welcome_page()

elif st.session_state["page"] == "chat":
    show_chat_page()
