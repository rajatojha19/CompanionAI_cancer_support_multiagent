import streamlit as st

from ui.sidebar import show_sidebar
from ui.welcome import show_welcome_page
from ui.chat import show_chat_page
from ui.settings import show_settings


st.set_page_config(
    page_title="CompanionAI",
    page_icon="🤖",
    layout="wide"
)


# Initialize current page
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"


# Display sidebar navigation
show_sidebar()


# Page routing
if st.session_state["page"] == "welcome":
    show_welcome_page()

elif st.session_state["page"] == "chat":
    show_chat_page()

elif st.session_state["page"] == "settings":
    show_settings()
    
st.caption(
    "Developed by Rajat Ojha | "
    "Powered by Google Gemini AI • Streamlit • Python"
)