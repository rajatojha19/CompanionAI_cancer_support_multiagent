# 🤖 Cancer Support CompanionAI

A multi-agent AI application designed to provide **emotional support, general cancer education, and assistance in preparing questions for healthcare professionals**.

CompanionAI uses specialized AI agents coordinated through a central orchestration layer and provides an interactive Streamlit web interface with text and voice interaction.

Built for the **Agents for Good** track of the Google AI Agents Intensive Capstone Project.

---

## 🌐 Live Application

Try CompanionAI directly in your browser:

🚀 **Live Demo:** https://companionai-cancer-support.streamlit.app/

The deployed application provides:

- 💬 Interactive AI conversation
- 🎙️ Voice input
- 🔊 Read Aloud responses
- ❤️ Emotional support
- 📚 Cancer-related education
- 📝 Doctor question organization
- 📜 Conversation history
- ⚙️ Settings
- 🌐 English, Hindi, and Hinglish interaction

---

## ✨ Features

### ❤️ Emotional Support

The Emotional Support Agent provides supportive conversation for users experiencing emotions such as:

- Fear
- Sadness
- Anxiety
- Stress
- Feeling overwhelmed

The agent provides concise, empathetic responses while avoiding diagnosis or treatment recommendations.

---

### 📚 Cancer Education

The Educational Agent provides general information about cancer-related concepts such as:

- Chemotherapy
- Radiation therapy
- Biopsy
- Remission
- Side effects
- Supportive care
- Other general cancer-related concepts

The agent adapts its response to the user's language:

- English → English
- Hindi → Hindi
- Hinglish / Roman Hindi → Hinglish / Roman Hindi

Responses are intentionally kept concise and easy to understand.

---

### 📝 Doctor Question Organizer

The Question Organizer Agent helps users prepare questions for their healthcare professionals.

It can organize concerns related to:

- Treatment
- Symptoms and side effects
- Lifestyle
- Follow-up and monitoring

Example:

```text
User:
doctor se biopsy ke baare mein kya puchna chahiye?

CompanionAI:
1. Biopsy ka purpose kya hai?
2. Biopsy ke results kab tak milenge?