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

## ✨ Features

### ❤️ Emotional Support
- Detects basic emotions such as scared, sad, and overwhelmed.
- Provides supportive and empathetic responses.
- Includes safety disclaimers.

### 📚 Cancer Education
Provides general information about:
- Chemotherapy
- Radiation
- Biopsy
- Remission
- Side effects
- Supportive care

### 📝 Doctor Question Organizer
- Extracts concerns from user messages.
- Categorizes concerns into treatment, symptoms, lifestyle, and follow-up.
- Generates questions users can discuss with their medical team.

### 💬 Streamlit Interface
- Welcome page
- Interactive chat
- Sidebar navigation
- New Chat
- Chat History
- Previous session loading
- Settings page

### 🧠 Session Management
Sessions currently contain:
- Session ID
- User name
- Creation time
- Conversation history
- User preference 

## 🏗️ Architecture

```text
                    Streamlit UI
                         │
                         ▼
              CancerSupportCompanion
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Emotional       Question       Educational
     Support Agent   Organizer       Agent
                         │
                         ▼
                  SessionManager
```

### Main Components

| Component | Responsibility |
|---|---|
| `EmotionalSupportAgent` | Emotional support and basic emotion detection |
| `EducationalAgent` | General cancer-related education |
| `QuestionOrganizerAgent` | Extracts concerns and generates doctor questions |
| `CancerSupportCompanion` | Coordinates the agents |
| `SessionManager` | Manages conversation sessions |
| `GeminiClient` | Optional Gemini integration |
| `Streamlit UI` | User interface |

---

## 🛠️ Technology Stack

- **Python**
- **Streamlit**
- **Google Gemini API**
- **python-dotenv**
- **Requests**
- **Pytest**
- **Git**

---

## 📁 Project Structure

```text
CompanionAI_cancer_support_multiagent/
│
├── agents/
│   ├── emotional_support_agent.py
│   ├── educational_agent.py
│   └── question_organizer_agent.py
│
├── core/
│   └── companion.py
│
├── models/
│   └── session.py
│
├── services/
│   ├── gemini_service.py
│   └── session_manager.py
│
├── tests/
│   ├── test_agents.py
│   ├── test_companion.py
│   └── test_session_manager.py
│
├── ui/
│   ├── chat.py
│   ├── components.py
│   ├── settings.py
│   ├── sidebar.py
│   └── welcome.py
│
├── utils/
│   └── logger.py
│
├── docs/
├── .env.example
├── .gitignore
├── app.py
├── config.py
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CompanionAI_cancer_support_multiagent
```

### 2. Install dependencies

```bash
py -m pip install -r requirements.txt
```

---

## 🔐 Gemini Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key
```

The repository contains `.env.example` as a template:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit your real API key.

The `.gitignore` file excludes:

```text
.env
```

If no Gemini API key is configured, the application can run using its available stub-mode behavior.

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
py -m streamlit run app.py
```

Then open the local URL provided by Streamlit, normally:

```text
http://localhost:8501
```

---

## 🧪 Run Tests

Run the complete test suite:

```bash
py -m pytest -v
```

### Current Test Coverage

```text
15 Agent tests
7 Companion integration tests
4 SessionManager tests
-------------------------
26 Total tests
```

Current result:

```text
26 passed
```

The tests cover:

- Emotion detection
- Emotional support responses
- Educational topic explanations
- Question extraction
- Question generation
- Session creation
- Session retrieval
- Message routing
- Conversation history
- Metrics
- Invalid session handling

---

## 📊 Logging and Metrics

Centralized logging is implemented through:

```text
utils/logger.py
```

The application tracks metrics including:

```text
sessions_created
messages_processed
emotional_support_given
questions_generated
concepts_explained
```

---

## 🛡️ Responsible AI

CompanionAI is intended as a support and educational tool.

It does **not** provide:

- Cancer diagnosis
- Medical diagnosis
- Medication prescriptions
- Medication dosages
- Treatment decisions
- Personalized clinical instructions

The system instead focuses on:

- Emotional support
- General educational information
- Preparing questions for healthcare professionals

---

## 📌 Version 1 Status

- [x] Multi-agent architecture
- [x] Emotional Support Agent
- [x] Educational Agent
- [x] Question Organizer Agent
- [x] Session Management
- [x] Streamlit UI
- [x] Sidebar navigation
- [x] New Chat
- [x] Chat History
- [x] Settings
- [x] Logging
- [x] Metrics
- [x] Optional Gemini integration
- [x] Automated testing
- [x] 26 passing tests

⚠️ **Medical Disclaimer:** CompanionAI is not a doctor and does not provide diagnosis, treatment decisions, medication instructions, or clinical advice. It provides emotional support and general educational information. Always consult a qualified healthcare professional for medical decisions.
