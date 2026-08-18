# 🤖 Cancer Support CompanionAI — Multi-Agent AI System

## 🎯 Project Overview

**Cancer Support CompanionAI** is a multi-agent AI application designed to provide emotional support, general cancer-related education, and assistance in organizing questions for medical appointments.

The project was developed for the **Agents for Good** track of the **Google AI Agents Intensive Capstone Project**, with a focus on responsible AI use in healthcare-support applications.

CompanionAI is designed to help users:

- 💗 Talk through difficult emotions
- 📚 Understand general cancer-related concepts
- 📝 Organize questions before medical appointments
- 💬 Maintain conversation sessions
- 📜 Access previous conversations during the current application session

> ⚠️ **Medical Disclaimer:** CompanionAI is not a doctor and does not provide diagnosis, treatment decisions, medication instructions, or clinical advice. It provides emotional support and general educational information. Always consult a qualified healthcare professional regarding medical decisions.

# ✨ Features

## ❤️ Emotional Support

The `EmotionalSupportAgent` provides supportive responses when users express emotions such as:

- Fear
- Sadness
- Anxiety
- Feeling overwhelmed
- Worry

The agent can detect basic emotional states and provide empathetic responses.

Example:

User:
I am feeling scared about my upcoming surgery.

CompanionAI:
I hear the fear in your words. Would you like to talk about
what's making you feel most anxious?

⚠️ I am not a doctor and cannot provide medical advice.
Please consult a qualified medical professional.

## 📚 Cancer Education

The `EducationalAgent` provides general, easy-to-understand information about cancer-related concepts.

Currently supported topics include:

- Chemotherapy
- Radiation
- Biopsy
- Remission
- Side effects
- Supportive care

Example:

User:
What is chemotherapy?

CompanionAI:
Here's some general information about chemotherapy:

Chemotherapy uses medications to treat cancer.
These medications work by targeting rapidly dividing cells.

⚠️ I am not a doctor and cannot provide medical advice.
Please consult a qualified medical professional.

The educational agent is intentionally designed to remain at a general informational level.


## 📝 Doctor Question Organizer

The `QuestionOrganizerAgent` helps users prepare questions for their healthcare team.

It identifies concerns related to:

- Treatment
- Symptoms
- Lifestyle
- Follow-up

For example:

User:
What should I ask my doctor about my treatment
and my next appointment?

CompanionAI:
Based on what you've shared, here are some questions
you might want to ask your medical team:

1. What are the goals of this treatment?
2. What are the potential side effects I should watch for?

Remember to write down your questions before appointments.

# 🧠 Multi-Agent Architecture

The application uses a central coordinator called:

>CancerSupportCompanion


It coordinates the specialized agents and session management.


                         ┌───────────────────────┐
                         │     Streamlit UI      │
                         │                       │
                         │ Welcome / Chat /      │
                         │ Sidebar / Settings    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │  CancerSupportCompanion  │
                       │      Core Coordinator    │
                       └────────────┬─────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
     ┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
     │   Emotional    │   │     Question     │   │   Educational   │
     │ Support Agent  │   │ Organizer Agent  │   │      Agent      │
     └────────────────┘   └──────────────────┘   └─────────────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  SessionManager  │
                           │ Session & Memory │
                           └──────────────────┘


# 🤖 Agents

## 1. EmotionalSupportAgent

File:
agents/emotional_support_agent.py


Responsibilities:

- Detect basic emotional states
- Provide supportive responses
- Encourage users to talk about their feelings
- Provide a safety disclaimer
- Store generated responses in the conversation session

Supported emotional categories include:

>scared
>sad
>overwhelmed
>neutral

## 2. EducationalAgent

File:
agents/educational_agent.py

Responsibilities:

- Explain common cancer-related concepts
- Use simple language
- Keep explanations general
- Avoid treatment decisions and medical instructions
- Provide a medical disclaimer

Supported topics include:


chemotherapy
radiation
biopsy
remission
side_effects
support_care


## 3. QuestionOrganizerAgent

File:
agents/question_organizer_agent.py


Responsibilities:

- Extract concerns from user messages
- Categorize concerns
- Generate questions for medical appointments

Categories:


treatment
symptoms
lifestyle
follow_up


The agent currently uses pattern matching and predefined question templates.

# 🧩 Core Components

## CancerSupportCompanion

File:
core/companion.py

This is the main coordinator of the multi-agent system.

It:
- Creates new conversations
- Routes messages to appropriate agents
- Maintains session information
- Tracks application metrics
- Provides conversation history

Message routing is currently based on simple keyword and phrase matching.

## SessionManager

File:
services/session_manager.py

The `SessionManager` manages conversation sessions in application memory.

Each session contains:

- Session ID
- User name
- Creation timestamp
- Conversation history
- User preferences

Example:


UserSession
├── session_id
├── user_name
├── created_at
├── conversation_history
└── user_preferences


> **Current Version 1 limitation:** Sessions are stored in application memory. Persistent database storage is not implemented yet.

## GeminiClient

File:
services/gemini_service.py

The project includes an optional Gemini integration.

Gemini can be used by:

- `EmotionalSupportAgent`
- `EducationalAgent`

The application can also operate without a Gemini API key using the available stub-mode behavior.

This makes it possible to run and test the application without requiring an active API key.

# 🎨 Streamlit User Interface

Version 1 uses **Streamlit** as the frontend.

The UI contains several components.

## 🏠 Welcome Page

The welcome page introduces CompanionAI and provides access to:

- Emotional Support
- Cancer Education
- Doctor Questions
- Start Conversation

## 💬 Chat Page

The chat interface allows users to:

- Send messages
- Receive agent responses
- View conversation history
- Continue an active session

The chat is connected directly to:


CancerSupportCompanion


## 📂 Sidebar

The sidebar provides:

- New Chat
- Emotional Support
- Cancer Education
- Doctor Questions
- Chat History
- Settings

## 📜 Chat History

Users can select previous sessions created during the current application run.

Selecting a session loads its stored conversation history into the chat interface.

> Persistent history across application restarts is not implemented in Version 1.

## ⚙️ Settings

The Settings page currently provides:

- User name management
- Session information
- Message count
- Clear current chat
- Medical disclaimer

# 📊 Logging and Metrics

Centralized logging is implemented through:
utils/logger.py


The application records useful events such as:

- Session creation
- Agent initialization
- Emotion detection
- Question generation
- Message processing

The core system also tracks metrics:

>sessions_created
>messages_processed
>emotional_support_given
>questions_generated
>concepts_explained


# 🛠️ Technology Stack

| Technology | Purpose ||
| Python | Core application |
| Streamlit | User interface |
| Google Gemini | Optional LLM integration |
| python-dotenv | Environment variable management |
| Requests | HTTP requests |
| Pytest | Automated testing |
| Git | Version control |

# 📁 Project Structure


CompanionAI_cancer_support_multiagent/
│
├── agents/
│   ├── __init__.py
│   ├── emotional_support_agent.py
│   ├── educational_agent.py
│   └── question_organizer_agent.py
│
├── core/
│   ├── __init__.py
│   └── companion.py
│
├── docs/
│
├── models/
│   ├── __init__.py
│   └── session.py
│
├── services/
│   ├── __init__.py
│   ├── gemini_service.py
│   └── session_manager.py
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_companion.py
│   └── test_session_manager.py
│
├── ui/
│   ├── __init__.py
│   ├── chat.py
│   ├── components.py
│   ├── settings.py
│   ├── sidebar.py
│   └── welcome.py
│
├── utils/
│   ├── __init__.py
│   └── logger.py
│
├── .env.example
├── .gitignore
├── app.py
├── config.py
├── LICENSE
├── main.py
├── README.md
└── requirements.txt


# 🚀 Installation

## 1. Clone the Repository

bash
git clone <YOUR_GITHUB_REPOSITORY_URL>


Then move into the project directory:

bash
cd CompanionAI_cancer_support_multiagent


## 2. Install Python Dependencies

Install the required packages:

bash
py -m pip install -r requirements.txt


# 🔐 Gemini API Configuration

Gemini integration is optional for the current Version 1 implementation.

The repository contains:

.env.example


with:

env
GEMINI_API_KEY=your_gemini_api_key_here


To configure Gemini locally:

### Step 1

Create a `.env` file in the project root.

### Step 2

Add your API key:

>GEMINI_API_KEY=your_actual_api_key


### Step 3

Run the application.

> ⚠️ **Never commit your real `.env` file or API key to GitHub.**

The `.gitignore` file already contains:

>.env


so the local API key should remain untracked.

If no API key is configured, the application can run using the available stub-mode behavior.

# ▶️ Running the Streamlit Application

From the project root:

bash
py -m streamlit run app.py


Streamlit will display a local URL similar to:
http://localhost:8501


Open that address in your browser.

# 🖥️ Application Flow

The basic application flow is:


Start Application
       │
       ▼
 Welcome Page
       │
       ▼
 Start Conversation
       │
       ▼
 Create User Session
       │
       ▼
 Chat Interface
       │
       ├───────────────┐
       ▼               ▼
 Emotional         Educational
 Support           Information
       │               │
       └───────┬───────┘
               ▼
       Question Organizer
               │
               ▼
       Save Session State


# 🧪 Automated Testing

The project includes automated tests using `pytest`.

The current test suite contains:


15 Agent tests
7 CancerSupportCompanion tests
4 SessionManager test-
26 Total tests


-> Run the complete test suite:

bash
py -m pytest -v


-> Expected result:
26 passed


## Agent Tests

`tests/test_agents.py` verifies:

### EmotionalSupportAgent

- Scared emotion detection
- Sad emotion detection
- Overwhelmed emotion detection
- Neutral emotion detection
- Support response generation

### EducationalAgent

- Chemotherapy explanation
- Radiation explanation
- Biopsy explanation
- Unknown-topic fallback

### QuestionOrganizerAgent

- Treatment concern extraction
- Multiple concern extraction
- No concern detection
- Treatment question generation
- Follow-up question generation
- Fallback question generation

## CancerSupportCompanion Tests

`tests/test_companion.py` verifies:

- New conversation creation
- Emotional message routing
- Question message routing
- Educational message routing
- Invalid session handling
- Conversation history
- Metrics

## SessionManager Tests

`tests/test_session_manager.py` verifies:

- Session creation
- Session retrieval
- Getting all sessions
- Adding messages to sessions

# 🛡️ Safety and Responsible AI

CompanionAI is designed as a support and educational application.

It is **not intended to replace healthcare professionals**.

The application does not provide:

- Cancer diagnosis
- Medical diagnosis
- Medication prescriptions
- Medication dosages
- Treatment decisions
- Personalized clinical instructions

Instead, the system focuses on:

- Emotional support
- General educational information
- Preparing questions for healthcare professionals

Users should always consult qualified healthcare professionals for medical decisions.

# ⚠️ Current Limitations

Version 1 is a prototype and has several limitations.

### 1. In-Memory Sessions

Conversation sessions are stored in application memory.

Restarting the application removes the current sessions.

### 2. No Authentication

User authentication and account management are not implemented.

### 3. No Persistent Database

There is currently no PostgreSQL, MySQL, MongoDB, or other persistent database.

### 4. Basic Message Routing

The core coordinator currently uses keyword and phrase matching to determine which agent should process a message.

### 5. Optional Gemini Integration

The application can run without Gemini, using the available stub-mode behavior.

### 6. Not a Clinical System

The system has not been designed or validated as a clinical decision-support system.

# 📌 Version 1 Status

## Version 1 — Streamlit Prototype

Current Version 1 functionality includes:

- [x] Multi-agent architecture
- [x] Emotional support agent
- [x] Educational agent
- [x] Question organizer agent
- [x] Session management
- [x] Streamlit welcome page
- [x] Streamlit chat interface
- [x] Sidebar navigation
- [x] New Chat
- [x] Chat History
- [x] Settings page
- [x] Logging
- [x] Metrics
- [x] Optional Gemini integration
- [x] Automated testing
- [x] 26 passing tests

# 🏆 Project Objective

The goal of Cancer Support CompanionAI is to demonstrate how a multi-agent AI system can provide useful emotional and educational support while maintaining clear boundaries around medical advice.

The project emphasizes:

- Responsible AI
- Human-centered design
- Modular architecture
- Agent specialization
- Safety awareness
- Testable software components


# 👨‍💻 Project

**Cancer Support CompanionAI**

Built as a multi-agent AI project for the **Agents for Good** track of the Google AI Agents Intensive Capstone Project.