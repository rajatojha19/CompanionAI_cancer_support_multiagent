# Cancer Support CompanionAI - Multi-Agent AI System

## 🎯 Project Overview
**Cancer Support Companion** is a multi-agent AI system designed to provide emotional support, educational assistance, and question organization for individuals affected by cancer. Built for the **Agents for Good** track in the Google AI Agents Intensive Capstone Project, this system demonstrates responsible AI development in healthcare applications.

## 🏆 Competition Alignment
- **Track**: Agents for Good (Healthcare)
- **Category**: Emotional support and educational assistance for cancer patients
- **Key Features**: Multi-agent system, session management, safety protocols, observability

## 🎗️ Problem Statement
Cancer patients and caregivers face:
- Emotional distress and anxiety during treatment
- Information overload about medical concepts
- Difficulty organizing questions for medical appointments
- Feelings of isolation and overwhelm

## 💡 Solution Architecture
A sophisticated multi-agent system that provides:

### 🤗 Emotional Support Agent
- Detects emotional states (scared, sad, overwhelmed)
- Provides empathetic responses and coping strategies
- Uses Gemini AI for nuanced emotional support
- Maintains psychological safety boundaries

### 📝 Question Organizer Agent
- Extracts concerns from user messages using pattern matching
- Generates categorized questions for medical teams
- Covers treatment, symptoms, lifestyle, and follow-up topics
- Helps users prepare effectively for medical appointments

### 📚 Educational Agent
- Explains cancer-related concepts in simple terms
- Uses Gemini AI for detailed explanations
- Maintains general educational scope without medical advice
- Provides safety-disclaimed information

### 🎛️ Core System Components
- **Session Manager**: Persistent conversation memory and state management
- **Gemini Client**: LLM integration with safety constraints
- **Safety Layer**: Mandatory medical disclaimers and content filtering
- **Observability**: Comprehensive logging and metrics tracking

## 🚀 Technical Implementation

### Multi-Agent System Features
```python
# Core architecture
CancerSupportCompanion
├── EmotionalSupportAgent (LLM-powered)
├── QuestionOrganizerAgent (Rule-based + Patterns)
├── EducationalAgent (LLM-powered)
└── SessionManager (State Management)
