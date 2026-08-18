from agents.emotional_support_agent import EmotionalSupportAgent
from agents.educational_agent import EducationalAgent
from agents.question_organizer_agent import QuestionOrganizerAgent
from core.companion import CancerSupportCompanion
from models.session import UserSession
from datetime import datetime


def create_test_session():
    return UserSession(
        session_id="test_session",
        user_name="Rajat",
        created_at=datetime.now(),
        conversation_history=[],
        user_preferences={},
    )


def test_detect_scared_emotion():
    agent = EmotionalSupportAgent()

    emotion = agent.detect_emotion(
        "I am feeling scared about my treatment."
    )

    assert emotion == "scared"


def test_detect_sad_emotion():
    agent = EmotionalSupportAgent()

    emotion = agent.detect_emotion(
        "I feel very sad today."
    )

    assert emotion == "sad"


def test_detect_overwhelmed_emotion():
    agent = EmotionalSupportAgent()

    emotion = agent.detect_emotion(
        "I feel overwhelmed and stressed."
    )

    assert emotion == "overwhelmed"


def test_detect_neutral_emotion():
    agent = EmotionalSupportAgent()

    emotion = agent.detect_emotion(
        "Hello, I have a question."
    )

    assert emotion == "neutral"


def test_provide_support():
    agent = EmotionalSupportAgent()
    session = create_test_session()

    response = agent.provide_support(
        "I am feeling scared.",
        session,
    )

    assert response
    assert "I am not a doctor" in response
    assert len(session.conversation_history) == 1
    assert session.conversation_history[0]["role"] == "emotional_agent"

    from agents.educational_agent import EducationalAgent


def test_explain_chemotherapy():
    agent = EducationalAgent()

    response = agent.explain_concept(
        "Can you explain what chemotherapy is?"
    )

    assert "chemotherapy" in response.lower()
    assert "medications" in response.lower()
    assert "I am not a doctor" in response


def test_explain_radiation():
    agent = EducationalAgent()

    response = agent.explain_concept(
        "What is radiation therapy?"
    )

    assert "radiation" in response.lower()
    assert "high-energy beams" in response
    assert "I am not a doctor" in response


def test_explain_biopsy():
    agent = EducationalAgent()

    response = agent.explain_concept(
        "What is a biopsy?"
    )

    assert "biopsy" in response.lower()
    assert "sample of tissue" in response
    assert "I am not a doctor" in response


def test_unknown_educational_topic():
    agent = EducationalAgent()

    response = agent.explain_concept(
        "Tell me about something completely unknown."
    )

    assert "general information" in response
    assert "What specific concept would you like me to explain?" in response
    assert "I am not a doctor" in response

    from agents.question_organizer_agent import QuestionOrganizerAgent


def test_extract_treatment_concern():
    agent = QuestionOrganizerAgent()

    concerns = agent.extract_concerns(
        "What treatment and chemotherapy options are available?"
    )

    assert "treatment" in concerns


def test_extract_multiple_concerns():
    agent = QuestionOrganizerAgent()

    concerns = agent.extract_concerns(
        "I have an appointment tomorrow and I am having pain."
    )

    assert "symptoms" in concerns
    assert "follow_up" in concerns


def test_extract_no_concerns():
    agent = QuestionOrganizerAgent()

    concerns = agent.extract_concerns(
        "Hello, thank you for listening."
    )

    assert concerns == []


def test_generate_treatment_questions():
    agent = QuestionOrganizerAgent()

    response = agent.generate_questions(["treatment"])

    assert "goals of this treatment" in response
    assert "potential side effects" in response
    assert "I am not a doctor" in response


def test_generate_follow_up_questions():
    agent = QuestionOrganizerAgent()

    response = agent.generate_questions(["follow_up"])

    assert "next appointment" in response
    assert "What tests will we do next?" in response


def test_generate_fallback_questions():
    agent = QuestionOrganizerAgent()

    response = agent.generate_questions([])

    assert "Can you explain my diagnosis" in response
    assert "What are the next steps" in response
    assert "I am not a doctor" in response

    from core.companion import CancerSupportCompanion


