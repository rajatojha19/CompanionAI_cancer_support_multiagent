from core.companion import CancerSupportCompanion


def test_start_new_conversation():
    companion = CancerSupportCompanion()

    welcome_message, session_id = companion.start_new_conversation("Rajat")

    assert session_id is not None
    assert "Hello Rajat" in welcome_message
    assert session_id in companion.session_manager.sessions
    assert companion.metrics["sessions_created"] == 1


def test_emotional_message_routing():
    companion = CancerSupportCompanion()

    _, session_id = companion.start_new_conversation("Rajat")

    response = companion.process_message(
        session_id,
        "I am feeling scared about my treatment."
    )

    assert response
    assert "I am not a doctor" in response
    assert companion.metrics["messages_processed"] == 1
    assert companion.metrics["emotional_support_given"] == 1


def test_question_message_routing():
    companion = CancerSupportCompanion()

    _, session_id = companion.start_new_conversation("Rajat")

    response = companion.process_message(
        session_id,
        "What should I ask my doctor at my appointment?"
    )

    assert response
    assert "next appointment" in response
    assert companion.metrics["messages_processed"] == 1
    assert companion.metrics["questions_generated"] == 1


def test_educational_message_routing():
    companion = CancerSupportCompanion()

    _, session_id = companion.start_new_conversation("Rajat")

    response = companion.process_message(
        session_id,
        "What is chemotherapy?"
    )

    assert response
    assert "chemotherapy" in response.lower()
    assert companion.metrics["messages_processed"] == 1
    assert companion.metrics["concepts_explained"] == 1


def test_invalid_session():
    companion = CancerSupportCompanion()

    response = companion.process_message(
        "invalid_session_id",
        "Hello"
    )

    assert "couldn't find your conversation" in response


def test_get_conversation_history():
    companion = CancerSupportCompanion()

    _, session_id = companion.start_new_conversation("Rajat")

    companion.process_message(
        session_id,
        "I am feeling scared."
    )

    history = companion.get_conversation_history(session_id)

    assert len(history) >= 2
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "I am feeling scared."


def test_get_metrics():
    companion = CancerSupportCompanion()

    _, session_id = companion.start_new_conversation("Rajat")

    companion.process_message(
        session_id,
        "I am feeling worried."
    )

    metrics = companion.get_metrics()

    assert metrics["sessions_created"] == 1
    assert metrics["messages_processed"] == 1
    assert metrics["emotional_support_given"] == 1