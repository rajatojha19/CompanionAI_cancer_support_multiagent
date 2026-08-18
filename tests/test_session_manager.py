from services.session_manager import SessionManager


def test_create_session():
    manager = SessionManager()

    session = manager.create_session("Rajat")

    assert session.user_name == "Rajat"
    assert session.session_id in manager.sessions
    assert session.conversation_history == []


def test_get_session():
    manager = SessionManager()

    session = manager.create_session("Rajat")

    result = manager.get_session(session.session_id)

    assert result is session


def test_get_all_sessions():
    manager = SessionManager()

    session1 = manager.create_session("Rajat")
    session2 = manager.create_session("Alex")

    sessions = manager.get_all_sessions()

    assert len(sessions) == 2
    assert session1.session_id in sessions
    assert session2.session_id in sessions


def test_session_add_message():
    manager = SessionManager()

    session = manager.create_session("Rajat")

    session.add_message("user", "Hello")

    assert len(session.conversation_history) == 1
    assert session.conversation_history[0]["role"] == "user"
    assert session.conversation_history[0]["content"] == "Hello"