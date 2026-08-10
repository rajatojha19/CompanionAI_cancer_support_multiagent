from datetime import datetime
from typing import Dict, Optional
from utils.logger import logger
from models.session import UserSession

class SessionManager:
    """Manages user sessions and memory"""

    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        logger.info("SessionManager: Initialized session management")

    def create_session(self, user_name: str) -> UserSession:
        session_id = f"session_{len(self.sessions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = UserSession(
            session_id=session_id,
            user_name=user_name,
            created_at=datetime.now(),
            conversation_history=[],
            user_preferences={},
        )
        self.sessions[session_id] = session
        logger.info(f"SessionManager: Created new session {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        return self.sessions.get(session_id)

    def save_session_state(self, session: UserSession):
        logger.info(f"SessionManager: Saved state for session {session.session_id}")

    def get_all_sessions(self) -> Dict[str, UserSession]:
        """Return all active user sessions."""
        return self.sessions.copy()