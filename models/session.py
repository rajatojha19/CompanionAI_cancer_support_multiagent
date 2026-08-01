from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger("CancerSupportCompanion")

@dataclass
class UserSession:
    """Session management for user conversations"""

    session_id: str
    user_name: str
    created_at: datetime
    conversation_history: List[Dict]
    user_preferences: Dict

    def add_message(self, role: str, content: str):
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
            }
        )
        logger.info(f"Session {self.session_id}: Added {role} message")