"""Persistent Multi-Turn Chat History Manager for Project Jane.

Maintains conversation history across turns and sessions:
- Persists dialogues to logs/chat_history.jsonl
- Provides rolling context window (last N turns) for LLaMA prompt construction
- Enables conversational continuity across session restarts
"""

import os
import json
import time
from collections import deque
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("chat_history")


class ChatHistoryManager:
    """
    Manages multi-turn conversation memory with JSONL persistence and capped ring buffer.
    Maintains an unbounded critical_events list to ensure safety violations are never evicted.
    """

    DEFAULT_LOG_PATH = "logs/chat_history.jsonl"

    def __init__(
        self,
        history_path: Optional[str] = None,
        auto_load: bool = True,
        max_messages: int = 1000
    ):
        if history_path is not None:
            self.history_path = history_path
        else:
            env_path = os.environ.get("JANE_CHAT_HISTORY_PATH")
            if env_path:
                self.history_path = env_path
            elif "PYTEST_CURRENT_TEST" in os.environ:
                self.history_path = os.environ.get(
                    "JANE_TEST_CHAT_HISTORY_PATH",
                    "logs/test_chat_history.jsonl"
                )
            else:
                self.history_path = self.DEFAULT_LOG_PATH
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)
        self.critical_events: List[Dict[str, Any]] = []

        # Ensure directory exists
        dir_name = os.path.dirname(self.history_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if auto_load:
            self.load_history()

    @staticmethod
    def _is_critical_record(record: Dict[str, Any]) -> bool:
        """Determines if a message is a critical safety, halt, or violation event."""
        if record.get("metadata", {}).get("critical"):
            return True
        role = str(record.get("role", "")).lower()
        content = str(record.get("content", "")).upper()
        if role == "system" and any(k in content for k in ("EMERGENCY", "HALT", "VIOLATION", "DISSONANCE", "WATCHDOG")):
            return True
        if any(k in content for k in ("EMERGENCY_HALT", "SAFETY INTERRUPT", "Z3 SMT ARITHMETIC CONTRADICTION")):
            return True
        return False

    def load_history(self) -> List[Dict[str, Any]]:
        """Loads conversation history from the persistent JSONL log."""
        self.messages.clear()
        self.critical_events.clear()
        if not os.path.exists(self.history_path):
            return list(self.messages)

        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "role" in record and "content" in record:
                            self.messages.append(record)
                            if self._is_critical_record(record):
                                self.critical_events.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning("Corrupted JSON line in %s: %s", self.history_path, e)
                        continue
        except FileNotFoundError:
            logger.debug("Chat history file %s not yet created; starting fresh session", self.history_path)
        except (OSError, IOError) as e:
            logger.error("Failed to read chat history from %s: %s", self.history_path, e, exc_info=True)

        return list(self.messages)

    def append_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Appends a single message to history and persists to JSONL."""
        msg: Dict[str, Any] = {
            "timestamp": time.time(),
            "role": role,
            "content": content
        }
        if metadata:
            msg["metadata"] = metadata

        self.messages.append(msg)
        if self._is_critical_record(msg):
            self.critical_events.append(msg)

        try:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except (OSError, IOError) as e:
            logger.error("Failed to append message to %s: %s", self.history_path, e, exc_info=True)

        return msg

    def append_turn(
        self,
        user_text: str,
        assistant_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Appends both user query and assistant response turn to memory and disk."""
        self.append_message("user", user_text, metadata=metadata)
        self.append_message("assistant", assistant_text, metadata=metadata)

    def get_recent_history(self, max_turns: int = 6) -> List[Dict[str, str]]:
        """
        Returns the last `max_turns` dialogue turns (up to max_turns * 2 messages)
        in standard [{"role": "user"|"assistant", "content": "..."}] format.
        """
        max_messages = max_turns * 2
        msg_list = list(self.messages)
        recent = msg_list[-max_messages:] if len(msg_list) > max_messages else msg_list
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_critical_events(self) -> List[Dict[str, Any]]:
        """Returns all recorded critical safety events and system violations."""
        return list(self.critical_events)

    def clear(self):
        """Clears memory and truncates the persistent file."""
        self.messages.clear()
        self.critical_events.clear()
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                pass
        except (OSError, IOError) as e:
            logger.error("Failed to clear chat history file %s: %s", self.history_path, e, exc_info=True)

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def turn_count(self) -> int:
        """Counts complete user-assistant dialogue turns."""
        return len(self.messages) // 2
