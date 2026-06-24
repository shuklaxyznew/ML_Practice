from typing import List, Dict, Any
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class SessionMemory:
    """
    In-process memory for a single agent workflow session.
    Stores the conversation and tool call history.
    Cleared when the session ends.
    """

    def __init__(self, limit: int = None):
        self.limit = limit or settings.session_memory_limit
        self._messages: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.limit:
            self._messages.pop(0)
        logger.debug(f"Session memory: added {role} message")

    def add_tool_call(self, tool_name: str, input_data: Any, output: str) -> None:
        self._tool_calls.append({
            "tool": tool_name,
            "input": input_data,
            "output": output,
        })
        logger.debug(f"Session memory: logged tool call to {tool_name}")

    def get_messages(self) -> List[Dict[str, Any]]:
        return self._messages.copy()

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return self._tool_calls.copy()

    def get_context_summary(self) -> str:
        if not self._messages and not self._tool_calls:
            return "No session history yet."

        lines = [f"Session has {len(self._messages)} messages "
                 f"and {len(self._tool_calls)} tool calls."]

        if self._tool_calls:
            tools_used = [tc["tool"] for tc in self._tool_calls]
            lines.append(f"Tools used: {', '.join(set(tools_used))}")

        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()
        self._tool_calls.clear()
        logger.info("Session memory cleared")