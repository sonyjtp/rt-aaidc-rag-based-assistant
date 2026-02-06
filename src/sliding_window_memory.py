"""
Sliding window memory implementation for conversation management.
Summarizes conversation in fixed-size windows to manage token usage efficiently.
"""

from collections import deque

from app_constants import CHAT_HISTORY
from config import DEFAULT_MEMORY_SLIDING_WINDOW_SIZE
from log_manager import logger


class SlidingWindowMemory:
    """Custom memory implementation using sliding window summarization."""

    def __init__(
        self,
        llm,
        window_size: int = DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
        memory_key: str = CHAT_HISTORY,
        summarization_prompt: str = None,
    ):
        """
        Initialize sliding window memory.
        Steps:
        1. Store LLM for summarization
        2. Set window size and memory key
        3. Initialize deque for message storage
        4. Initialize empty summary string
        5. Set summarization prompt (use default if not provided)

        Features:
        - Maintains a fixed-size window of recent messages
        - Summarizes messages when window is full
        - Combines summary with recent messages for context retrieval

        Args:
            llm: Language model for summarization
            window_size: Number of recent messages to keep before summarizing
            memory_key: Key for storing memory variables
            summarization_prompt: Custom prompt for summarization (optional)
        """
        self.llm = llm
        self.window_size = window_size
        self.memory_key = memory_key
        self.messages = deque(maxlen=window_size)
        self.summary = ""
        self.summarization_prompt = (
            summarization_prompt or "Summarize the key points from this conversation segment:\n\n{window_text}"
        )

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Add a message pair to the sliding window. If window is full, summarize and reset

        Args:
            inputs: Dictionary with user input text
            outputs: Dictionary with assistant output text
        """
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")
        self.messages.append({"input": input_text, "output": output_text})

        # When window is full, create a summary and reset
        if len(self.messages) == self.window_size:
            self._summarize_window()

    def _summarize_window(self) -> None:
        """Summarize the current window and reset for next window."""
        if not self.messages:
            return

        # Format messages for summarization
        window_text = "\n".join([f"User: {msg['input']}\nAssistant: {msg['output']}" for msg in self.messages])

        try:
            # Summarize using LLM with configured prompt
            prompt = self.summarization_prompt.format(window_text=window_text)
            response = self.llm.invoke(prompt)
            self.summary = response.content if hasattr(response, "content") else str(response)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error summarizing window: {e}")
            self.summary = window_text

        # Clear window for next iteration
        self.messages.clear()
        logger.info(f"Memory window shifted (summarized {self.window_size} messages)")

    def load_memory_variables(self) -> dict:  # pylint: disable=unused-argument
        """Get current memory state including summary and recent messages."""
        recent_messages = "\n".join([f"User: {msg['input']}\nAssistant: {msg['output']}" for msg in self.messages])

        memory_content = ""
        if self.summary:
            memory_content += f"Summary of previous conversation:\n{self.summary}\n\n"
        if recent_messages:
            memory_content += f"Recent messages:\n{recent_messages}"

        return {self.memory_key: memory_content}
