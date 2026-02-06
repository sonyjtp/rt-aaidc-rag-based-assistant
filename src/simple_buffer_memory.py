"""Simple buffer memory implementation.

Stores conversation history in a simple buffer without summarization.
"""

from collections import deque

from log_manager import logger


class SimpleBufferMemory:
    """Simple in-memory buffer for storing conversation history."""

    def __init__(self, memory_key: str = "chat_history", max_messages: int = 50):
        """Initialize simple buffer memory.

        Args:
            memory_key: Key to use for storing memory variables
            max_messages: Maximum number of messages to store
        """
        self.memory_key = memory_key
        self.max_messages = max_messages
        self.buffer = deque(maxlen=max_messages)

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Save input and output to memory buffer.

        Args:
            inputs: Dictionary containing input text with key "input"
            outputs: Dictionary containing output text with key "output"
        """
        try:
            input_text = inputs.get("input", "")
            output_text = outputs.get("output", "")

            if input_text:
                self.buffer.append(f"User: {input_text}")
            if output_text:
                self.buffer.append(f"Assistant: {output_text}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error saving to buffer memory: {e}")

    def load_memory_variables(self) -> dict:
        """Load memory variables from buffer.

        Returns:
            Dictionary with memory key and formatted chat history
        """
        try:
            chat_history = "\n".join(self.buffer)
            return {self.memory_key: chat_history}
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading memory variables: {e}")
            return {self.memory_key: ""}
