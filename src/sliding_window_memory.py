"""
Sliding window memory implementation for conversation management.
Summarizes conversation in fixed-size windows to manage token usage efficiently.
"""

from collections import deque

from logger import logger


class SlidingWindowMemory:
    """Custom memory implementation using sliding window summarization."""

    def __init__(self, llm, window_size: int = 5, memory_key: str = "chat_history"):
        """
        Initialize sliding window memory.

        Args:
            llm: Language model for summarization
            window_size: Number of recent messages to keep before summarizing
            memory_key: Key for storing memory variables
        """
        self.llm = llm
        self.window_size = window_size
        self.memory_key = memory_key
        self.messages = deque(maxlen=window_size)
        self.summary = ""

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Add a message pair to the sliding window."""
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
        window_text = "\n".join(
            [
                f"User: {msg['input']}\nAssistant: {msg['output']}"
                for msg in self.messages
            ]
        )

        try:
            # Summarize using LLM
            prompt = f"Summarize the key points from this conversation segment:\n\n{window_text}"
            response = self.llm.invoke(prompt)
            self.summary = (
                response.content if hasattr(response, "content") else str(response)
            )
        except (
            AttributeError,
            ValueError,
        ) as e:  # pylint: disable=broad-exception-caught
            logger.error(
                f"Error summarizing window: {e}",
            )
            self.summary = window_text

        # Clear window for next iteration
        self.messages.clear()
        logger.info(f"Memory window shifted (summarized  {self.window_size} messages)")

    def load_memory_variables(self) -> dict:  # pylint: disable=unused-argument
        """Get current memory state including summary and recent messages."""
        recent_messages = "\n".join(
            [
                f"User: {msg['input']}\nAssistant: {msg['output']}"
                for msg in self.messages
            ]
        )

        memory_content = ""
        if self.summary:
            memory_content += f"Summary of previous conversation:\n{self.summary}\n\n"
        if recent_messages:
            memory_content += f"Recent messages:\n{recent_messages}"

        return {self.memory_key: memory_content}
