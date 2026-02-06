"""Summary memory implementation.

Maintains a running summary of the conversation using the LLM.
"""
from config import DEFAULT_SUMMARY_INTERVAL
from log_manager import logger


class SummaryMemory:
    """Memory that maintains a running summary of conversation."""

    def __init__(
        self,
        llm,
        memory_key: str = "chat_history",
        summary_prompt: str = "Summarize the conversation so far in a few sentences.",
        update_interval: int = DEFAULT_SUMMARY_INTERVAL,
    ):
        """Initialize summary memory.

        Args:
            llm: Language model instance for generating summaries
            memory_key: Key to use for storing memory variables
            summary_prompt: Prompt to use for generating summaries
        """
        self.llm = llm
        self.memory_key = memory_key
        self.summary_prompt = summary_prompt
        self.update_interval = update_interval
        self.summary = ""
        self.message_count = 0

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Save input and output, update running summary.

        Args:
            inputs: Dictionary containing input text with key "input"
            outputs: Dictionary containing output text with key "output"
        """
        try:
            input_text = inputs.get("input", "")
            output_text = outputs.get("output", "")

            self.message_count += 1

            # Update summary if the update interval is reached
            if self.message_count % self.update_interval == 0:
                self._update_summary(input_text, output_text)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error saving to summary memory: {e}")

    def _update_summary(self, input_text: str, output_text: str) -> None:
        """Update the running summary with new messages.

        Args:
            input_text: Latest user input
            output_text: Latest assistant output
        """
        try:
            if not self.llm:
                return

            # Create prompt for summarization
            current_exchange = f"User: {input_text}\nAssistant: {output_text}"

            if self.summary:
                prompt = f"""Previous summary: {self.summary}
                New exchange:\n{current_exchange}\n\n
                {self.summary_prompt}"""
            else:
                prompt = f"Exchange:\n{current_exchange}\n\n{self.summary_prompt}"

            # Call LLM to generate summary (simplified - direct call)
            try:
                self.summary = self.llm.invoke(prompt)
            except (
                ValueError,
                TypeError,
                AttributeError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ) as e:
                logger.warning(f"Unable to summarize. LLM invocation failed: {e}")
                # Fallback: keep existing summary or create simple one
                if not self.summary:
                    self.summary = f"Conversation with {self.message_count} messages"
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not update summary: {e}")

    def load_memory_variables(self) -> dict:
        """Load memory variables.

        Returns:
            Dictionary with memory key and current summary
        """
        try:
            return {self.memory_key: self.summary or f"Conversation with {self.message_count} messages"}
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading memory variables: {e}")
            return {self.memory_key: ""}
