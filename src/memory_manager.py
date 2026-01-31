"""Memory manager for the RAG assistant.

This module provides MemoryManager which selects and initializes a
conversation memory strategy based on configuration. It supports multiple
memory implementations and falls back gracefully when optional dependencies
are not available.
"""
from yaml import YAMLError

from config import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
    DEFAULT_SUMMARY_PROMPT,
    MEMORY_KEY_PARAM,
    MEMORY_PARAMETERS_KEY,
    MEMORY_STRATEGIES_FPATH,
    MEMORY_STRATEGY,
)
from file_utils import load_yaml
from logger import logger
from simple_buffer_memory import SimpleBufferMemory
from sliding_window_memory import SlidingWindowMemory
from summary_memory import SummaryMemory

# Make PyYAML optional to avoid import errors in lightweight environments
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


class MemoryManager:

    """Manages conversation memory based on configured strategy."""

    def __init__(self, llm):
        """Initialize memory manager with the configured strategy.
        Steps:
        1. Load memory strategy configuration from YAML file
        2. Initialize the appropriate memory strategy

        Args:
            llm: The language model instance to use for memory strategies
                 that require it.

        """
        self.strategy = MEMORY_STRATEGY or "none"
        self.config = self._load_memory_strategy_config()
        self.llm = llm
        self.memory = None
        self._initialize_memory()

    def _load_memory_strategy_config(self) -> dict:
        """
        Load memory strategy configuration from YAML file.
        Returns an empty dict if loading fails.
        """
        try:
            return load_yaml(MEMORY_STRATEGIES_FPATH).get(self.strategy, {})
        except (FileNotFoundError, RuntimeError, IOError, YAMLError):
            logger.warning("Unable to initialize memory strategy.")
            return {}

    def _initialize_memory(self):
        """Initialize the appropriate memory strategy.

        Supports:
        - summarization_sliding_window: Sliding window with summarization
        - simple_buffer: Simple in-memory buffer
        - summary: LLM-based running summary
        - none: No memory

        Falls back gracefully if initialization fails.
        """
        match self.strategy:
            case "summarization_sliding_window":
                self._initialize_sliding_window_memory()
            case "simple_buffer":
                self._initialize_simple_buffer_memory()
            case "summary":
                self._initialize_summary_memory()
            case "none":
                self.memory = None
            case _:
                logger.warning(
                    f"Unknown memory strategy: {self.strategy}. No memory applied."
                )
                self.memory = None

    def _initialize_sliding_window_memory(self):
        """Initialize sliding window memory strategy.

        This is the primary memory implementation as it's the most reliable
        and doesn't depend on external langchain memory classes that may not
        be available or may have compatibility issues.
        """
        try:
            parameters = self.config.get(MEMORY_PARAMETERS_KEY, {})
            window_size = parameters.get(
                "window_size", DEFAULT_MEMORY_SLIDING_WINDOW_SIZE
            )
            memory_key = parameters.get("memory_key", MEMORY_KEY_PARAM)
            self.memory = SlidingWindowMemory(
                llm=self.llm, window_size=window_size, memory_key=memory_key
            )
            logger.debug(
                f"SlidingWindowMemory initialized with window_size={window_size}"
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"SlidingWindowMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def _initialize_simple_buffer_memory(self):
        """Initialize simple buffer memory strategy.

        Stores conversation history in a simple in-memory buffer.
        No summarization or advanced features.
        """
        try:
            parameters = self.config.get(MEMORY_PARAMETERS_KEY, {})
            memory_key = parameters.get("memory_key", MEMORY_KEY_PARAM)
            max_messages = parameters.get("max_messages", DEFAULT_MAX_MESSAGES)
            self.memory = SimpleBufferMemory(
                memory_key=memory_key, max_messages=max_messages
            )
            logger.debug(
                f"SimpleBufferMemory initialized with max_messages={max_messages}"
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"SimpleBufferMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def _initialize_summary_memory(self):
        """Initialize summary memory strategy.

        Maintains a running summary of the conversation using the LLM.
        """
        try:
            parameters = self.config.get(MEMORY_PARAMETERS_KEY, {})
            memory_key = parameters.get("memory_key", MEMORY_KEY_PARAM)
            summary_prompt = parameters.get("summary_prompt", DEFAULT_SUMMARY_PROMPT)
            self.memory = SummaryMemory(
                llm=self.llm, memory_key=memory_key, summary_prompt=summary_prompt
            )
            logger.debug("SummaryMemory initialized")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"SummaryMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def add_message(self, input_text: str, output_text: str) -> None:
        """Add a message pair to memory."""
        if self.memory:
            try:
                self.memory.save_context(
                    inputs={"input": input_text}, outputs={"output": output_text}
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error saving to memory: {e}")

    def get_memory_variables(self) -> dict:
        """Get current memory variables for the chain."""
        if self.memory:
            try:
                return self.memory.load_memory_variables()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error loading memory variables: {e}")
                return {}
        return {}
