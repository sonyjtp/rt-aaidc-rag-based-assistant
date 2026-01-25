"""Memory manager for the RAG assistant.

This module provides MemoryManager which selects and initializes a
conversation memory strategy based on configuration. It supports a
sliding-window implementation and falls back gracefully when optional
dependencies (PyYAML, langchain) are not available.
"""

from config import MEMORY_STRATEGIES_FPATH, MEMORY_STRATEGY
from logger import logger
from sliding_window_memory import SlidingWindowMemory

# Make PyYAML optional to avoid import errors in lightweight environments
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

# Try to import optional langchain memory classes at module import time so
# we avoid import-outside-toplevel lint warnings. If unavailable, set to None.
try:
    from langchain.memory import (  # type: ignore
        ConversationBufferMemory,
        ConversationSummaryMemory,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    ConversationSummaryMemory = None  # type: ignore
    ConversationBufferMemory = None  # type: ignore


class MemoryManager:
    """Manages conversation memory based on configured strategy."""

    def __init__(self, llm):
        """Initialize memory manager with the configured strategy."""
        self.strategy = MEMORY_STRATEGY
        self.config = self._load_memory_strategy_config()
        self.llm = llm
        self.memory = None
        self._initialize_memory()

    def _load_memory_strategy_config(self):
        """Load memory strategy configuration from YAML file.

        If PyYAML is not installed, return an empty configuration and log a
        warning so the application can continue running without memory.
        """
        if yaml is None:
            logger.warning("PyYAML not installed; memory strategies unavailable.")
            return {}

        try:
            with open(MEMORY_STRATEGIES_FPATH, "r", encoding="utf-8") as f:
                strategies = yaml.safe_load(f) or {}
            return strategies.get(self.strategy, {})
        except FileNotFoundError:
            logger.warning(
                f"Memory strategies config not found at {MEMORY_STRATEGIES_FPATH}"
            )
            return {}

    def _initialize_memory(self):
        """Initialize the appropriate memory strategy."""
        if self.strategy == "summarization_sliding_window":
            self._initialize_sliding_window_memory()
        elif self.strategy == "summarization":
            self._initialize_summarization_memory()
        elif self.strategy == "conversation_buffer_memory":
            self._initialize_buffer_memory()
        else:
            logger.warning("No memory strategy applied.")

    def _initialize_sliding_window_memory(self):
        """Initialize sliding window memory strategy."""
        window_size = self.config.get("parameters", {}).get("window_size", 20)
        memory_key = self.config.get("parameters", {}).get("memory_key", "chat_history")
        self.memory = SlidingWindowMemory(
            llm=self.llm,
            window_size=window_size,
            memory_key=memory_key,
        )

    def _initialize_summarization_memory(self):
        """Initialize ConversationSummaryMemory if available."""
        if ConversationSummaryMemory is None:
            logger.warning(
                "ConversationSummaryMemory not available. Falling back to no memory."
            )
            self.memory = None
            return

        memory_key = self.config.get("parameters", {}).get("memory_key", "chat_history")
        logger.info("ConversationSummaryMemory initialized")
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key=memory_key,
        )

    def _initialize_buffer_memory(self):
        """Initialize ConversationBufferMemory if available."""
        if ConversationBufferMemory is None:
            logger.warning(
                "ConversationBufferMemory not available. Falling back to no memory."
            )
            self.memory = None
            return

        logger.info("ConversationBufferMemory initialized")
        try:
            self.memory = ConversationBufferMemory()
        except TypeError:
            # Some langchain versions require different ctor args; treat this as
            # unavailability for our purposes.
            logger.warning(
                "ConversationBufferMemory parameters not supported. Falling back to no memory."
            )
            self.memory = None

    def add_message(self, input_text: str, output_text: str) -> None:
        """Add a message pair to memory."""
        if self.memory:
            try:
                self.memory.save_context({"input": input_text}, {"output": output_text})
            except (
                ValueError,
                AttributeError,
            ) as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error saving to memory: {e}")

    def get_memory_variables(self) -> dict:
        """Get current memory variables for the chain."""
        if self.memory:
            try:
                return self.memory.load_memory_variables({})
            except (
                ValueError,
                AttributeError,
            ) as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error loading memory variables: {e}")
                return {}
        return {}
