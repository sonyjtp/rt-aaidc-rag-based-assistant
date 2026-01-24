# src/memory_manager.py
import yaml
from config import MEMORY_STRATEGIES_FPATH, MEMORY_STRATEGY
from sliding_window_memory import SlidingWindowMemory


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
        """Load memory strategy configuration from YAML file."""
        try:
            with open(MEMORY_STRATEGIES_FPATH, 'r') as f:
                strategies = yaml.safe_load(f)
            return strategies.get(self.strategy, {})
        except FileNotFoundError:
            print(f"⚠ Memory strategies config not found at {MEMORY_STRATEGIES_FPATH}")
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
            print(f"⚠ No memory strategy applied.")
            self.memory = None

    def _initialize_sliding_window_memory(self):
        """Initialize sliding window memory strategy."""
        window_size = self.config.get("parameters", {}).get("window_size", 20)
        memory_key = self.config.get("parameters", {}).get("memory_key", "chat_history")
        self.memory = SlidingWindowMemory(
            llm=self.llm,
            window_size=window_size,
            memory_key=memory_key
        )

    def _initialize_summarization_memory(self):
        """Lazy import and initialize ConversationSummaryMemory."""
        try:
            from langchain.memory import ConversationSummaryMemory
            memory_key = self.config.get("parameters", {}).get("memory_key", "chat_history")
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key=memory_key
            )
        except ImportError:
            print("⚠ ConversationSummaryMemory not available. Falling back to no memory.")
            self.memory = None

    def _initialize_buffer_memory(self):
        """Lazy import and initialize ConversationBufferMemory."""
        try:
            from langchain.memory import ConversationBufferMemory
            self.memory = ConversationBufferMemory()
        except (ImportError, TypeError):
            print("⚠ ConversationBufferMemory not available or parameters not supported. Falling back to no memory.")
            self.memory = None

    def add_message(self, input_text: str, output_text: str) -> None:
        """Add a message pair to memory."""
        if self.memory:
            try:
                self.memory.save_context(
                    {"input": input_text},
                    {"output": output_text}
                )
            except Exception as e:
                print(f"⚠ Error saving to memory: {e}")

    def get_memory_variables(self) -> dict:
        """Get current memory variables for the chain."""
        if self.memory:
            try:
                return self.memory.load_memory_variables({})
            except Exception as e:
                print(f"⚠ Error loading memory variables: {e}")
                return {}
        return {}
