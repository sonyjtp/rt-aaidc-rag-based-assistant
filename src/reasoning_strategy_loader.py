"""
Reasoning strategy loader and builder.
Loads reasoning strategy configurations and provides methods to apply them.
"""

import config
from file_utils import load_yaml
from logger import logger


class ReasoningStrategyLoader:
    """Loads and manages reasoning strategies from YAML configuration."""

    def __init__(self, config_path: str = None):
        """
        Initialize the reasoning strategy loader.

        Args:
            config_path: Path to reasoning_strategies.yaml configuration file
        """
        if config_path is None:
            config_path = config.REASONING_STRATEGIES_FPATH
        self.config_path = config_path
        self.strategies = load_yaml(config_path).get("reasoning_strategies", {})
        self.active_strategy = config.REASONING_STRATEGY
        logger.info(
            f"Reasoning strategy loader initialized with strategy: {self.active_strategy}"
        )

    def get_active_strategy(self) -> dict:
        """
        Get the currently active reasoning strategy configuration.

        Returns:
            Dictionary containing the active strategy configuration

        Raises:
            ValueError: If active strategy not found in configuration
        """
        if self.active_strategy not in self.strategies:
            raise ValueError(
                f"Reasoning strategy '{self.active_strategy}' not found in configuration. "
                f"Available strategies: {list(self.strategies.keys())}"
            )
        return self.strategies[self.active_strategy]

    def get_strategy_instructions(self) -> list[str]:
        """
        Get prompt instructions for the active reasoning strategy.

        Returns:
            List of instruction strings for the strategy
        """
        strategy = self.get_active_strategy()
        return strategy.get("prompt_instructions", [])

    def get_strategy_name(self) -> str:
        """Get the name of the active reasoning strategy."""
        strategy = self.get_active_strategy()
        return strategy.get("name", self.active_strategy)

    def get_strategy_description(self) -> str:
        """Get the description of the active reasoning strategy."""
        strategy = self.get_active_strategy()
        return strategy.get("description", "")

    def is_strategy_enabled(self) -> bool:
        """Check if the active strategy is enabled."""
        strategy = self.get_active_strategy()
        return strategy.get("enabled", False)

    def get_few_shot_examples(self) -> list[dict]:
        """
        Get few-shot examples if available in the active strategy.

        Returns:
            List of example dictionaries with 'question' and 'answer' keys
        """
        strategy = self.get_active_strategy()
        return strategy.get("examples", [])

    def get_all_enabled_strategies(self) -> list[str]:
        """
        Get all enabled strategies.

        Returns:
            List of enabled strategy names
        """
        enabled = []
        for name, strategy_config in self.strategies.items():
            if strategy_config.get("enabled", False):
                enabled.append(name)
        return enabled

    def build_strategy_prompt(self) -> str:
        """
        Build a complete prompt section for the active reasoning strategy.

        Returns:
            Formatted string with strategy name, description, and instructions
        """
        strategy = self.get_active_strategy()
        name = strategy.get("name", self.active_strategy)
        description = strategy.get("description", "")
        instructions = strategy.get("prompt_instructions", [])

        prompt_parts = [
            f"Reasoning Strategy: {name}",
            f"Description: {description}",
            "Instructions:",
        ]

        for i, instruction in enumerate(instructions, 1):
            prompt_parts.append(f"{i}. {instruction}")

        return "\n".join(prompt_parts)
