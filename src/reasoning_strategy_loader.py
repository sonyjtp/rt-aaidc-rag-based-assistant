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

    def get_active_strategy(self) -> dict:
        """
        Get the currently active reasoning strategy configuration.

        Returns:
            Dictionary containing the active strategy configuration

        Raises:
            ValueError: If active strategy not found in configuration
        """
        if self.active_strategy not in self.strategies:
            logger.warning(
                f"Reasoning strategy {self.active_strategy} not found in configuration"
            )
            raise ValueError(
                f"Reasoning strategy '{self.active_strategy}' not found in configuration. "
                f"Available strategies: {list(self.strategies.keys())}"
            )
        return self.strategies[self.active_strategy]

    def get(self, property_name: str, default=None):
        """
        Get a property from the active reasoning strategy.

        Args:
            property_name: The configuration key to retrieve
            default: Default value if key not found

        Returns:
            The requested property value or default
        """
        return self._get_strategy_value(property_name, default)

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
        prompt_parts = [
            f"Reasoning Strategy: {self.get('name', self.active_strategy)}",
            f"Description: {self.get('description', '')}",
            "Instructions:",
        ]

        for i, instruction in enumerate(self.get("prompt_instructions", []), 1):
            prompt_parts.append(f"{i}. {instruction}")

        return "\n".join(prompt_parts)

    def _get_strategy_value(self, key: str, default):
        """
        Get a value from the active strategy configuration.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Value from strategy configuration or default
        """
        if self.active_strategy not in self.strategies:
            logger.warning(
                f"Reasoning strategy {self.active_strategy} not found in configuration"
            )
            return default

        return self.strategies[self.active_strategy].get(key, default)
