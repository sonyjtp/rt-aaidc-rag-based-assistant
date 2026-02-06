"""
Reasoning strategy loader and builder.
"""

import config
from app_constants import REASONING_STRATEGIES_FPATH
from file_utils import load_yaml
from log_manager import logger


class ReasoningStrategyLoader:
    """Loads and manages reasoning strategies from YAML configuration.
    Provides methods to retrieve and build prompts based on the active strategy.
    """

    def __init__(self):
        """
        Initialize the reasoning strategy loader. Sets the active strategy
        based on the configuration.


        """
        # store the requested strategy key so we can fall back to it when name is missing
        self.strategy_key = getattr(config, "REASONING_STRATEGY", None)
        # Allow exceptions from load_yaml to propagate (tests expect this behavior)
        strategy = self._get_active_strategy(self.strategy_key)
        if not strategy:
            logger.warning(f"Reasoning strategy '{self.strategy_key}' not found.")
            self.active_strategy = None
        else:
            self.active_strategy = strategy

    @staticmethod
    def _get_active_strategy(strategy_key: str) -> dict:
        """
        Get the currently active reasoning strategy configuration.

        Returns:
            Dictionary containing the active strategy configuration
        """
        strategies = load_yaml(REASONING_STRATEGIES_FPATH).get("reasoning_strategies", {})
        # Return None if not found (caller will handle warnings/errors)
        return strategies.get(strategy_key)

    def get_active_strategy(self) -> dict:
        """Public accessor for the active strategy configuration.

        Raises:
            ValueError: If no active strategy is loaded or configured.

        Returns:
            dict: active strategy configuration
        """
        if not self.active_strategy:
            raise ValueError(f"Reasoning strategy '{self.strategy_key}' not available")
        return self.active_strategy

    def get_strategy_instructions(self) -> list[str]:
        """
        Get prompt instructions for the active reasoning strategy.

        Returns:
            List of instruction strings for the strategy
        """
        if not self.active_strategy:
            return []
        return self.active_strategy.get("prompt_instructions", [])

    def get_strategy_name(self) -> str:
        """Get the name of the active reasoning strategy."""
        if not self.active_strategy:
            # fallback to strategy key when no active strategy
            return self.strategy_key
        return self.active_strategy.get("name") or self.strategy_key

    def get_strategy_description(self) -> str:
        """Get the description of the active reasoning strategy."""
        if not self.active_strategy:
            return ""
        return self.active_strategy.get("description", "")

    def is_strategy_enabled(self) -> bool:
        """Check if the active strategy is enabled."""
        if not self.active_strategy:
            return False
        return self.active_strategy.get("enabled", False)

    def get_few_shot_examples(self) -> list[dict]:
        """
        Get few-shot examples if available in the active strategy.

        Returns:
            List of example dictionaries with 'question' and 'answer' keys
        """
        if not self.active_strategy:
            return []
        return self.active_strategy.get("examples", [])

    def build_strategy_prompt(self) -> str:
        """
        Build a complete prompt section for the active reasoning strategy.

        Returns:
            Formatted string with strategy name, description, and instructions
        """
        name = (
            (self.active_strategy.get("name") if self.active_strategy else None)
            or self.strategy_key
            or "Unnamed Strategy"
        )
        description = self.active_strategy.get("description", "") if self.active_strategy else ""
        instructions = self.active_strategy.get("prompt_instructions", []) if self.active_strategy else []

        prompt_parts = [
            f"Reasoning Strategy: {name}",
            f"Description: {description}",
            "Instructions:",
        ]

        for i, instruction in enumerate(instructions, 1):
            prompt_parts.append(f"{i}. {instruction}")

        return "\n".join(prompt_parts)
