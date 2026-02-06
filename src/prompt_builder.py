"""Build system prompts from configuration and reasoning strategies."""

from langchain_core.prompts import ChatPromptTemplate

import config
from app_constants import PROMPT_CONFIG_FPATH
from error_messages import REASONING_STRATEGY_MISSING
from file_utils import load_yaml
from log_manager import logger
from reasoning_strategy_loader import ReasoningStrategyLoader


def get_default_system_prompts() -> list[str]:
    """Returns a list of default system prompts for fallback scenarios."""
    return [
        "You are a helpful AI assistant.",
        "Answer questions based only on the provided documents.",
        "If you cannot find the answer in the documents, say so.",
        "Do not make up answers.",
        "Be concise and clear in your responses.",
        "If the question is out of scope, respond politely that you cannot assist with that topic.",
        "If the question is about your identity or capabilities, respond with a brief description of yourself.",
        "Greet the user politely if they start with a greeting.",
        "If the question is meta (e.g., about how you work), respond appropriately based on your design as a RAG "
        "assistant.",
        "If the question is ambiguous, ask for clarification.",
        "If the question is offensive or inappropriate, respond politely that you cannot assist with that request.",
        "If the question is a follow-up, use the conversation context to provide a relevant answer.",
    ]


class PromptBuilder:
    """Builds system prompts for the RAG assistant based on configuration"""

    def __init__(self, reasoning_strategy: ReasoningStrategyLoader = None) -> None:
        """Initialize the prompt builder.
        Steps:
        1. Load prompt configurations from YAML file
        2. Set the default system prompt configuration

        Features:
        - Supports role, style/tone, output constraints, and output format
        - Integrates reasoning strategy instructions if enabled
        - Provides method to create ChatPromptTemplate

        Args:
            reasoning_strategy: An instance of ReasoningStrategyLoader

        """
        self.prompt_configs = load_yaml(PROMPT_CONFIG_FPATH)
        self.reasoning_strategy = reasoning_strategy
        self.system_prompt_config = self.prompt_configs[config.PROMPT_CONFIG_NAME_DEFAULT]
        if not self.system_prompt_config:
            msg = f"System prompt config for '{config.PROMPT_CONFIG_NAME_DEFAULT}' not found"
            raise ValueError(msg)

    def build_system_prompts(self) -> list[str]:
        """Builds system prompts from the prompt configuration file and reasoning strategy.
        Steps:
        1. Initialize an empty list for system prompts
        2. Add role, style/tone, output constraints, and output format sections
        3. Add reasoning strategy instructions if enabled

        Returns:
            List of system prompt strings.
        """
        system_prompts = []
        self._add_prompt_sections(system_prompts)
        self._add_reasoning_strategy_instructions(system_prompts)
        return system_prompts

    def _add_reasoning_strategy_instructions(self, prompts: list[str]) -> None:
        """Add reasoning strategy instructions to system prompts if enabled.

        Args:
            prompts: List to append to
        """
        try:
            if self.reasoning_strategy.is_strategy_enabled():
                strategy_instructions = self.reasoning_strategy.get_strategy_instructions()
                if strategy_instructions:
                    reasoning_prompt = "Apply the following reasoning approach:\n" + "\n".join(
                        f"- {instruction}" for instruction in strategy_instructions
                    )
                    prompts.append(reasoning_prompt)
                    strategy_name = self.reasoning_strategy.get_strategy_name()
                    logger.debug(f"Added reasoning strategy {strategy_name} to system prompts.")
            else:
                logger.warning(f"Reasoning strategy {config.REASONING_STRATEGY} is disabled.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"{REASONING_STRATEGY_MISSING}: {e}")

    def _add_prompt_sections(self, prompts: list[str]) -> None:
        """Add multiple prompt sections based on config keys.

        Args:
            prompts: List to append to
        """
        keys_and_prefixes = [
            ("role", "You are"),
            ("style_or_tone", "Adopt the following style or tone:"),
            ("output_constraints", "Follow these output constraints:"),
            ("output_format", "Use the following output format:"),
        ]
        for key, prefix in keys_and_prefixes:
            self._add_prompt_section(prompts, key, prefix)

    def _add_prompt_section(self, prompts: list[str], key: str, prefix: str) -> None:
        """Helper to add a prompt section if the config key exists.

        Args:
            prompts: List to append to
            key: Config key to check
            prefix: Text prefix for the prompt
        """
        if value := self.system_prompt_config.get(key):
            if isinstance(value, list):
                value = "\n".join(str(v).strip() for v in value)
            else:
                value = value.strip()

            if key == "role":
                value = value.lower()
            prompts.append(f"{prefix}\n{value}" if "\n" not in prefix else f"{prefix}{value}")
            logger.debug(f"Added {key} to system prompts.")

    @staticmethod
    def create_prompt_template(system_prompts: list[str]) -> ChatPromptTemplate:
        """Create the prompt template from system prompts.

        Args:
            system_prompts: List of system prompt strings.

        Returns:
            ChatPromptTemplate instance.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", "\n".join(system_prompts)),
                (
                    "human",
                    """Previous conversation context:
                        {chat_history}
                    Context from documents:
                    {context}

                    Question: {question}""",
                ),
            ]
        )
