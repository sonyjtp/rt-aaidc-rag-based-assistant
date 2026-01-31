"""Build system prompts from configuration and reasoning strategies."""

from langchain_core.prompts import ChatPromptTemplate

import config
from config import PROMPT_CONFIG_FPATH
from file_utils import load_yaml
from logger import logger
from reasoning_strategy_loader import ReasoningStrategyLoader


def build_system_prompts(
    reasoning_strategy: ReasoningStrategyLoader = None,
) -> list[str]:
    """Builds system prompts from the prompt configuration file and reasoning strategy.
    Steps:
    1. Load prompt configuration from YAML file
    2. Extract role, style/tone, output constraints, and output format
    3. If a reasoning strategy is provided and enabled, include its instructions

    Args:
        reasoning_strategy: An instance of ReasoningStrategyLoader to include
            reasoning strategy instructions in the system prompts. If None,
            a new instance will be created.

    Returns:
        List of system prompt strings.
    """
    prompt_configs = load_yaml(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs[config.PROMPT_CONFIG_NAME]
    if not system_prompt_config:
        msg = f"System prompt config for '{config.PROMPT_CONFIG_NAME}' not found"
        raise ValueError(msg)
    system_prompts = []

    if role := system_prompt_config.get("role", "A helpful AI assistant."):
        system_prompts.append(f"You are {role.strip().lower()}.\n")
        logger.debug("Added role to system prompts.")

    if tone := system_prompt_config.get("style_or_tone"):
        system_prompts.append(f"Adopt the following style or tone:\n{tone}")
        logger.debug("Added style/tone to system prompts.")

    if constraints := system_prompt_config.get("output_constraints"):
        system_prompts.append(f"Follow these output constraints:\n{constraints}")
        logger.debug("Added output constraints to system prompts.")

    if output_format := system_prompt_config.get("output_format"):
        system_prompts.append(f"Use the following output format:\n{output_format}")
        logger.debug("Added output format to system prompts.")

    # Add reasoning strategy instructions
    try:
        strategy_loader = reasoning_strategy or ReasoningStrategyLoader()
        if strategy_loader.get("enabled", False):
            strategy_instructions = strategy_loader.get("prompt_instructions", [])
            if strategy_instructions:
                reasoning_prompt = (
                    "Apply the following reasoning approach:\n"
                    + "\n".join(
                        f"- {instruction}" for instruction in strategy_instructions
                    )
                )
                system_prompts.append(reasoning_prompt)
                strategy_name = strategy_loader.get(
                    "name", strategy_loader.active_strategy
                )
                logger.debug(
                    f"Added reasoning strategy {strategy_name} to system prompts."
                )
        else:
            logger.warning(
                f"Reasoning strategy {config.REASONING_STRATEGY} is disabled."
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not load reasoning strategy: {e}")

    logger.info(
        "System prompts built with role, style, constraints, format, and reasoning."
    )
    return system_prompts


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
