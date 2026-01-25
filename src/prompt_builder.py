"""Build system prompts from configuration and reasoning strategies."""
import config
from config import PROMPT_CONFIG_FPATH
from file_utils import load_yaml
from logger import logger
from reasoning_strategy_loader import ReasoningStrategyLoader


def build_system_prompts() -> list[str]:
    """Builds system prompts from the prompt configuration file and reasoning strategy.

    Returns:
        List of system prompt strings.
    """
    prompt_configs = load_yaml(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs[config.PROMPT_CONFIG_NAME]
    if not system_prompt_config:
        msg = f"System prompt config for '{config.PROMPT_CONFIG_NAME}' not found"
        raise ValueError(msg)
    system_prompts = []
    logger.info("Building system prompts...")

    if role := system_prompt_config.get("role", "helpful AI assistant."):
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
        strategy_loader = ReasoningStrategyLoader()
        if strategy_loader.is_strategy_enabled():
            strategy_instructions = strategy_loader.get_strategy_instructions()
            if strategy_instructions:
                reasoning_prompt = (
                    "Apply the following reasoning approach:\n"
                    + "\n".join(
                        f"- {instruction}" for instruction in strategy_instructions
                    )
                )
                system_prompts.append(reasoning_prompt)
                strategy_name = strategy_loader.get_strategy_name()
                logger.debug(
                    f"Added reasoning strategy {strategy_name} to system prompts."
                )
        else:
            logger.warning(
                f"Reasoning strategy {config.REASONING_STRATEGY} is disabled."
            )
    except (AttributeError, ValueError) as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not load reasoning strategy: {e}")

    logger.info("System prompts built successfully")
    return system_prompts
