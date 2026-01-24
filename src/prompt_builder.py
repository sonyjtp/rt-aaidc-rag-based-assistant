import config
from config import PROMPT_CONFIG_FPATH
from file_utils import load_yaml
from logger import logger


def build_system_prompts() -> list[str]:
    """Builds system prompts from the prompt configuration file.

    Returns:
        List of system prompt strings.
    """
    prompt_configs = load_yaml(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs[config.PROMPT_CONFIG_NAME]
    if not system_prompt_config:
        raise ValueError(f"System prompt config for '{config.PROMPT_CONFIG_NAME}' not found")
    system_prompts = []
    logger.info("Building system prompts...")
    if role := system_prompt_config.get("role", "helpful AI assistant."):
        system_prompts.append(
            f"You are {role.strip().lower()}.\n"
        )
        logger.debug("Added role to system prompts.")
    if tone := system_prompt_config.get("style_or_tone"):
        system_prompts.append(
            f"Adopt the following style or tone:\n{tone}"
        )
        logger.debug("Added style/tone to system prompts.")
    if constraints := system_prompt_config.get("output_constraints"):
        system_prompts.append(
            f"Follow these output constraints:\n{constraints}"
        )
        logger.debug("Added output constraints to system prompts.")
    if output_format := system_prompt_config.get("output_format"):
        system_prompts.append(
            f"Use the following output format:\n{output_format}"
        )
        logger.debug("Added output format to system prompts.")
    logger.info("System prompts built successfully")
    return system_prompts