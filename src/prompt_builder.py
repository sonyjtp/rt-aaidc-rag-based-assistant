import config
from config import PROMPT_CONFIG_FPATH
from file_utils import load_yaml


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
    print("Building system prompts...")
    if role := system_prompt_config.get("role", "helpful AI assistant."):
        system_prompts.append(
            f"You are {role.strip().lower()}.\n"
        )
        print("...added role.")
    if tone := system_prompt_config.get("style_or_tone"):
        system_prompts.append(
            f"Adopt the following style or tone:\n{tone}"
        )
        print("...added style/tone.")
    if constraints := system_prompt_config.get("output_constraints"):
        system_prompts.append(
            f"Follow these output constraints:\n{constraints}"
        )
        print("...added output constraints.")
    if output_format := system_prompt_config.get("output_format"):
        system_prompts.append(
            f"Use the following output format:\n{output_format}"
        )
        print("...added output format.")
    print("✓ System prompts built")
    return system_prompts