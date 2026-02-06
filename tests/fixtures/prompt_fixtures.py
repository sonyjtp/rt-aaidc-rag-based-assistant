from unittest.mock import MagicMock, patch

import pytest

from src.prompt_builder import PromptBuilder, get_default_system_prompts
from src.reasoning_strategy_loader import ReasoningStrategyLoader


@pytest.fixture
def prompt_builder_factory():
    """Fixture returning a PromptBuilder instance initialized with a real reasoning strategy."""
    # Maintain backward-compatibility: return an instance
    reasoning_strategy = ReasoningStrategyLoader()
    return PromptBuilder(reasoning_strategy=reasoning_strategy)


@pytest.fixture
def prompt_builder_instance():
    """If a test needs a factory-like behavior, use this helper to create instances.

    Example: pb = prompt_builder_instance()  # returns a new PromptBuilder
    """

    def _create(strategy: ReasoningStrategyLoader | None = None):
        if strategy is None:
            strategy = ReasoningStrategyLoader()
        return PromptBuilder(reasoning_strategy=strategy)

    return _create


@pytest.fixture
def prompts(prompt_builder_factory):
    """Fixture providing built system prompts for all tests."""
    pb = prompt_builder_factory
    return pb.build_system_prompts()


@pytest.fixture
def system_prompt_text(prompts):
    """Fixture providing the combined system prompt text."""
    return "\n".join(prompts)


@pytest.fixture
def prompt_text(prompts):
    """Fixture providing joined prompt text for assertion tests."""
    return "\n".join(prompts)


@pytest.fixture
def default_prompts():
    """Fixture providing default system prompts."""
    return get_default_system_prompts()


@pytest.fixture
def default_prompt_text(default_prompts):
    """Fixture providing joined default prompt text."""
    return "\n".join(default_prompts)


@pytest.fixture
def mock_strategy():
    """Fixture providing a mocked reasoning strategy."""
    strategy = MagicMock()
    strategy.is_strategy_enabled.return_value = True
    strategy.get_strategy_instructions.return_value = [
        "Test instruction 1",
        "Test instruction 2",
    ]
    strategy.get_strategy_name.return_value = "Test Strategy"
    return strategy


@pytest.fixture
def mock_yaml_config():
    """Fixture providing a mock YAML config dictionary."""
    return {
        "default": {
            "role": "Assistant",
            "style_or_tone": None,
            "output_constraints": None,
            "output_format": None,
        }
    }


@pytest.fixture
def patch_config_name():
    """Fixture that yields a function to patch PROMPT_CONFIG_NAME temporarily."""

    def _patch(name="default"):
        return patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", name)

    return _patch


@pytest.fixture
def patch_strategy_loader():
    """Fixture that yields a function to patch ReasoningStrategyLoader with an optional mock."""

    def _patch(mock_strategy: MagicMock | None = None):
        # Return a contextmanager that yields (mock_loader, mock_strategy)
        return patch(
            "src.prompt_builder.ReasoningStrategyLoader",
            return_value=(mock_strategy if mock_strategy is not None else MagicMock()),
        )

    return _patch


@pytest.fixture
def patch_yaml_loader():
    """Fixture that yields a function to patch load_yaml with a provided dict."""

    def _patch(config_dict):
        return patch("src.prompt_builder.load_yaml", return_value=config_dict)

    return _patch


@pytest.fixture
def patched_config():
    """Fixture that patches PROMPT_CONFIG_NAME to 'default'."""
    with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
        yield


@pytest.fixture
def patched_strategy(patch_strategy_loader):
    """Fixture that patches ReasoningStrategyLoader with a disabled strategy."""
    with patch("src.prompt_builder.ReasoningStrategyLoader") as mock_loader:
        default = MagicMock()
        default.is_strategy_enabled.return_value = False
        mock_loader.return_value = default
        yield mock_loader, default


@pytest.fixture
def patched_yaml(mock_yaml_config, patch_yaml_loader):
    """Fixture that patches load_yaml with default config."""
    with patch("src.prompt_builder.load_yaml", return_value=mock_yaml_config) as mock_yaml:
        yield mock_yaml


@pytest.fixture
def patched_config_and_yaml(mock_yaml_config, patch_config_name, patch_yaml_loader):
    """Fixture that patches both PROMPT_CONFIG_NAME and load_yaml."""
    with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
        with patch("src.prompt_builder.load_yaml", return_value=mock_yaml_config) as mock_yaml:
            yield mock_yaml


@pytest.fixture
def patched_all_builders(mock_yaml_config, patch_config_name, patch_strategy_loader, patch_yaml_loader):
    """Fixture that patches all builder dependencies: config, yaml, and strategy."""
    with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
        with patch("src.prompt_builder.load_yaml", return_value=mock_yaml_config):
            with patch("src.prompt_builder.ReasoningStrategyLoader") as mock_loader:
                default = MagicMock()
                default.is_strategy_enabled.return_value = False
                mock_loader.return_value = default
                yield mock_loader, default
