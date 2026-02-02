"""Handles meta questions about the assistant itself."""
import re
from pathlib import Path
from typing import List, Optional, Tuple

from config import METAQUESTIONS_FPATH, PROMPT_CONFIG_FPATH
from error_messages import DEFAULT_NOT_KNOWN_ERROR_MESSAGE
from file_utils import load_yaml
from logger import logger
from readme_extractor import ReadmeExtractor


# pylint: disable=too-few-public-methods
class MetaPattern:
    """Represents a meta question pattern."""

    def __init__(self, pattern: str, kind: str, response: str = "", response_type: str = ""):
        self.pattern = pattern
        try:
            self.rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # fallback to literal match
            self.rx = re.compile(re.escape(pattern), re.IGNORECASE)
        self.kind = kind
        self.response = response
        self.response_type = response_type  # For readme_extract patterns


class PersonaHandler:
    """Handles meta questions about the assistant itself."""

    def __init__(self, config_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else None
        meta_cfg_primary = Path(METAQUESTIONS_FPATH)
        prompt_cfg = Path(PROMPT_CONFIG_FPATH)
        if cfg_path and cfg_path.exists():
            self.cfg_path = cfg_path
        elif meta_cfg_primary.exists():
            self.cfg_path = meta_cfg_primary
        else:
            self.cfg_path = prompt_cfg

        self.patterns: List[MetaPattern] = []
        self.allow_self_description = True
        self.default_meta_refusal = DEFAULT_NOT_KNOWN_ERROR_MESSAGE

        # Initialize README extractor for dynamic content
        try:
            self.readme_extractor = ReadmeExtractor()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not initialize README extractor: {e}")
            self.readme_extractor = None

        self._load()

    def _load(self):
        if not self.cfg_path.exists():
            return
        try:
            data = load_yaml(self.cfg_path) or {}
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to load config from {self.cfg_path}: {e}")
            return
        meta_questions = data.get("meta_questions", [])
        for item in meta_questions:
            pattern = item.get("pattern", "")
            kind = item.get("kind", "refuse")
            response = item.get("response", "")
            response_type = item.get("response_type", "")
            self.patterns.append(MetaPattern(pattern, kind, response, response_type))
        self.allow_self_description = bool(data.get("allow_self_description", True))
        self.default_meta_refusal = data.get("default_meta_refusal", self.default_meta_refusal)

    def is_meta_question(self, query: str) -> Optional[Tuple[str, str, str]]:
        """Return (kind, response, response_type) if query matches a meta pattern, else None."""
        if not query:
            return None
        for p in self.patterns:
            if p.rx.search(query):
                return p.kind, p.response or self.default_meta_refusal, p.response_type

        return None

    def handle_meta_question(self, query: str) -> Optional[str]:
        """Handle meta question if detected."""

        m = self.is_meta_question(query)
        if not m:
            return None
        kind, response, response_type = m

        if kind == "sensitive":
            return self.default_meta_refusal

        if kind == "describe":
            if self.allow_self_description:
                return response
            return self.default_meta_refusal

        if kind == "readme_extract":
            return self._get_readme_content(response_type)

        # default: refusal
        return self.default_meta_refusal

    def _get_readme_content(self, response_type: str) -> str:
        """Extract README content based on response_type.

        Args:
            response_type: Type of content to extract (tools_and_models, overview, etc.)

        Returns:
            Extracted README content or default error message
        """
        if not self.readme_extractor:
            logger.warning("README extractor not available")
            return "I couldn't access the documentation. Please try a different question."

        try:
            match response_type:
                case "tools_and_models":
                    return self.readme_extractor.get_tools_and_models()
                case "overview":
                    return self.readme_extractor.get_overview()
                case "architecture":
                    return self.readme_extractor.get_architecture()
                case "customization":
                    return self.readme_extractor.get_customization()
                case "quick_start":
                    return self.readme_extractor.get_quick_start()
                case "features":
                    return self.readme_extractor.get_features()
                case _:
                    logger.warning(f"Unknown response_type: {response_type}")
                    return "I couldn't find that information in the documentation."
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error extracting README content: {e}")
            return "I encountered an error while accessing the documentation. Please try again."
