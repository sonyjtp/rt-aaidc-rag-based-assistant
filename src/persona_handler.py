"""Handles meta questions about the assistant itself."""
import re
from pathlib import Path
from typing import List, Optional, Tuple

from app_constants import METAQUESTIONS_FPATH
from error_messages import META_QUESTION_CONFIG_ERROR, NO_RESULTS_ERROR_MESSAGE
from file_utils import load_yaml
from log_manager import logger
from readme_extractor import ReadmeExtractor


# pylint: disable=too-few-public-methods
class MetaPattern:
    """Represents a meta question pattern."""

    def __init__(self, pattern: str, kind: str, response: str = "", response_type: str = ""):
        """Initialize the meta pattern.

        Args:
            pattern: Regex pattern to match meta questions
            kind: Type of meta question (sensitive, describe, readme_extract)
            response: Predefined response for the meta question
            response_type: Type of response for readme_extract patterns
        """

        try:
            self.rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # fallback to literal match
            self.rx = re.compile(re.escape(pattern), re.IGNORECASE)
        self.kind = kind
        self.response = response
        self.response_type = response_type  # For readme_extract patterns


class PersonaHandler:
    """Handles meta questions about the assistant itself.

    Features:
    1. Load meta question patterns from a YAML configuration file.
    2. Detect if a user query matches any meta question patterns.
    3. Provide appropriate responses based on the type of meta question:
       - Refusal for sensitive questions.
       - Self-description if allowed.
       - Dynamic README content extraction for documentation-related questions.
    """

    def __init__(self):
        self.meta_question_config_path = Path(METAQUESTIONS_FPATH)
        self.patterns: List[MetaPattern] = []
        self.allow_self_description = True
        self.default_meta_refusal = NO_RESULTS_ERROR_MESSAGE

        # Initialize README extractor for dynamic content
        try:
            self.readme_extractor = ReadmeExtractor()
            logger.debug("README extractor initialized.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not initialize README extractor: {e}")
            self.readme_extractor = None

        # Load meta question patterns from config
        self._load()
        logger.debug("Meta question config loaded.")

    def _load(self):
        if not self.meta_question_config_path.exists():
            logger.error(
                f"""Meta question config  {self.meta_question_config_path} does not exist.
                    {META_QUESTION_CONFIG_ERROR}
                """
            )
            return
        try:
            meta_question_config = load_yaml(self.meta_question_config_path) or {}
            meta_questions = meta_question_config.get("meta_questions", [])
            for item in meta_questions:
                pattern = item.get("pattern", "")
                kind = item.get("kind", "refuse")
                response = item.get("response", "")
                response_type = item.get("response_type", "")
                self.patterns.append(MetaPattern(pattern, kind, response, response_type))
            self.allow_self_description = bool(meta_question_config.get("allow_self_description", True))
            self.default_meta_refusal = meta_question_config.get("default_meta_refusal", self.default_meta_refusal)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to load config from {self.meta_question_config_path}: {e}")
            logger.error(META_QUESTION_CONFIG_ERROR)
            return

    def is_meta_question(self, query: str) -> Optional[Tuple[str, str, str]]:
        """Return (kind, response, response_type) if query matches a meta pattern, else None."""
        if not query:
            return None
        for p in self.patterns:
            if p.rx.search(query):
                return p.kind, p.response or self.default_meta_refusal, p.response_type

        return None

    def handle_meta_question(self, query: str) -> Optional[str]:
        """Handle meta question if detected.

        Steps:
        1. Check if the query matches any meta question patterns.
        2. Based on the type of meta question:
           - For "sensitive" questions, return a refusal response.
           - For "describe" questions, return self-description if allowed; otherwise, refusal.
           - For "readme_extract" questions, extract and return relevant README content.
        3. If no patterns match, return None.

        Args:
            query: User's input query string

        Returns:
            Appropriate response string if meta question detected, else None
        """

        m = self.is_meta_question(query)
        if not m:
            return None
        kind, response, response_type = m

        if kind == "sensitive":
            logger.warning(f"Refusing to answer sensitive meta question: {query}")
            return self.default_meta_refusal

        if kind == "describe":
            logger.info(f"Handling self-description meta question: {query}")
            if self.allow_self_description:
                return response
            return self.default_meta_refusal

        if kind == "readme_extract":
            logger.info(f"Handling README extraction meta question: {query}")
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
            handlers = {
                "tools_and_models": self.readme_extractor.get_tools_and_models,
                "tools": self.readme_extractor.get_tools_and_models,
                "overview": self.readme_extractor.get_overview,
                "architecture": self.readme_extractor.get_architecture,
                "customization": self.readme_extractor.get_customization,
                "quick_start": self.readme_extractor.get_quick_start,
                "features": self.readme_extractor.get_features,
                "capabilities": self.readme_extractor.get_capabilities,
            }

            handler = handlers.get(response_type)
            if not handler:
                logger.warning(f"Unknown response_type: {response_type}")
                return NO_RESULTS_ERROR_MESSAGE

            return handler()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error extracting README content: {e}")
            return "I encountered an error while accessing the documentation. Please try again."
