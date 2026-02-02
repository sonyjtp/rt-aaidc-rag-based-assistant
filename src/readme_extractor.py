"""Extract and cache README information for meta-question responses."""

import re
from pathlib import Path
from typing import Optional

from config import README
from file_utils import _get_text_loader
from logger import logger


class ReadmeExtractor:
    """Extract key information from README.md for meta-question responses."""

    def __init__(self, readme_path: Optional[str] = None):
        """Initialize README extractor with path to README.

        Args:
            readme_path: Path to README.md file. If None, searches for README.md
                        in the project root.
        """
        self.readme_path = self._find_readme(readme_path)
        self.content = None
        self.features = None
        self.overview = None
        self.architecture = None
        self.customization = None
        self.quick_start = None
        self.tools = None
        self.capabilities = None

        if self.readme_path and self.readme_path.exists():
            self._load_content()

    @staticmethod
    def _find_readme(custom_path: Optional[str]) -> Optional[Path]:
        """Find README.md file.

        Args:
            custom_path: Custom path to README.md if provided

        Returns:
            Path to README.md or None if not found
        """
        if custom_path:
            path = Path(custom_path)
            if path.exists():
                return path

        # Search in common locations
        search_paths = [
            Path(__file__).parent.parent / README,
            Path.cwd() / README,  # Current directory
            Path(__file__).parent / README,  # src directory
        ]

        for path in search_paths:
            if path.exists():
                logger.debug(f"Found README at: {path}")
                return path

        logger.warning(f"{README} not found in common locations")
        return None

    def _load_content(self) -> None:
        """Load and parse README content using file_utils."""
        try:
            loader = _get_text_loader(str(self.readme_path))
            for doc in loader.load():
                self.content = doc.page_content
            logger.debug(f"Loaded README from {self.readme_path}")
        except Exception as e:
            logger.error(f"Failed to load README: {e}")
            self.content = None

    def get_features(self) -> str:
        """Extract features section from README.

        Returns:
            Features list formatted as string (markdown formatting removed)
        """
        if self.features:
            return self.features

        if not self.content:
            return "Features information not available."

        # Try multiple header variations
        headers = ["## ✨ Features", "## Features", "### Core RAG Capabilities"]

        features_section = None
        for header in headers:
            features_section = self._extract_section(header)
            if features_section:
                break

        self.features = (
            self._strip_markdown(features_section) if features_section else "Features information not available."
        )
        return self.features

    def get_tools_and_models(self) -> str:
        """Extract tools, models, and LLM information from README.

        Returns:
            Information about models and tools used (markdown formatting removed)
        """
        if self.tools:
            return self.tools

        if not self.content:
            return "Tools and models information not available."

        # Extract LLM Integration section
        llm_section = self._extract_section("### LLM Integration")
        memory_section = self._extract_section("### Memory Management")
        reasoning_section = self._extract_section("### Reasoning Strategies")

        parts = []
        if llm_section:
            parts.append("LLM Providers:\n" + self._strip_markdown(llm_section))
        if memory_section:
            parts.append("\nMemory Management:\n" + self._strip_markdown(memory_section))
        if reasoning_section:
            parts.append("\nReasoning Strategies:\n" + self._strip_markdown(reasoning_section))

        self.tools = "\n".join(parts) if parts else "Tools information not available."
        return self.tools

    def get_capabilities(self) -> str:
        """Extract assistant capabilities from README.

        Returns:
            Capabilities description (markdown formatting removed)
        """
        if self.capabilities:
            return self.capabilities

        if not self.content:
            return "Capabilities information not available."

        # Extract core RAG capabilities and feature sections
        core_rag = self._extract_section("### Core RAG Capabilities")
        if core_rag:
            self.capabilities = "Core Capabilities:\n" + self._strip_markdown(core_rag)
            return self.capabilities

        return "Capabilities information not available."

    def get_overview(self) -> str:
        """Extract overview section from README.

        Returns:
            Overview of the project (markdown formatting removed)
        """
        if self.overview:
            return self.overview

        if not self.content:
            return "Overview not available."

        # Try multiple header variations
        headers = ["## 🎯 Overview", "## Overview"]

        for header in headers:
            overview_section = self._extract_section(header)
            if overview_section:
                self.overview = self._strip_markdown(overview_section)
                return self.overview

        self.overview = "Overview not available."
        return self.overview

    def get_architecture(self) -> str:
        """Extract architecture and system design from README.

        Returns:
            Architecture description (markdown formatting removed)
        """
        if self.architecture:
            return self.architecture

        if not self.content:
            return "Architecture information not available."

        # Try multiple header variations
        headers = [
            "## 🏗️ Project Architecture",
            "## Project Architecture",
            "### System Architecture",
        ]

        arch_section = None
        for header in headers:
            arch_section = self._extract_section(header)
            if arch_section:
                break

        if arch_section:
            cleaned = self._strip_markdown(arch_section)
            # Limit to first 1500 chars for brevity
            self.architecture = cleaned[:1500] + "..." if len(cleaned) > 1500 else cleaned
            return self.architecture

        self.architecture = "Architecture information not available."
        return self.architecture

    def get_customization(self) -> str:
        """Extract customization options from README.

        Returns:
            Customization guide (markdown formatting removed)
        """
        if self.customization:
            return self.customization

        if not self.content:
            return "Customization information not available."

        # Try multiple header variations
        headers = [
            "## 🎛️ Customization Guide",
            "## Customization Guide",
            "## Customization",
        ]

        for header in headers:
            custom_section = self._extract_section(header)
            if custom_section:
                self.customization = self._strip_markdown(custom_section)
                return self.customization

        self.customization = "Customization information not available."
        return self.customization

    def get_quick_start(self) -> str:
        """Extract quick start instructions from README.

        Returns:
            Quick start guide (markdown formatting removed)
        """
        if self.quick_start:
            return self.quick_start

        if not self.content:
            return "Quick start information not available."

        # Try multiple header variations
        headers = ["## 🚀 Quick Start", "## Quick Start", "## Getting Started"]

        for header in headers:
            quick_start = self._extract_section(header)
            if quick_start:
                self.quick_start = self._strip_markdown(quick_start)
                return self.quick_start

        self.quick_start = "Quick start information not available."
        return self.quick_start

    def _extract_section(self, section_header: str) -> Optional[str]:
        """Extract a section from README content.

        Args:
            section_header: The section header to search for (e.g., "## Features")

        Returns:
            Section content between the header and next header, or None
        """
        if not self.content:
            return None

        lines = self.content.split("\n")
        start_idx = None
        end_idx = None

        # Find section start
        for i, line in enumerate(lines):
            if section_header in line:
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Find next section header (line starting with #)
        for i in range(start_idx, len(lines)):
            if lines[i].strip().startswith("#") and lines[i].strip() != "":
                end_idx = i
                break

        if end_idx is None:
            end_idx = len(lines)

        # Join lines and clean up
        section_content = "\n".join(lines[start_idx:end_idx]).strip()
        return section_content if section_content else None

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip markdown formatting and emojis from text.

        Removes:
        - Bold (**text** or __text__)
        - Italic (*text* or _text_)
        - Code blocks (```...```)
        - Inline code (`text`)
        - Headers (#, ##, etc.)
        - Links [text](url)
        - Horizontal rules (---, ***, ___)
        - Lists (-, *, +)
        - Emoji icons (🤖, 🎯, ✨, etc.)

        Args:
            text: Markdown formatted text

        Returns:
            Plain text with markdown formatting and emojis removed
        """
        if not text:
            return text

        # Remove emojis and icons (Unicode ranges for emoji characters)
        text = re.sub(
            r"[\U0001F000-\U0001F9FF]|[\u2600-\u27BF]|[\u2700-\u27BF]|[\u2640-\u2642]",
            "",
            text,
        )

        # Remove code blocks (``` ... ```)
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Remove inline code (`text`)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove bold (**text** or __text__)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)

        # Remove italic (*text* or _text_)
        # Be careful not to remove underscores in the middle of words
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove markdown links: [text](url) -> text
        pattern = r"\[(.+?)\]" + r"\(.+?\)"
        text = re.sub(pattern, r"\1", text)

        # Remove headers (# Header)
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

        # Remove horizontal rules (---, ***, ___)
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

        # Remove list markers (-, *, +) at start of lines
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

        # Clean up multiple blank lines
        text = re.sub(r"\n\n+", "\n\n", text)

        # Clean up extra spaces left by emoji removal
        text = re.sub(r"  +", " ", text)

        return text.strip()
