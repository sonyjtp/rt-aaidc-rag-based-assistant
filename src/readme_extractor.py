"""Extract and cache README information for meta-question responses."""

import re
from pathlib import Path
from typing import Optional

from app_constants import README
from file_utils import load_text_file
from log_manager import logger


class ReadmeExtractor:
    """Extract key information from README.md for meta-question responses.

    Features:
    1. Load README content from specified path.
    2. Extract sections: Features, Overview, Architecture, Customization,
       Quick Start, Tools and Models, Capabilities.
    3. Cache extracted sections for efficient retrieval.
    4. Strip Markdown formatting and emojis from extracted text.
    """

    def __init__(self):
        """Initialize README extractor with path to README."""
        self.readme_path = Path(README)
        self.content = None
        self.features = None
        self.overview = None
        self.architecture = None
        self.customization = None
        self.quick_start = None
        self.tools = None
        self.capabilities = None

        if self.readme_path and self.readme_path.exists():
            self.content = load_text_file(self.readme_path)
            if self.content:
                logger.debug(f"Loaded README from {self.readme_path}")
            else:
                logger.error(
                    f"""Failed to load README from {self.readme_path}.
                        Meta-questions on README content will not work.
                    """
                )
        else:
            self.content = None

    def get_features(self) -> str:
        """Extract comprehensive features section from README.

        Returns:
            Complete features list including all subsections formatted as bullet points
        """
        if self.features:
            return self.features

        if not self.content:
            return "Features information not available."

        # Try multiple header variations for the Features section
        # This includes all subsections: Core RAG, Memory, LLM, Reasoning, Safety, User Interfaces
        headers_to_try = [
            "## ✨ Features",
            "## Features",
            "# Features",
        ]

        features_section = None
        for header in headers_to_try:
            features_section = self._extract_section(header)
            if features_section:
                logger.debug(f"Found features section with header: {header}")
                break

        if features_section:
            cleaned = self._strip_markdown(features_section)
            if cleaned and cleaned.strip():
                # Preserve or restore bullet formatting for readability
                self.features = self._format_as_bullets(cleaned)
            else:
                self.features = "Features information not available."
        else:
            logger.warning("Could not find Features section in README")
            self.features = "Features information not available."

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
        """Extract architecture and system design descriptions from README.

        Returns:
            Architecture description (text only, excluding diagrams)
        """
        if self.architecture:
            return self.architecture

        if not self.content:
            return "Architecture information not available."

        # Extract both System Architecture Overview and Data Flow Overview sections
        system_arch_desc = self._extract_architecture_description("System Architecture Overview:")
        data_flow_desc = self._extract_architecture_description("Data Flow Overview:")

        parts = []
        if system_arch_desc:
            parts.append("## System Architecture\n\n" + system_arch_desc)
        if data_flow_desc:
            parts.append("\n## Data Flow\n\n" + data_flow_desc)

        if parts:
            combined = "\n".join(parts)
            cleaned = self._strip_markdown(combined)
            self.architecture = cleaned if cleaned else "Architecture information not available."
        else:
            self.architecture = "Architecture information not available."

        return self.architecture

    def _extract_architecture_description(self, marker: str) -> Optional[str]:
        """Extract text description for architecture between markers.

        Stops at diagrams (lines starting with box-drawing characters) or section headers.

        Args:
            marker: The text marker to look for (e.g., "System Architecture Overview:")

        Returns:
            Text content until diagrams or next section, excluding ASCII art
        """
        if not self.content:
            return None

        lines = self.content.split("\n")
        start_idx = None
        end_idx = None

        # Find the marker
        for i, line in enumerate(lines):
            if marker in line:
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Find the end: stop at ASCII diagrams (┌, │, └, etc.) or section headers or "---"
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            # Stop at horizontal rule (---) or next section header
            if line.startswith("---") or (line.startswith("#") and line != ""):
                end_idx = i
                break
            # Stop at ASCII diagram lines (box-drawing characters)
            if line.startswith(("┌", "│", "└", "├", "┬", "┤", "┴", "┼")):
                end_idx = i
                break
            # Stop at code blocks
            if line.startswith("```"):
                end_idx = i
                break

        if end_idx is None:
            end_idx = len(lines)

        # Extract and clean the text
        section_content = "\n".join(lines[start_idx:end_idx]).strip()
        return section_content if section_content else None

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
            Quick start guide formatted as bullet points (markdown formatting removed, without code blocks)
        """
        if self.quick_start:
            return self.quick_start

        if not self.content:
            return "Quick start information not available."

        # Try multiple header variations
        headers = ["## 🚀 Quick Start", "## Quick Start", "## Getting Started"]

        quick_start_section = None
        for header in headers:
            quick_start_section = self._extract_quick_start_text(header)
            if quick_start_section:
                break

        if quick_start_section:
            cleaned = self._strip_markdown(quick_start_section)
            # Format as bullet points
            formatted = self._format_as_bullets(cleaned) if cleaned else "Quick start information not available."
            self.quick_start = formatted
        else:
            self.quick_start = "Quick start information not available."

        return self.quick_start

    def _extract_quick_start_text(self, section_header: str) -> Optional[str]:
        """Extract quick start section excluding code blocks and diagrams.

        Args:
            section_header: The section header to search for

        Returns:
            Text content of quick start excluding code blocks
        """
        if not self.content:
            return None

        lines = self.content.split("\n")
        start_idx = None
        end_idx = None

        # Find the header
        for i, line in enumerate(lines):
            if section_header in line:
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Find the end: stop at next same-level header or "---" separator
        header_level = len(section_header) - len(section_header.lstrip("#"))

        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith("#"):
                current_level = len(line) - len(line.lstrip("#"))
                if current_level <= header_level and line != "":
                    end_idx = i
                    break
            elif line.startswith("---"):
                end_idx = i
                break

        if end_idx is None:
            end_idx = len(lines)

        # Extract text but exclude code blocks
        section_lines = []
        in_code_block = False

        for i in range(start_idx, end_idx):
            line = lines[i]
            # Track code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue  # Skip the code block markers

            # Skip lines inside code blocks
            if in_code_block:
                continue

            section_lines.append(line)

        section_content = "\n".join(section_lines).strip()
        return section_content if section_content else None

    def _extract_section(self, section_header: str) -> Optional[str]:
        """Extract a section from README content.

        Extracts content from the specified header until the next same-level or higher-level header.
        For example, if searching for "## Features", it will include all "###" subsections until
        the next "##" or "#" header is found.

        Args:
            section_header: The section header to search for (e.g., "## Features")

        Returns:
            Section content between the header and next same-level/higher-level header, or None
        """
        if not self.content:
            return None

        lines = self.content.split("\n")
        start_idx = None
        end_idx = None

        # Find section start - look for exact or close match
        for i, line in enumerate(lines):
            if section_header in line:
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Determine the header level of the section we're extracting
        # Count leading # symbols to determine level
        header_level = len(section_header) - len(section_header.lstrip("#"))

        # Find next section header at same level or higher (fewer # symbols)
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith("#"):
                # Count # symbols in this line
                current_level = len(line) - len(line.lstrip("#"))
                # Stop if we find a header at same level or higher (fewer or equal #)
                if current_level <= header_level and line != "":
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

        # Remove keycap sequences (e.g., 1️⃣ is digit + FE0F + 20E3)
        # This regex matches a digit followed by variation selectors and keycap combining mark
        text = re.sub(r"[\d][\uFE0E\uFE0F]?\u20E3", "", text)

        # Remove emojis and icons (Unicode ranges for emoji characters)
        # Include variation selectors (U+FE0E, U+FE0F) and keycap combining mark (U+20E3)
        emoji_pattern = (
            r"[\U0001F000-\U0001F9FF][\uFE0E\uFE0F\u20E3]*|"
            r"[\u2600-\u27BF][\uFE0E\uFE0F\u20E3]*|"
            r"[\u2700-\u27BF][\uFE0E\uFE0F\u20E3]*|"
            r"[\u2640-\u2642][\uFE0E\uFE0F\u20E3]*"
        )
        text = re.sub(emoji_pattern, "", text)

        # Remove emoji modifier sequences and variation selectors that might be left behind
        text = re.sub(r"[\uFE0E\uFE0F\u20E3]+", "", text)

        # Remove code blocks (``` ... ```) including those with language specifiers
        text = re.sub(r"```[\w]*\n[\s\S]*?```", "", text)

        # Remove any remaining backticks
        text = text.replace("```", "")

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

    @staticmethod
    def _format_as_bullets(text: str) -> str:
        """Format text as bullet points for better readability.

        Converts plain text lines into bullet-pointed format with proper markdown
        line breaks so they display on separate lines in Streamlit.
        Assumes each line (or sentence) should be a bullet point.

        Args:
            text: Plain text to format

        Returns:
            Text formatted as bullet points with Markdown line breaks
        """
        if not text or not text.strip():
            return text

        lines = text.split("\n")
        bullet_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # If line already starts with bullet, keep it
            if stripped.startswith("•") or stripped.startswith("-"):
                bullet_lines.append("• " + stripped.lstrip("•-").strip())
            # Otherwise, add a bullet
            else:
                bullet_lines.append("• " + stripped)

        # Use double newlines for markdown paragraph breaks in Streamlit
        return "\n\n".join(bullet_lines) if bullet_lines else text
