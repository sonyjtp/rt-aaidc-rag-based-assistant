"""Utilities for loading and processing documents and configuration files."""
import os
from pathlib import Path
from typing import Iterator

# Make YAML and LangChain imports optional so linting doesn't fail in environments
# where these dependencies aren't installed. Provide lightweight fallbacks.
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from langchain_community.document_loaders import TextLoader
except Exception:  # pragma: no cover - optional dependency
    TextLoader = None

from logger import logger


# Lightweight fallback loader that mimics the minimal interface used in this module
class _SimpleDoc:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _SimpleTextLoader:
    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self) -> Iterator[_SimpleDoc]:
        with open(self._path, "r", encoding=self._encoding) as fh:
            yield _SimpleDoc(fh.read())


def _get_text_loader(path: str, encoding: str = "utf-8"):
    """Return a TextLoader-compatible object for the given path.

    Prefers the langchain_community loader when available, otherwise uses the
    simple fallback loader above.
    """
    if TextLoader is not None:
        return TextLoader(path, encoding=encoding)
    return _SimpleTextLoader(path, encoding=encoding)


def load_documents(folder: str, file_extns: str | tuple[str, ...] = ".txt") -> list[dict[str, str]]:
    """
    Load documents from text files using a TextLoader-like interface.

    The function reads all files in `folder` with matching extensions and
    returns a list of document dictionaries with keys: 'filename', 'title',
    'tags', and 'content'. Title is taken from the first non-empty line and
    an optional second line beginning with "Tags:" is parsed into `tags`.
    """
    documents: list[dict[str, str]] = []
    for filename in os.listdir(folder):
        if not filename.endswith(file_extns):
            # keep this message short to satisfy line length rules
            logger.debug("Skipping %s: extension mismatch", filename)
            continue
        path = os.path.join(folder, filename)
        try:
            loader = _get_text_loader(path, encoding="utf-8")
            for doc in loader.load():
                # Extract non-empty lines safely and within line-length limits
                raw = doc.page_content.strip()
                lines = raw.split("\n")
                content_lines = [ln.strip() for ln in lines if ln.strip()]

                # Title is the first non-empty line
                title = content_lines[0] if content_lines else filename

                # Tags: take second non-empty line if it starts with "Tags:"
                tags = ""
                if len(content_lines) > 1 and content_lines[1].startswith("Tags:"):
                    tags = content_lines[1].replace("Tags:", "").strip()

                documents.append({
                    "filename": filename,
                    "title": title,
                    "tags": tags,
                    "content": doc.page_content,
                })
        except IOError as e:
            logger.error("Error loading %s: %s", filename, e)
    return documents


def load_yaml(file_path: str | Path) -> dict:
    """Loads a YAML configuration file.

    If the `yaml` package is not installed, a RuntimeError is raised with a
    clear message explaining the missing dependency.
    """
    if yaml is None:  # pragma: no cover - environment without PyYAML
        raise RuntimeError(
            "PyYAML is required to load YAML files. Install it with `pip install pyyaml`."
        )

    file_path = Path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e
