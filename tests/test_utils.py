"""
Shared utilities for tests.
"""


def clean_response(response):
    """Clean response by removing headers and separators."""
    lines = response.split("\n")
    cleaned_lines = []
    skip_next = False

    for line in lines:
        if line.strip().startswith("#"):
            skip_next = True
            continue
        if skip_next and all(c in "=-_" for c in line.strip()) and len(line.strip()) > 3:
            skip_next = False
            continue
        if cleaned_lines or line.strip():
            cleaned_lines.append(line)
        skip_next = False

    return "\n".join(cleaned_lines).strip()
