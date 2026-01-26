"""String formatting utilities."""


def format_tags(tags: list[str] | str | None) -> str:
    """
    Format tags properly for storage in metadata.

    Args:
        tags: Tags as either a list of strings, a single string, or None

    Returns:
        Comma-separated string of tags, or empty string if input is None or invalid
    """
    match tags:
        case str():
            return tags
        case list():
            # Convert all elements to strings and filter out empty/falsy values
            valid_tags = [str(tag).strip() for tag in tags if tag and str(tag).strip()]
            return ",".join(valid_tags)
        case _:
            return ""
