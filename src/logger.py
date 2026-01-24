"""
Logging configuration for the RAG-based AI assistant.
Uses loguru for clean, structured logging.
"""

import os
from loguru import logger

# Configure logger
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag_assistant.log")

# Get log level from environment variable (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Validate log level
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in valid_levels:
    LOG_LEVEL = "INFO"

# Remove default handler
logger.remove()

# Add console handler with color
logger.add(
    lambda msg: print(msg, end=""),
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level=LOG_LEVEL
)

# Add file handler with rotation
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    level=LOG_LEVEL
)

__all__ = ["logger"]



