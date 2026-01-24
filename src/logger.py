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

# Remove default handler
logger.remove()

# Add console handler with color
logger.add(
    lambda msg: print(msg, end=""),
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG"
)

# Add file handler with rotation
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    level="DEBUG"
)

__all__ = ["logger"]

