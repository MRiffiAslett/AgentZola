"""
Logging module for WhiteFox generation.

Contains all logging-related functionality:
- Structured logging (logger.py)
- Log parsing (log_parser.py)
"""

from generation.logging.logger import WhiteFoxLogger
from generation.logging.log_parser import extract_triggered_passes

__all__ = ['WhiteFoxLogger', 'extract_triggered_passes']

