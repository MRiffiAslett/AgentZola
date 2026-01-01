"""
Logging module for WhiteFox generation.

Contains all logging-related functionality:
- Structured logging (logger.py)
- Sanity checking (sanity_checker.py)
- Log parsing (log_parser.py)
"""

print(f"[__init__.py] Starting import of generation.logging.logger")
from generation.logging.logger import WhiteFoxLogger
print(f"[__init__.py] Successfully imported WhiteFoxLogger")
from generation.logging.sanity_checker import run_sanity_check
from generation.logging.log_parser import extract_triggered_passes

__all__ = ['WhiteFoxLogger', 'run_sanity_check', 'extract_triggered_passes']

