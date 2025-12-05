"""
XLA pass log parsing.

Extracts which optimization passes fired from TensorFlow XLA compilation logs.
"""

import re
from typing import Set


def extract_triggered_passes(log_text: str) -> Set[str]:
    """
    Extract pass names from WHITEFOX_PASS_START lines.
    
    Scans every line that starts with WHITEFOX_PASS_START,
    extracts the substring after pass= up to the next whitespace or end of line.
    
    Args:
        log_text: Full log text from TensorFlow execution.
        
    Returns:
        Set of pass names that were triggered (e.g. {"flatten-call-graph", "sharding-remover"}).
    """
    passes = set()
    
    # Pattern to match WHITEFOX_PASS_START lines and extract pass= value
    # Format: WHITEFOX_PASS_START pipeline=... pass=flatten-call-graph module=...
    pattern = r'WHITEFOX_PASS_START[^\n]*\bpass=([^\s]+)'
    
    for match in re.finditer(pattern, log_text):
        pass_name = match.group(1)
        passes.add(pass_name)
    
    return passes

