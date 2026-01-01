"""
XLA pass log parsing.

Extracts which optimization passes fired from TensorFlow XLA compilation logs.
"""

import re
from typing import Set


def extract_triggered_passes(log_text: str) -> Set[str]:
    passes = set()
    
    pattern = r'WHITEFOX_PASS_START[^\n]*\bpass=([^\s]+)'
    
    for match in re.finditer(pattern, log_text):
        pass_name = match.group(1)
        passes.add(pass_name)
    
    return passes

