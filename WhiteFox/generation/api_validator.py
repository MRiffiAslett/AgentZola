"""
TensorFlow API validator.

Detects invalid or non-existent TensorFlow APIs in generated code to prevent
runtime errors from using APIs that don't exist in the current TensorFlow version.
"""

import re
from typing import List, Set, Tuple


# Known invalid APIs that commonly appear in generated code
INVALID_APIS = {
    # Distribution strategies
    'tf.distribute.experimental.collective_all_reduce_strategy',
    'tf.distribute.experimental.CollectiveAllReduceStrategy',
    'tf.distribute.CollectiveCommunicator',
    
    # Raw ops that don't exist
    'tf.raw_ops.AllReduce',
    'tf.raw_ops.AllGather',
    
    # Other invalid patterns
    'tf.distribute.get_replica_context().values',
    'tf.distribute.HParams',
}

# Patterns to detect invalid API usage
INVALID_PATTERNS = [
    r'tf\.distribute\.experimental\.collective_all_reduce_strategy',
    r'tf\.distribute\.experimental\.CollectiveAllReduceStrategy',
    r'tf\.distribute\.CollectiveCommunicator',
    r'tf\.raw_ops\.AllReduce',
    r'tf\.raw_ops\.AllGather',
    r'\.collective_all_reduce_strategy',
    r'CollectiveAllReduceStrategy',
    r'CollectiveCommunicator',
]


def validate_tensorflow_apis(code: str) -> Tuple[bool, List[str]]:
    """
    Validate that code doesn't use invalid TensorFlow APIs.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for invalid API strings
    for invalid_api in INVALID_APIS:
        if invalid_api in code:
            errors.append(f"Invalid API detected: {invalid_api}")
    
    # Check for invalid patterns
    for pattern in INVALID_PATTERNS:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            errors.append(f"Invalid API pattern detected: {pattern} (found: {matches[0]})")
    
    # Check for common problematic patterns
    # StrategyBase.reduce() without axis
    if re.search(r'\.reduce\(\)\s*$', code, re.MULTILINE):
        errors.append("StrategyBase.reduce() called without required 'axis' argument")
    
    # OneDeviceStrategy._default_device (doesn't exist)
    if '_default_device' in code and 'OneDeviceStrategy' in code:
        errors.append("OneDeviceStrategy._default_device doesn't exist")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def filter_invalid_code(code: str) -> Tuple[str, List[str]]:
    """
    Filter out code with invalid APIs, returning cleaned code and errors.
    
    Returns:
        (cleaned_code, list_of_errors)
    """
    is_valid, errors = validate_tensorflow_apis(code)
    
    if is_valid:
        return code, []
    
    # If invalid, return empty code with errors
    # The caller can decide whether to reject or attempt to fix
    return "", errors

