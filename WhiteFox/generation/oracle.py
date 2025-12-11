"""
Bug detection oracles for WhiteFox.

Applies oracles to detect bugs in test execution results, including
crashes, status mismatches, and result inconsistencies (miscompilation).

This implementation matches the original WhiteFox oracle logic.
"""

from typing import List, Tuple, Optional
from enum import IntEnum, auto

import numpy as np
import tensorflow as tf

from AgentZola.WhiteFox.domain.harness import ExecutionResult, BugReport


class ResType(IntEnum):
    NaiveFail = auto()
    XLAFail = auto()
    ACFail = auto()
    Naive_XLAFail = auto()
    Naive_ACFail = auto()
    XLA_ACFail = auto()
    AllFail = auto()
    AllPass = auto()
    XLA_ACDiff = auto()
    Naive_ACDiff = auto()
    NaiveXLADiff = auto()
    AllDiff = auto()
    AllDiff_Rand = auto()
    AllDiff_LessLikely = auto()
    AllDiff_TypeMismatch = auto()


class DataType(IntEnum):
    Float = auto()
    Bool = auto()
    Int = auto()
    Str = auto()
    Null = auto()
    Tuple = auto()
    List = auto()
    TFTensor = auto()
    KerasTensor = auto()
    Unknown = auto()


FAIL_MAPPING = {
    "[1, 0, 0]": ResType.NaiveFail,
    "[0, 1, 0]": ResType.XLAFail,
    "[0, 0, 1]": ResType.ACFail,
    "[1, 1, 0]": ResType.Naive_XLAFail,
    "[1, 0, 1]": ResType.Naive_ACFail,
    "[0, 1, 1]": ResType.XLA_ACFail,
    "[1, 1, 1]": ResType.AllFail,
    "[0, 0, 0]": ResType.AllPass,
}


def get_type(output_data) -> DataType:
    if output_data is None:
        return DataType.Null
    elif isinstance(output_data, bool):
        return DataType.Bool
    elif isinstance(output_data, int):
        return DataType.Int
    elif isinstance(output_data, str):
        return DataType.Str
    elif isinstance(output_data, float):
        return DataType.Float
    elif isinstance(output_data, tuple):
        return DataType.Tuple
    elif isinstance(output_data, list):
        return DataType.List
    elif isinstance(output_data, tf.Tensor):
        return DataType.TFTensor
    elif hasattr(tf.keras.backend, 'is_keras_tensor') and tf.keras.backend.is_keras_tensor(output_data):
        return DataType.KerasTensor
    else:
        return DataType.Unknown


def is_equal(x, y) -> Tuple[bool, Optional[str]]:
    x_type, y_type = get_type(x), get_type(y)
    
    if x_type != y_type and not (x_type in [DataType.List, DataType.Tuple] and 
                                  y_type in [DataType.List, DataType.Tuple]):
        try:
            equal = np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True)
            return equal, "Value mismatch: {} vs {}".format(x, y)
        except:
            return False, "Type mismatch: {} vs {}".format(str(x_type), str(y_type))
    
    if x_type in [DataType.Int, DataType.Bool, DataType.Null, DataType.Str]:
        return x == y, "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.Float:
        return abs(x - y) < 1e-2, "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.TFTensor:
        return np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True), \
               "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.KerasTensor:
        return np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True), \
               "Value mismatch: {} vs {}".format(x, y)
    elif x_type in [DataType.List, DataType.Tuple]:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for i in range(len(x)):
            equal, msg = is_equal(x[i], y[i])
            if not equal:
                return False, msg
        return True, None
    else:
        return False, "Unsupported type: {} <-- {}".format(x_type, type(x))


def check_code_randomness(code: str) -> bool:
    if "tf.random" in code:
        return True
    if "dropout" in code.lower():
        return True
    return False


def check_less_possible_bug(code: str) -> bool:
    if "tf.cast" in code:
        return True
    return False


def value_diff_type(code: str, msg: str) -> ResType:
    if check_code_randomness(code):
        return ResType.AllDiff_Rand
    elif check_less_possible_bug(code):
        return ResType.AllDiff_LessLikely
    elif "Type mismatch" in msg:
        return ResType.AllDiff_TypeMismatch
    return ResType.AllDiff


def is_allowed_err(error) -> bool:
    error = str(error)
    allowed_errors = [
        'tf.function only supports singleton tf.Variables created on the first call',
        'Using a symbolic `tf.Tensor` as a Python `bool` is not allowed',
        'len is not well defined for a symbolic Tensor',
        'To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape.',
        'Python functions must return zero or more Tensors or ExtensionTypes or None values',
        'out of scope and cannot be used here',
        "This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.",
        "'SymbolicTensor' object has no attribute",
        "Iterating over a symbolic `tf.Tensor` is not allowed",
        "We failed to lift variable creations out of this tf.functio",
        "Attempting to capture an EagerTensor without building a function",
    ]
    for err in allowed_errors:
        if err in error:
            return True
    return False


def check_oracles(
    result: ExecutionResult,
    test_code: Optional[str] = None
) -> List[BugReport]:
    bug_reports = []
    test_id = result.test_file.stem
    optimizations_triggered = list(result.triggered_passes)
    
    fail = [0, 0, 0]
    allowed = [0, 0, 0]
    outputs = []
    
    if not result.runtime_success_naive:
        fail[0] = 1
        error_msg = result.runtime_error_naive or result.compile_error_naive
        if error_msg and is_allowed_err(error_msg):
            allowed[0] = 1
    else:
        outputs.append(result.output_naive)
    
    if not result.runtime_success_xla:
        fail[1] = 1
        error_msg = result.runtime_error_xla or result.compile_error_xla
        if error_msg and is_allowed_err(error_msg):
            allowed[1] = 1
    else:
        outputs.append(result.output_xla)
    
    if not result.runtime_success_autocluster:
        fail[2] = 1
        error_msg = result.runtime_error_autocluster or result.compile_error_autocluster
        if error_msg and is_allowed_err(error_msg):
            allowed[2] = 1
    else:
        outputs.append(result.output_autocluster)
    
    if fail != [0, 0, 0]:
        temp = []
        for i in range(3):
            temp.append(fail[i] - allowed[i])
        
        if temp == [0, 0, 0]:
            return bug_reports
        
        fail_str = str(fail)
        if fail_str in FAIL_MAPPING:
            res_type = FAIL_MAPPING[fail_str]
            
            error_details = {
                "Naive Fail": result.runtime_error_naive or result.compile_error_naive if fail[0] else None,
                "XLA Fail": result.runtime_error_xla or result.compile_error_xla if fail[1] else None,
                "AC Fail": result.runtime_error_autocluster or result.compile_error_autocluster if fail[2] else None,
                "Num Diff": None,
            }
            
            bug_reports.append(BugReport(
                test_id=test_id,
                optimizations_triggered=optimizations_triggered,
                oracle_type=res_type.name,
                details=error_details,
                test_file=result.test_file,
                logs_file=result.test_file.with_suffix(".log"),
            ))
        
        return bug_reports
    
    if len(outputs) == 3:
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error_details = {
                "Naive Fail": None,
                "XLA Fail": None,
                "AC Fail": None,
                "Num Diff": msg,
            }
            
            if test_code:
                res_type = value_diff_type(test_code, msg)
            else:
                res_type = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            
            bug_reports.append(BugReport(
                test_id=test_id,
                optimizations_triggered=optimizations_triggered,
                oracle_type=res_type.name,
                details=error_details,
                test_file=result.test_file,
                logs_file=result.test_file.with_suffix(".log"),
            ))
            return bug_reports
        
        equal, msg = is_equal(outputs[0], outputs[2])
        if not equal:
            error_details = {
                "Naive Fail": None,
                "XLA Fail": None,
                "AC Fail": None,
                "Num Diff": msg,
            }
            
            if test_code:
                res_type = value_diff_type(test_code, msg)
            else:
                res_type = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            
            bug_reports.append(BugReport(
                test_id=test_id,
                optimizations_triggered=optimizations_triggered,
                oracle_type=res_type.name,
                details=error_details,
                test_file=result.test_file,
                logs_file=result.test_file.with_suffix(".log"),
            ))
            return bug_reports
    
    elif len(outputs) == 2:
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error_details = {
                "Naive Fail": None,
                "XLA Fail": None,
                "AC Fail": None,
                "Num Diff": msg,
            }
            
            if test_code:
                res_type = value_diff_type(test_code, msg)
            else:
                res_type = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            
            bug_reports.append(BugReport(
                test_id=test_id,
                optimizations_triggered=optimizations_triggered,
                oracle_type=res_type.name,
                details=error_details,
                test_file=result.test_file,
                logs_file=result.test_file.with_suffix(".log"),
            ))
            return bug_reports
    
    return bug_reports

