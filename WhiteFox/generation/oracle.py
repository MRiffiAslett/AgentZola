from enum import IntEnum, auto
from typing import List, Optional, Tuple

import numpy as np
from domain.harness import BugReport, ExecutionResult


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
    TorchTensor = auto()
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
    else:
        try:
            import tensorflow as tf

            if isinstance(output_data, tf.Tensor):
                return DataType.TFTensor
            if hasattr(
                tf.keras.backend, "is_keras_tensor"
            ) and tf.keras.backend.is_keras_tensor(output_data):
                return DataType.KerasTensor
        except (ImportError, Exception):
            pass

        try:
            import torch

            if isinstance(output_data, torch.Tensor):
                return DataType.TorchTensor
        except (ImportError, Exception):
            pass

        return DataType.Unknown


def is_equal(x, y) -> Tuple[bool, Optional[str]]:
    x_type, y_type = get_type(x), get_type(y)

    if x_type != y_type and not (
        x_type in [DataType.List, DataType.Tuple]
        and y_type in [DataType.List, DataType.Tuple]
    ):
        try:
            equal = np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True)
            return equal, "Value mismatch: {} vs {}".format(x, y)
        except (ValueError, TypeError):
            return False, "Type mismatch: {} vs {}".format(str(x_type), str(y_type))

    if x_type in [DataType.Int, DataType.Bool, DataType.Null, DataType.Str]:
        return x == y, "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.Float:
        return abs(x - y) < 1e-2, "Value mismatch: {} vs {}".format(x, y)
    elif x_type in [DataType.TFTensor, DataType.KerasTensor, DataType.TorchTensor]:
        return np.allclose(
            np.array(x), np.array(y), atol=1e-02, equal_nan=True
        ), "Value mismatch: {} vs {}".format(x, y)
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
    if "tf.random" in code or "torch.rand" in code:
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


def is_allowed_err(error, allowed_errors: Optional[List[str]] = None) -> bool:
    error = str(error)
    if allowed_errors is None:
        allowed_errors = []
    for err in allowed_errors:
        if err in error:
            return True
    return False


def check_oracles(
    result: ExecutionResult,
    test_code: Optional[str] = None,
    allowed_errors: Optional[List[str]] = None,
) -> List[BugReport]:
    bug_reports = []
    test_id = result.test_file.stem
    optimizations_triggered = list(result.triggered_passes)

    modes = result.modes
    if not modes:
        return bug_reports

    fail = []
    allowed = []
    outputs = []
    mode_names = []

    for mode in modes:
        mr = result.get_mode(mode)
        if not mr.runtime_success:
            fail.append(1)
            error_msg = mr.runtime_error or mr.compile_error
            if error_msg and is_allowed_err(error_msg, allowed_errors):
                allowed.append(1)
            else:
                allowed.append(0)
        else:
            fail.append(0)
            allowed.append(0)
            outputs.append(mr.output)
        mode_names.append(mode)

    if any(f == 1 for f in fail):
        temp = [fail[i] - allowed[i] for i in range(len(fail))]

        if all(t == 0 for t in temp):
            return bug_reports

        fail_str = str(fail)
        if fail_str in FAIL_MAPPING:
            res_type = FAIL_MAPPING[fail_str]

            if res_type == ResType.AllFail:
                return bug_reports

            error_details = {}
            for i, mode in enumerate(mode_names):
                mr = result.get_mode(mode)
                if fail[i]:
                    error_details[f"{mode} Fail"] = mr.runtime_error or mr.compile_error
                else:
                    error_details[f"{mode} Fail"] = None
            error_details["Num Diff"] = None

            bug_reports.append(
                BugReport(
                    test_id=test_id,
                    optimizations_triggered=optimizations_triggered,
                    oracle_type=res_type.name,
                    details=error_details,
                    test_file=result.test_file,
                    logs_file=result.test_file.with_suffix(".log"),
                )
            )
        else:
            if not all(f == 1 for f in fail):
                error_details = {}
                for i, mode in enumerate(mode_names):
                    mr = result.get_mode(mode)
                    if fail[i]:
                        error_details[f"{mode} Fail"] = (
                            mr.runtime_error or mr.compile_error
                        )
                    else:
                        error_details[f"{mode} Fail"] = None
                error_details["Num Diff"] = None

                failed_modes = [mode_names[i] for i in range(len(fail)) if fail[i]]
                oracle_type = "_".join(m.capitalize() for m in failed_modes) + "Fail"

                bug_reports.append(
                    BugReport(
                        test_id=test_id,
                        optimizations_triggered=optimizations_triggered,
                        oracle_type=oracle_type,
                        details=error_details,
                        test_file=result.test_file,
                        logs_file=result.test_file.with_suffix(".log"),
                    )
                )

        return bug_reports

    def create_diff_bug_report(msg: str) -> BugReport:
        error_details = {f"{mode} Fail": None for mode in mode_names}
        error_details["Num Diff"] = msg

        if test_code:
            res_type = value_diff_type(test_code, msg)
        else:
            res_type = (
                ResType.AllDiff_TypeMismatch
                if "Type mismatch" in msg
                else ResType.AllDiff
            )

        return BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type=res_type.name,
            details=error_details,
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        )

    if len(outputs) >= 2:
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            bug_reports.append(create_diff_bug_report(msg))
            return bug_reports

        for i in range(2, len(outputs)):
            equal, msg = is_equal(outputs[0], outputs[i])
            if not equal:
                bug_reports.append(create_diff_bug_report(msg))
                return bug_reports

    return bug_reports
