from pathlib import Path
from typing import Optional

from domain.harness import ExecutionResult
from generation.harness.base import TestHarness
from generation.harness.inductor import PyTorchInductorHarness
from generation.harness.tflite import TensorFlowLiteHarness
from generation.harness.xla import TensorFlowXLAHarness

_default_harness = TensorFlowXLAHarness()


def execute_test_in_subprocess(
    test_file: Path,
    whitefox_logger=None,
    optimization_name: Optional[str] = None,
    iteration: Optional[int] = None,
    sample_idx: Optional[int] = None,
    timeout: int = 60,
    harness=None,
) -> ExecutionResult:
    h = harness or _default_harness
    return h.execute_test_in_subprocess(
        test_file=test_file,
        whitefox_logger=whitefox_logger,
        optimization_name=optimization_name,
        iteration=iteration,
        sample_idx=sample_idx,
        timeout=timeout,
    )


__all__ = [
    "TestHarness",
    "TensorFlowXLAHarness",
    "PyTorchInductorHarness",
    "TensorFlowLiteHarness",
    "execute_test_in_subprocess",
]
