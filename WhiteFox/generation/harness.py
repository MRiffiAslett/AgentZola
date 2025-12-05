"""
Execution harness and oracles for WhiteFox.

Executes generated test programs with and without XLA optimizations,
captures logs, and applies bug detection oracles.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from .log_parser import extract_triggered_passes


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a test program."""
    test_file: Path
    compile_success_baseline: bool
    compile_success_optimized: bool
    runtime_success_baseline: bool
    runtime_success_optimized: bool
    compile_error_baseline: Optional[str] = None
    compile_error_optimized: Optional[str] = None
    runtime_error_baseline: Optional[str] = None
    runtime_error_optimized: Optional[str] = None
    output_baseline: Optional[Any] = None
    output_optimized: Optional[Any] = None
    output_shape_baseline: Optional[Tuple] = None
    output_shape_optimized: Optional[Tuple] = None
    output_dtype_baseline: Optional[str] = None
    output_dtype_optimized: Optional[str] = None
    log_text: str = ""
    triggered_passes: set = None
    
    def __post_init__(self):
        if self.triggered_passes is None:
            self.triggered_passes = set()


@dataclass
class BugReport:
    """Bug report for a test that triggered an oracle condition."""
    test_id: str
    optimizations_triggered: List[str]
    oracle_type: str  # e.g. "miscompilation", "compile_crash_optimized", ...
    details: Dict[str, Any]  # serialized info: exception messages, shapes, etc.
    test_file: Path
    logs_file: Path
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "optimizations_triggered": self.optimizations_triggered,
            "oracle_type": self.oracle_type,
            "details": self.details,
            "test_file": str(self.test_file),
            "logs_file": str(self.logs_file),
        }
    
    def save(self, file_path: Path) -> None:
        """Save bug report to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def execute_test_in_subprocess(test_file: Path, timeout: int = 60) -> ExecutionResult:
    """
    Execute a test program in a subprocess and capture results.
    
    The test program should be a standard TensorFlow model/program (like in WhiteFox paper).
    This harness executes it twice: with and without XLA optimization.
    
    Args:
        test_file: Path to the test Python file.
        timeout: Maximum execution time in seconds.
        
    Returns:
        ExecutionResult with all captured information.
    """
    result = ExecutionResult(
        test_file=test_file,
        compile_success_baseline=False,
        compile_success_optimized=False,
        runtime_success_baseline=False,
        runtime_success_optimized=False,
    )
    
    # Create a wrapper script that executes the test and captures results
    # Use repr to safely escape the file path
    test_file_repr = repr(str(test_file))
    
    wrapper_script = f"""
import sys
import json
import traceback
import io
from pathlib import Path

# Capture stdout/stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

# Redirect stdout/stderr to capture logs
import sys
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeOutput(original_stdout, stdout_capture)
sys.stderr = TeeOutput(original_stderr, stderr_capture)

result = {{
    "compile_success_baseline": False,
    "compile_success_optimized": False,
    "runtime_success_baseline": False,
    "runtime_success_optimized": False,
    "compile_error_baseline": None,
    "compile_error_optimized": None,
    "runtime_error_baseline": None,
    "runtime_error_optimized": None,
    "output_baseline": None,
    "output_optimized": None,
    "output_shape_baseline": None,
    "output_shape_optimized": None,
    "output_dtype_baseline": None,
    "output_dtype_optimized": None,
    "log_text": "",
}}

try:
    import tensorflow as tf
    import numpy as np
    
    # Execute the test file to define model, inputs, etc.
    # This follows the WhiteFox paper approach: standard TF models
    test_path = Path({test_file_repr})
    test_globals = {{'__name__': '__main__'}}.copy()
    
    try:
        with open(test_path, 'r') as f:
            test_code = f.read()
        exec(test_code, test_globals)
        result["compile_success_baseline"] = True
        result["compile_success_optimized"] = True
    except Exception as e:
        result["compile_error_baseline"] = str(e) + "\\n" + traceback.format_exc()
        result["compile_error_optimized"] = str(e) + "\\n" + traceback.format_exc()
        raise
    
    # Extract model and inputs from executed test
    # Standard WhiteFox tests define: Model class, instantiate it as 'm' or 'model', and create inputs
    model = None
    inputs = None
    
    # Find model instance (typically named 'm', 'model', 'net', etc.)
    for name in ['m', 'model', 'M', 'net', 'Model']:
        if name in test_globals:
            obj = test_globals[name]
            # Check if it's a model instance or class
            if isinstance(obj, type) and issubclass(obj, tf.keras.Model):
                # It's a class, instantiate it
                model = obj()
            elif isinstance(obj, tf.keras.Model):
                # It's an instance
                model = obj
                break
    
    # Find inputs (typically named with 'x', 'input', etc.)
    for name in ['x1', 'x', 'input', 'inputs', 'input1', 'X', 'inp']:
        if name in test_globals:
            obj = test_globals[name]
            if isinstance(obj, (tf.Tensor, tf.Variable, np.ndarray)):
                if isinstance(obj, np.ndarray):
                    inputs = tf.constant(obj)
                else:
                    inputs = obj
                break
            elif isinstance(obj, (list, tuple)):
                # Convert list/tuple to tensor
                try:
                    inputs = tf.constant(obj)
                    break
                except Exception:
                    pass
    
    # If no explicit inputs found, check for any variable ending with 'input' or starting with 'x'
    if inputs is None:
        for name, obj in test_globals.items():
            if (('input' in name.lower() or name.startswith('x')) and 
                isinstance(obj, (tf.Tensor, tf.Variable, np.ndarray))):
                if isinstance(obj, np.ndarray):
                    inputs = tf.constant(obj)
                else:
                    inputs = obj
                break
    
    # If still no inputs found, create default ones
    if inputs is None and model is not None:
        try:
            # Try to infer input shape from model
            inputs = tf.ones([1, 10])  # Default small input
        except Exception:
            inputs = tf.ones([1])
    
    # Run baseline (without XLA optimization)
    if model is not None and inputs is not None:
        try:
            output_baseline = model(inputs)
            result["runtime_success_baseline"] = True
            
            # Convert output to serializable format
            if isinstance(output_baseline, tf.Tensor):
                result["output_baseline"] = output_baseline.numpy().tolist()
                result["output_shape_baseline"] = list(output_baseline.shape)
                result["output_dtype_baseline"] = str(output_baseline.dtype)
            elif isinstance(output_baseline, (tuple, list)):
                result["output_baseline"] = [t.numpy().tolist() if isinstance(t, tf.Tensor) else t for t in output_baseline]
                if output_baseline and isinstance(output_baseline[0], tf.Tensor):
                    result["output_shape_baseline"] = list(output_baseline[0].shape)
                    result["output_dtype_baseline"] = str(output_baseline[0].dtype)
            else:
                result["output_baseline"] = str(output_baseline)
        except Exception as e:
            result["runtime_error_baseline"] = str(e) + "\\n" + traceback.format_exc()
    
    # Run optimized (with XLA optimization via jit_compile=True)
    # This follows the WhiteFox paper approach: compile each test twice
    if model is not None and inputs is not None:
        try:
            # Wrap model call in tf.function with jit_compile=True for XLA optimization
            @tf.function(jit_compile=True)
            def optimized_call(x):
                return model(x, training=False)
            
            # Force compilation by calling once
            output_optimized = optimized_call(inputs)
            result["runtime_success_optimized"] = True
            
            # Convert output to serializable format
            if isinstance(output_optimized, tf.Tensor):
                result["output_optimized"] = output_optimized.numpy().tolist()
                result["output_shape_optimized"] = list(output_optimized.shape)
                result["output_dtype_optimized"] = str(output_optimized.dtype)
            elif isinstance(output_optimized, (tuple, list)):
                result["output_optimized"] = [t.numpy().tolist() if isinstance(t, tf.Tensor) else t for t in output_optimized]
                if output_optimized and isinstance(output_optimized[0], tf.Tensor):
                    result["output_shape_optimized"] = list(output_optimized[0].shape)
                    result["output_dtype_optimized"] = str(output_optimized[0].dtype)
            else:
                result["output_optimized"] = str(output_optimized)
        except Exception as e:
            result["runtime_error_optimized"] = str(e) + "\\n" + traceback.format_exc()
    
    # If no model/inputs found, mark as compile error
    if model is None or inputs is None:
        error_msg = f"Could not extract model or inputs from test. Model: {{model is not None}}, Inputs: {{inputs is not None}}"
        if not result["compile_error_baseline"]:
            result["compile_error_baseline"] = error_msg
        if not result["compile_error_optimized"]:
            result["compile_error_optimized"] = error_msg

except Exception as e:
    # General import/execution error
    if not result["compile_error_baseline"]:
        result["compile_error_baseline"] = str(e) + "\\n" + traceback.format_exc()
    if not result["compile_error_optimized"]:
        result["compile_error_optimized"] = str(e) + "\\n" + traceback.format_exc()

finally:
    # Capture all logs
    result["log_text"] = stdout_capture.getvalue() + stderr_capture.getvalue()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

# Output result as JSON
print("WHITEFOX_RESULT_START")
print(json.dumps(result))
print("WHITEFOX_RESULT_END")
"""
    
    try:
        # Run the wrapper script
        process = subprocess.run(
            [sys.executable, "-c", wrapper_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        # Parse output
        output = process.stdout + process.stderr
        result.log_text = output
        
        # Extract JSON result if present
        if "WHITEFOX_RESULT_START" in output and "WHITEFOX_RESULT_END" in output:
            start_idx = output.find("WHITEFOX_RESULT_START") + len("WHITEFOX_RESULT_START")
            end_idx = output.find("WHITEFOX_RESULT_END")
            json_str = output[start_idx:end_idx].strip()
            try:
                result_data = json.loads(json_str)
                result.compile_success_baseline = result_data.get("compile_success_baseline", False)
                result.compile_success_optimized = result_data.get("compile_success_optimized", False)
                result.runtime_success_baseline = result_data.get("runtime_success_baseline", False)
                result.runtime_success_optimized = result_data.get("runtime_success_optimized", False)
                result.compile_error_baseline = result_data.get("compile_error_baseline")
                result.compile_error_optimized = result_data.get("compile_error_optimized")
                result.runtime_error_baseline = result_data.get("runtime_error_baseline")
                result.runtime_error_optimized = result_data.get("runtime_error_optimized")
                result.output_baseline = result_data.get("output_baseline")
                result.output_optimized = result_data.get("output_optimized")
                result.output_shape_baseline = result_data.get("output_shape_baseline")
                result.output_shape_optimized = result_data.get("output_shape_optimized")
                result.output_dtype_baseline = result_data.get("output_dtype_baseline")
                result.output_dtype_optimized = result_data.get("output_dtype_optimized")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON result from {test_file}")
        
        # Extract triggered passes from logs
        result.triggered_passes = extract_triggered_passes(result.log_text)
        
    except subprocess.TimeoutExpired:
        result.runtime_error_baseline = "Execution timeout"
        result.runtime_error_optimized = "Execution timeout"
        logger.warning(f"Test {test_file} timed out after {timeout} seconds")
    except Exception as e:
        result.compile_error_baseline = str(e)
        result.compile_error_optimized = str(e)
        logger.error(f"Error executing {test_file}: {e}")
    
    return result


def check_oracles(
    result: ExecutionResult,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> List[BugReport]:
    """
    Apply oracles to detect bugs.
    
    Checks for:
    - Crashes (compile-time or runtime)
    - Compile/execution status mismatches
    - Result inconsistencies (miscompilation)
    
    Args:
        result: ExecutionResult to check.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.
        
    Returns:
        List of BugReport instances (empty if no bugs detected).
    """
    bug_reports = []
    test_id = result.test_file.stem
    
    # Determine which optimizations were triggered
    optimizations_triggered = list(result.triggered_passes)
    
    # Oracle 1: Compile crashes
    if not result.compile_success_baseline and result.compile_error_baseline:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="compile_crash_baseline",
            details={
                "error": result.compile_error_baseline,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    if not result.compile_success_optimized and result.compile_error_optimized:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="compile_crash_optimized",
            details={
                "error": result.compile_error_optimized,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    # Oracle 2: Runtime crashes
    if result.compile_success_baseline and not result.runtime_success_baseline:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="runtime_crash_baseline",
            details={
                "error": result.runtime_error_baseline,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    if result.compile_success_optimized and not result.runtime_success_optimized:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="runtime_crash_optimized",
            details={
                "error": result.runtime_error_optimized,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    # Oracle 3: Compile/execution status mismatches
    if result.compile_success_baseline != result.compile_success_optimized:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="compile_status_mismatch",
            details={
                "baseline_compile": result.compile_success_baseline,
                "optimized_compile": result.compile_success_optimized,
                "baseline_error": result.compile_error_baseline,
                "optimized_error": result.compile_error_optimized,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    if result.runtime_success_baseline != result.runtime_success_optimized:
        bug_reports.append(BugReport(
            test_id=test_id,
            optimizations_triggered=optimizations_triggered,
            oracle_type="runtime_status_mismatch",
            details={
                "baseline_runtime": result.runtime_success_baseline,
                "optimized_runtime": result.runtime_success_optimized,
                "baseline_error": result.runtime_error_baseline,
                "optimized_error": result.runtime_error_optimized,
            },
            test_file=result.test_file,
            logs_file=result.test_file.with_suffix(".log"),
        ))
    
    # Oracle 4: Result inconsistency (miscompilation)
    if (result.runtime_success_baseline and result.runtime_success_optimized and
        result.output_baseline is not None and result.output_optimized is not None):
        
        try:
            # Convert to numpy arrays for comparison
            baseline_arr = np.array(result.output_baseline)
            optimized_arr = np.array(result.output_optimized)
            
            if baseline_arr.shape != optimized_arr.shape:
                bug_reports.append(BugReport(
                    test_id=test_id,
                    optimizations_triggered=optimizations_triggered,
                    oracle_type="miscompilation_shape_mismatch",
                    details={
                        "baseline_shape": list(baseline_arr.shape),
                        "optimized_shape": list(optimized_arr.shape),
                    },
                    test_file=result.test_file,
                    logs_file=result.test_file.with_suffix(".log"),
                ))
            elif not np.allclose(baseline_arr, optimized_arr, rtol=rtol, atol=atol):
                bug_reports.append(BugReport(
                    test_id=test_id,
                    optimizations_triggered=optimizations_triggered,
                    oracle_type="miscompilation",
                    details={
                        "baseline_shape": list(baseline_arr.shape),
                        "optimized_shape": list(optimized_arr.shape),
                        "baseline_dtype": result.output_dtype_baseline,
                        "optimized_dtype": result.output_dtype_optimized,
                        "max_diff": float(np.max(np.abs(baseline_arr - optimized_arr))),
                        "rtol": rtol,
                        "atol": atol,
                    },
                    test_file=result.test_file,
                    logs_file=result.test_file.with_suffix(".log"),
                ))
        except Exception as e:
            # If comparison fails, report as potential bug
            bug_reports.append(BugReport(
                test_id=test_id,
                optimizations_triggered=optimizations_triggered,
                oracle_type="miscompilation_comparison_error",
                details={
                    "error": str(e),
                    "baseline_type": type(result.output_baseline).__name__,
                    "optimized_type": type(result.output_optimized).__name__,
                },
                test_file=result.test_file,
                logs_file=result.test_file.with_suffix(".log"),
            ))
    
    return bug_reports
