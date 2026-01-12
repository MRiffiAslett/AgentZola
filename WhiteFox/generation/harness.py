"""
Execution harness for WhiteFox.

Executes generated test programs with three execution modes matching original WhiteFox:
- naive: Standard TensorFlow execution
- xla: XLA compilation with jit_compile=True
- autocluster: XLA autoclustering

This implementation matches the original WhiteFox exec_wrapper and xla_run logic.
"""

import json
import subprocess
import sys
import os
import random
from pathlib import Path

_temp = sys.modules.pop('generation.logging', None)
import logging
if _temp:
    sys.modules['generation.logging'] = _temp

import numpy as np

from domain.harness import ExecutionResult
from generation.logging import extract_triggered_passes
from typing import Optional


logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def add_decorator(code: str, decorator: str) -> str:
    """Add a decorator to the 'call' method of a TensorFlow model.
    
    Args:
        code: Source code string
        decorator: Decorator to add (e.g., "@tf.function(jit_compile=True)")
        
    Returns:
        Modified code with decorator added
    """
    if "    def call" in code:
        # The indentation is 4 spaces
        code = code.replace("    def call", f"    {decorator}\n    def call")
    else:
        # The indentation is 2 spaces
        code = code.replace("  def call", f"  {decorator}\n  def call")
    return code



def execute_test_in_subprocess(
    test_file: Path, 
    whitefox_logger=None,
    optimization_name: Optional[str] = None,
    iteration: Optional[int] = None,
    sample_idx: Optional[int] = None,
    timeout: int = 60
) -> ExecutionResult:
    if whitefox_logger:
        whitefox_logger.trace(f"          [HARNESS] Starting execution", {
            "test_file": test_file.name,
            "timeout": timeout,
        })
    
    result = ExecutionResult(
        test_file=test_file,
        compile_success_naive=False,
        compile_success_xla=False,
        compile_success_autocluster=False,
        runtime_success_naive=False,
        runtime_success_xla=False,
        runtime_success_autocluster=False,
    )
    
    try:
        with open(test_file, 'r') as f:
            test_code = f.read()
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] Code read successfully", {
                "code_length": len(test_code),
            })
    except Exception as e:
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] Failed to read code", {
                "error": str(e),
            })
        result.compile_error_naive = str(e)
        result.compile_error_xla = str(e)
        result.compile_error_autocluster = str(e)
        return result
    
    test_file_repr = repr(str(test_file))
    test_code_repr = repr(test_code)
    
    wrapper_script = f"""
import sys
import json
import traceback
import io
from pathlib import Path

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

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
    def close(self):
        for f in self.files:
            if hasattr(f, 'close'):
                try:
                    f.close()
                except Exception:
                    pass  # Ignore errors when closing

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeOutput(original_stdout, stdout_capture)
sys.stderr = TeeOutput(original_stderr, stderr_capture)

result = {{
    "compile_success_naive": False,
    "compile_success_xla": False,
    "compile_success_autocluster": False,
    "runtime_success_naive": False,
    "runtime_success_xla": False,
    "runtime_success_autocluster": False,
    "compile_error_naive": None,
    "compile_error_xla": None,
    "compile_error_autocluster": None,
    "runtime_error_naive": None,
    "runtime_error_xla": None,
    "runtime_error_autocluster": None,
    "output_naive": None,
    "output_xla": None,
    "output_autocluster": None,
    "log_text": "",
}}

def _serialize_output(output):
    import tensorflow as tf
    if isinstance(output, tf.Tensor):
        return output.numpy().tolist()
    elif isinstance(output, (tuple, list)):
        return [_serialize_output(t) if isinstance(t, tf.Tensor) else t for t in output]
    else:
        return str(output)


def add_decorator_inline(code: str, decorator: str) -> str:
    \"\"\"Add a decorator to the call method - inline version for subprocess.\"\"\"
    if "    def call" in code:
        code = code.replace("    def call", f"    {decorator}\\n    def call")
    else:
        code = code.replace("  def call", f"  {decorator}\\n  def call")
    return code


# Set environment variables BEFORE importing TensorFlow
# TensorFlow reads XLA_FLAGS and TF_XLA_FLAGS at import time
import os
old_xla_flags_env = os.environ.get('XLA_FLAGS', '')
old_tf_xla_flags_env = os.environ.get('TF_XLA_FLAGS', '')

# Preserve existing flags and add our instrumentation flags
xla_flags_parts = []
if old_xla_flags_env:
    xla_flags_parts.append(old_xla_flags_env)
# Ensure XLA dumps are enabled to capture pass information
if '--xla_dump_to=' not in old_xla_flags_env:
    xla_flags_parts.append('--xla_dump_to=/tmp/xla_dump')
if '--xla_dump_hlo_pass_re=' not in old_xla_flags_env:
    xla_flags_parts.append('--xla_dump_hlo_pass_re=.*')
os.environ['XLA_FLAGS'] = ' '.join(xla_flags_parts) if xla_flags_parts else ''

# Note: TF_XLA_FLAGS will be set per-mode during execution
# Do NOT set globally here as it affects all modes

try:
    import tensorflow as tf
    import numpy as np
    import random
    
    random.seed({RANDOM_SEED})
    np.random.seed({RANDOM_SEED})
    tf.random.set_seed({RANDOM_SEED})
    
    test_code = {test_code_repr}
    model_key = 'm'
    input_data_key = 'input_data'
    
    # NAIVE EXECUTION (no decorators)
    try:
        test_globals_naive = {{'__name__': '__main__'}}.copy()
        exec(test_code, test_globals_naive)
        result["compile_success_naive"] = True
        
        if model_key not in test_globals_naive:
            raise Exception("Model 'm' not found in test code")
        if input_data_key not in test_globals_naive:
            raise Exception("input_data not found in test code - code processing may have failed")
        
        m = test_globals_naive[model_key]
        input_data = test_globals_naive[input_data_key]
        
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]
        
        output_naive = m(*input_data)
        result["runtime_success_naive"] = True
        result["output_naive"] = _serialize_output(output_naive)
    except Exception as e:
        error_msg = str(e) + "\\n" + traceback.format_exc()
        if not result["compile_success_naive"]:
            result["compile_error_naive"] = error_msg
        else:
            result["runtime_error_naive"] = error_msg
    
    # XLA EXECUTION (add @tf.function(jit_compile=True) decorator to source)
    try:
        test_code_xla = add_decorator_inline(test_code, "@tf.function(jit_compile=True)")
        test_globals_xla = {{'__name__': '__main__'}}.copy()
        exec(test_code_xla, test_globals_xla)
        result["compile_success_xla"] = True
        
        if model_key not in test_globals_xla or input_data_key not in test_globals_xla:
            raise Exception("Model or input_data not found in XLA execution")
        
        m_xla = test_globals_xla[model_key]
        input_data_xla = test_globals_xla[input_data_key]
        
        if not isinstance(input_data_xla, (list, tuple)):
            input_data_xla = [input_data_xla]
        
        output_xla = m_xla(*input_data_xla)
        result["runtime_success_xla"] = True
        result["output_xla"] = _serialize_output(output_xla)
    except Exception as e:
        error_msg = str(e) + "\\n" + traceback.format_exc()
        if not result["compile_success_xla"]:
            result["compile_error_xla"] = error_msg
        else:
            result["runtime_error_xla"] = error_msg
    
    # AUTOCLUSTER EXECUTION (set env vars, add @tf.function decorator to source)
    try:
        # Set autocluster environment variables
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        
        test_code_ac = add_decorator_inline(test_code, "@tf.function")
        test_globals_ac = {{'__name__': '__main__'}}.copy()
        exec(test_code_ac, test_globals_ac)
        result["compile_success_autocluster"] = True
        
        if model_key not in test_globals_ac or input_data_key not in test_globals_ac:
            raise Exception("Model or input_data not found in autocluster execution")
        
        m_ac = test_globals_ac[model_key]
        input_data_ac = test_globals_ac[input_data_key]
        
        if not isinstance(input_data_ac, (list, tuple)):
            input_data_ac = [input_data_ac]
        
        output_ac = m_ac(*input_data_ac)
        result["runtime_success_autocluster"] = True
        result["output_autocluster"] = _serialize_output(output_ac)
        
        # Restore environment variable
        if old_tf_xla_flags_env:
            os.environ['TF_XLA_FLAGS'] = old_tf_xla_flags_env
        else:
            os.environ.pop('TF_XLA_FLAGS', None)
    except Exception as e:
        error_msg = str(e) + "\\n" + traceback.format_exc()
        if not result["compile_success_autocluster"]:
            result["compile_error_autocluster"] = error_msg
        else:
            result["runtime_error_autocluster"] = error_msg
        # Restore environment variable even on error
        if old_tf_xla_flags_env:
            os.environ['TF_XLA_FLAGS'] = old_tf_xla_flags_env
        else:
            os.environ.pop('TF_XLA_FLAGS', None)

except Exception as e:
    error_msg = str(e) + "\\n" + traceback.format_exc()
    if not result["compile_error_naive"]:
        result["compile_error_naive"] = error_msg
    if not result["compile_error_xla"]:
        result["compile_error_xla"] = error_msg
    if not result["compile_error_autocluster"]:
        result["compile_error_autocluster"] = error_msg

finally:
    sys.stdout.flush()
    sys.stderr.flush()
    result["log_text"] = stdout_capture.getvalue() + stderr_capture.getvalue()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    # Restore original environment variables
    if old_xla_flags_env:
        os.environ['XLA_FLAGS'] = old_xla_flags_env
    else:
        os.environ.pop('XLA_FLAGS', None)
    if old_tf_xla_flags_env:
        os.environ['TF_XLA_FLAGS'] = old_tf_xla_flags_env
    else:
        os.environ.pop('TF_XLA_FLAGS', None)

print("WHITEFOX_RESULT_START")
print(json.dumps(result))
print("WHITEFOX_RESULT_END")
"""
    
    if whitefox_logger:
        whitefox_logger.trace(f"          [HARNESS] Launching subprocess", {
            "timeout": timeout,
        })
    
    try:
        # Ensure subprocess inherits environment variables (especially XLA_FLAGS)
        # This is important for the custom TensorFlow build to output WHITEFOX_PASS_START markers
        process = subprocess.run(
            [sys.executable, "-c", wrapper_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),  # Explicitly pass environment
        )
        
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] Subprocess completed", {
                "returncode": process.returncode,
            })
        
        output = process.stdout + process.stderr
        
        log_text_from_json = None
        if "WHITEFOX_RESULT_START" in output and "WHITEFOX_RESULT_END" in output:
            start_idx = output.find("WHITEFOX_RESULT_START") + len("WHITEFOX_RESULT_START")
            end_idx = output.find("WHITEFOX_RESULT_END")
            json_str = output[start_idx:end_idx].strip()
            try:
                result_data = json.loads(json_str)
                result.compile_success_naive = result_data.get("compile_success_naive", False)
                result.compile_success_xla = result_data.get("compile_success_xla", False)
                result.compile_success_autocluster = result_data.get("compile_success_autocluster", False)
                result.runtime_success_naive = result_data.get("runtime_success_naive", False)
                result.runtime_success_xla = result_data.get("runtime_success_xla", False)
                result.runtime_success_autocluster = result_data.get("runtime_success_autocluster", False)
                result.compile_error_naive = result_data.get("compile_error_naive")
                result.compile_error_xla = result_data.get("compile_error_xla")
                result.compile_error_autocluster = result_data.get("compile_error_autocluster")
                result.runtime_error_naive = result_data.get("runtime_error_naive")
                result.runtime_error_xla = result_data.get("runtime_error_xla")
                result.runtime_error_autocluster = result_data.get("runtime_error_autocluster")
                result.output_naive = result_data.get("output_naive")
                result.output_xla = result_data.get("output_xla")
                result.output_autocluster = result_data.get("output_autocluster")
                log_text_from_json = result_data.get("log_text", "")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON result from {test_file}")
        
        if log_text_from_json is not None and log_text_from_json:
            result.log_text = log_text_from_json
        else:
            result.log_text = output
        
        result.triggered_passes = extract_triggered_passes(result.log_text)
        
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] Parsed results", {
                "triggered_passes": list(result.triggered_passes),
                "naive_success": result.runtime_success_naive,
                "xla_success": result.runtime_success_xla,
                "ac_success": result.runtime_success_autocluster,
            })
        
        # Log diagnostic information after execution (failures only + pass detection)
        if whitefox_logger:
            # Log initial execution failures only
            if result.compile_error_naive:
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "exec_initial",
                    "failure",
                    {
                        "error": result.compile_error_naive[:500],
                        "error_type": "compile_error"
                    }
                )
            elif result.runtime_error_naive:
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "exec_initial",
                    "failure",
                    {
                        "error": result.runtime_error_naive[:500],
                        "error_type": "runtime_error"
                    }
                )
            elif result.compile_success_naive and result.runtime_success_naive:
                # Log success
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "exec_initial",
                    "success",
                    {}
                )
            
            # Log XLA execution failures only
            if result.compile_error_xla:
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "xla_exec",
                    "failure",
                    {
                        "error": result.compile_error_xla[:500],
                        "error_type": "xla_compile_error"
                    }
                )
            elif result.runtime_error_xla:
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "xla_exec",
                    "failure",
                    {
                        "error": result.runtime_error_xla[:500],
                        "error_type": "xla_runtime_error"
                    }
                )
            elif result.compile_success_xla and result.runtime_success_xla:
                # Log success
                whitefox_logger.log_diagnostic(
                    optimization_name or "unknown",
                    iteration or 0,
                    sample_idx or 0,
                    "xla_exec",
                    "success",
                    {"triggered_passes": list(result.triggered_passes)}
                )
        
        # Debug logging: check if WHITEFOX_PASS_START markers are present
        if "WHITEFOX_PASS_START" in result.log_text:
            logger.debug(f"Found WHITEFOX_PASS_START markers in log for {test_file}, "
                        f"detected passes: {result.triggered_passes}")
        elif result.runtime_success_xla:
            logger.warning(f"No WHITEFOX_PASS_START markers found in log for {test_file} "
                         f"despite successful XLA execution. Log length: {len(result.log_text)}, "
                         f"log_text_from_json was {'set' if log_text_from_json else 'None'}")
        elif result.compile_success_xla and not result.runtime_success_xla:
            logger.debug(f"XLA compiled but runtime failed for {test_file}, log length: {len(result.log_text)}")
        elif len(result.log_text) == 0:
            logger.debug(f"Empty log_text for {test_file}, XLA never ran or logs not captured")
        
    except subprocess.TimeoutExpired:
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] TIMEOUT after {timeout}s")
        result.runtime_error_naive = "Execution timeout"
        result.runtime_error_xla = "Execution timeout"
        result.runtime_error_autocluster = "Execution timeout"
        logger.warning(f"Test {test_file} timed out after {timeout} seconds")
    except Exception as e:
        if whitefox_logger:
            whitefox_logger.trace(f"          [HARNESS] Exception in subprocess", {
                "error": str(e)[:200],
            })
        result.compile_error_naive = str(e)
        result.compile_error_xla = str(e)
        result.compile_error_autocluster = str(e)
        logger.error(f"Error executing {test_file}: {e}")
    
    if whitefox_logger:
        whitefox_logger.trace(f"          [HARNESS] Returning result")
    
    return result
