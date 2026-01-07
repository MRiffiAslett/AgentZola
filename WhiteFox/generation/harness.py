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
    if "    def call" in code:
        code = code.replace("    def call", f"    {decorator}\n    def call")
    else:
        code = code.replace("  def call", f"  {decorator}\n  def call")
    return code


def extract_input_variable(code: str, test_globals: dict) -> tuple:
    """
    Extract input variable(s) from code by finding model call.
    
    Looks for patterns like:
    - y = m(x1)
    - y = m(x)
    - y = m.call(x1)
    
    Returns (input_var_name, input_value) or (None, None) if not found.
    """
    import re
    
    patterns = [
        r'\bm\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bm\.call\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            var_name = match.group(1)
            if var_name in test_globals:
                return var_name, test_globals[var_name]
    
    common_names = ['x1', 'x', 'input_data', 'inputs', 'input']
    for name in common_names:
        if name in test_globals:
            value = test_globals[name]
            if hasattr(value, 'shape') or isinstance(value, (list, tuple)):
                return name, value
    
    return None, None


def execute_test_in_subprocess(
    test_file: Path, 
    whitefox_logger=None,
    optimization_name: Optional[str] = None,
    iteration: Optional[int] = None,
    sample_idx: Optional[int] = None,
    timeout: int = 60
) -> ExecutionResult:
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
            whitefox_logger.log_diagnostic(
                optimization_name or "unknown",
                iteration or 0,
                sample_idx or 0,
                "code_read",
                "success",
                {"test_file": str(test_file), "code_length": len(test_code)}
            )
    except Exception as e:
        result.compile_error_naive = str(e)
        result.compile_error_xla = str(e)
        result.compile_error_autocluster = str(e)
        if whitefox_logger:
            whitefox_logger.log_diagnostic(
                optimization_name or "unknown",
                iteration or 0,
                sample_idx or 0,
                "code_read",
                "failure",
                {"test_file": str(test_file), "error": str(e)}
            )
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


def _add_decorator(code, decorator):
    if "    def call" in code:
        code = code.replace("    def call", decorator + "\\n    def call")
    else:
        code = code.replace("  def call", decorator + "\\n  def call")
    return code


def _extract_input_variable(code, test_globals):
    import re
    patterns = [
        r'\\bm\\s*\\(\\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\\bm\\.call\\s*\\(\\s*([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    for pattern in patterns:
        try:
            match = re.search(pattern, code)
            if match:
                var_name = match.group(1)
                if var_name in test_globals:
                    return var_name, test_globals[var_name]
        except Exception:
            pass
    common_names = ['x1', 'x', 'input_data', 'inputs', 'input']
    for name in common_names:
        if name in test_globals:
            value = test_globals[name]
            try:
                if hasattr(value, 'shape') or isinstance(value, (list, tuple, dict)):
                    return name, value
            except Exception:
                pass
    return None, None


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

# Set TF_XLA_FLAGS for autocluster mode (will be overridden later for XLA mode)
if not old_tf_xla_flags_env:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

try:
    import tensorflow as tf
    import numpy as np
    import random
    
    random.seed({RANDOM_SEED})
    np.random.seed({RANDOM_SEED})
    tf.random.set_seed({RANDOM_SEED})
    
    test_globals = {{'__name__': '__main__'}}.copy()
    test_code = {test_code_repr}
    
    model_key = 'm'
    input_data_key = 'input_data'
    initial_exec_success = False
    try:
        exec(test_code, test_globals)
        result["compile_success_naive"] = True
        result["compile_success_xla"] = True
        result["compile_success_autocluster"] = True
        initial_exec_success = True
    except Exception as e:
        error_msg = str(e) + "\\n" + traceback.format_exc()
        result["compile_error_naive"] = error_msg
        result["compile_error_xla"] = error_msg
        result["compile_error_autocluster"] = error_msg
        # Don't raise - continue to try XLA execution even if initial exec failed
        # XLA execution is in its own try block and will handle errors gracefully
    
    if model_key not in test_globals:
        # Model not found - XLA execution will fail gracefully in its try block
        pass
    else:
        m = test_globals[model_key]
        
        if input_data_key in test_globals:
            input_data = test_globals[input_data_key]
        else:
            input_var_name, input_data = _extract_input_variable(test_code, test_globals)
        
        if input_data is not None:
            if not isinstance(input_data, (list, tuple)):
                input_data = [input_data]
            
            try:
                output_naive = m(*input_data)
                result["runtime_success_naive"] = True
                result["output_naive"] = _serialize_output(output_naive)
            except Exception as e:
                result["runtime_error_naive"] = str(e) + "\\n" + traceback.format_exc()
    
    try:
        # XLA_FLAGS already set before TensorFlow import, so pass logging should work
        xla_code = _add_decorator(test_code, "@tf.function(jit_compile=True)")
        xla_globals = {{'__name__': '__main__'}}.copy()
        xla_globals.update({{
            'tf': tf,
            'np': np,
            'random': random,
        }})
        exec(xla_code, xla_globals)
        m_xla = xla_globals[model_key]
        if input_data_key in xla_globals:
            input_data_xla = xla_globals[input_data_key]
        else:
            _, input_data_xla = _extract_input_variable(xla_code, xla_globals)
            if input_data_xla is None:
                raise Exception("Could not find input data in XLA execution")
        if not isinstance(input_data_xla, (list, tuple)):
            input_data_xla = [input_data_xla]
        output_xla = m_xla(*input_data_xla)
        result["runtime_success_xla"] = True
        result["output_xla"] = _serialize_output(output_xla)
    except Exception as e:
        result["runtime_error_xla"] = str(e) + "\\n" + traceback.format_exc()
    
    try:
        # TF_XLA_FLAGS already set before TensorFlow import for autocluster mode
        ac_code = _add_decorator(test_code, "@tf.function")
        ac_globals = {{'__name__': '__main__'}}.copy()
        ac_globals.update({{
            'tf': tf,
            'np': np,
            'random': random,
        }})
        exec(ac_code, ac_globals)
        m_ac = ac_globals[model_key]
        if input_data_key in ac_globals:
            input_data_ac = ac_globals[input_data_key]
        else:
            _, input_data_ac = _extract_input_variable(ac_code, ac_globals)
            if input_data_ac is None:
                raise Exception("Could not find input data in autocluster execution")
        if not isinstance(input_data_ac, (list, tuple)):
            input_data_ac = [input_data_ac]
        output_ac = m_ac(*input_data_ac)
        result["runtime_success_autocluster"] = True
        result["output_autocluster"] = _serialize_output(output_ac)
    except Exception as e:
        result["runtime_error_autocluster"] = str(e) + "\\n" + traceback.format_exc()

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
        
        # Log diagnostic information after execution
        if whitefox_logger:
            # Determine execution stage status
            exec_stage_status = "failure"
            exec_stage_details = {}
            if result.compile_error_naive:
                exec_stage_status = "failure"
                exec_stage_details = {
                    "error": result.compile_error_naive[:500] if result.compile_error_naive else "Unknown error",
                    "error_type": "compile_error"
                }
            elif result.runtime_error_naive:
                exec_stage_status = "failure"
                exec_stage_details = {
                    "error": result.runtime_error_naive[:500] if result.runtime_error_naive else "Unknown error",
                    "error_type": "runtime_error"
                }
            elif result.compile_success_naive and result.runtime_success_naive:
                exec_stage_status = "success"
                exec_stage_details = {"model_executed": True}
            
            # Log initial execution stage
            whitefox_logger.log_diagnostic(
                optimization_name or "unknown",
                iteration or 0,
                sample_idx or 0,
                "exec_initial",
                exec_stage_status,
                exec_stage_details
            )
            
            # Log XLA execution stage
            xla_stage_status = "not_run"
            xla_stage_details = {}
            if result.compile_success_xla:
                if result.runtime_success_xla:
                    xla_stage_status = "success"
                    xla_stage_details = {"xla_executed": True, "log_text_length": len(result.log_text)}
                elif result.runtime_error_xla:
                    xla_stage_status = "failure"
                    xla_stage_details = {
                        "error": result.runtime_error_xla[:500] if result.runtime_error_xla else "Unknown error",
                        "error_type": "xla_runtime_error"
                    }
            elif result.compile_error_xla:
                xla_stage_status = "failure"
                xla_stage_details = {
                    "error": result.compile_error_xla[:500] if result.compile_error_xla else "Unknown error",
                    "error_type": "xla_compile_error"
                }
            
            whitefox_logger.log_diagnostic(
                optimization_name or "unknown",
                iteration or 0,
                sample_idx or 0,
                "xla_exec",
                xla_stage_status,
                xla_stage_details
            )
            
            # Log pass detection stage
            whitefox_logger.log_diagnostic(
                optimization_name or "unknown",
                iteration or 0,
                sample_idx or 0,
                "pass_detection",
                "success" if result.triggered_passes else "no_passes",
                {
                    "log_text_length": len(result.log_text),
                    "has_markers": "WHITEFOX_PASS_START" in result.log_text,
                    "triggered_passes": list(result.triggered_passes),
                    "log_text_from_json": log_text_from_json is not None and bool(log_text_from_json),
                    "xla_compile_success": result.compile_success_xla,
                    "xla_runtime_success": result.runtime_success_xla,
                }
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
        result.runtime_error_naive = "Execution timeout"
        result.runtime_error_xla = "Execution timeout"
        result.runtime_error_autocluster = "Execution timeout"
        logger.warning(f"Test {test_file} timed out after {timeout} seconds")
    except Exception as e:
        result.compile_error_naive = str(e)
        result.compile_error_xla = str(e)
        result.compile_error_autocluster = str(e)
        logger.error(f"Error executing {test_file}: {e}")
    
    return result
