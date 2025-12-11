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
import logging

import numpy as np

from ..domain.harness import ExecutionResult
from .log_parser import extract_triggered_passes


logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def add_decorator(code: str, decorator: str) -> str:
    if "    def call" in code:
        code = code.replace("    def call", f"    {decorator}\n    def call")
    else:
        code = code.replace("  def call", f"  {decorator}\n  def call")
    return code


def process_code(code: str) -> str:
    if "__call__" in code and " def call(" not in code:
        if "class Model" in code:
            lines = code.split("\n")
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if "def __call__" in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + "def call(self, *args, **kwargs):")
                    new_lines.append(" " * (indent + 4) + "return self.__call__(*args, **kwargs)")
            code = "\n".join(new_lines)
    return code


def execute_test_in_subprocess(test_file: Path, timeout: int = 60) -> ExecutionResult:
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
    except Exception as e:
        result.compile_error_naive = str(e)
        result.compile_error_xla = str(e)
        result.compile_error_autocluster = str(e)
        return result
    
    test_code = process_code(test_code)
    
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


try:
    import tensorflow as tf
    import numpy as np
    import random
    import os
    
    random.seed({RANDOM_SEED})
    np.random.seed({RANDOM_SEED})
    tf.random.set_seed({RANDOM_SEED})
    
    test_globals = {{'__name__': '__main__'}}.copy()
    test_code = {test_code_repr}
    
    try:
        exec(test_code, test_globals)
        result["compile_success_naive"] = True
        result["compile_success_xla"] = True
        result["compile_success_autocluster"] = True
    except Exception as e:
        error_msg = str(e) + "\\n" + traceback.format_exc()
        result["compile_error_naive"] = error_msg
        result["compile_error_xla"] = error_msg
        result["compile_error_autocluster"] = error_msg
        raise
    
    if 'm' not in test_globals:
        raise Exception("Model 'm' not found in test code")
    if 'input_data' not in test_globals:
        raise Exception("input_data not found in test code")
    
    m = test_globals['m']
    input_data = test_globals['input_data']
    
    if not isinstance(input_data, (list, tuple)):
        input_data = [input_data]
    
    try:
        output_naive = m(*input_data)
        result["runtime_success_naive"] = True
        result["output_naive"] = _serialize_output(output_naive)
    except Exception as e:
        result["runtime_error_naive"] = str(e) + "\\n" + traceback.format_exc()
    
    try:
        xla_code = _add_decorator(test_code, "@tf.function(jit_compile=True)")
        xla_globals = {{'__name__': '__main__'}}.copy()
        xla_globals.update({{
            'tf': tf,
            'np': np,
            'random': random,
        }})
        exec(xla_code, xla_globals)
        m_xla = xla_globals['m']
        input_data_xla = xla_globals['input_data']
        if not isinstance(input_data_xla, (list, tuple)):
            input_data_xla = [input_data_xla]
        output_xla = m_xla(*input_data_xla)
        result["runtime_success_xla"] = True
        result["output_xla"] = _serialize_output(output_xla)
    except Exception as e:
        result["runtime_error_xla"] = str(e) + "\\n" + traceback.format_exc()
    
    try:
        old_xla_flags = os.environ.get('TF_XLA_FLAGS', '')
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        ac_code = _add_decorator(test_code, "@tf.function")
        ac_globals = {{'__name__': '__main__'}}.copy()
        ac_globals.update({{
            'tf': tf,
            'np': np,
            'random': random,
        }})
        exec(ac_code, ac_globals)
        m_ac = ac_globals['m']
        input_data_ac = ac_globals['input_data']
        if not isinstance(input_data_ac, (list, tuple)):
            input_data_ac = [input_data_ac]
        output_ac = m_ac(*input_data_ac)
        result["runtime_success_autocluster"] = True
        result["output_autocluster"] = _serialize_output(output_ac)
        os.environ['TF_XLA_FLAGS'] = old_xla_flags
    except Exception as e:
        result["runtime_error_autocluster"] = str(e) + "\\n" + traceback.format_exc()
        if 'old_xla_flags' in locals():
            os.environ['TF_XLA_FLAGS'] = old_xla_flags

except Exception as e:
    error_msg = str(e) + "\\n" + traceback.format_exc()
    if not result["compile_error_naive"]:
        result["compile_error_naive"] = error_msg
    if not result["compile_error_xla"]:
        result["compile_error_xla"] = error_msg
    if not result["compile_error_autocluster"]:
        result["compile_error_autocluster"] = error_msg

finally:
    result["log_text"] = stdout_capture.getvalue() + stderr_capture.getvalue()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

print("WHITEFOX_RESULT_START")
print(json.dumps(result))
print("WHITEFOX_RESULT_END")
"""
    
    try:
        process = subprocess.run(
            [sys.executable, "-c", wrapper_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = process.stdout + process.stderr
        result.log_text = output
        
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
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON result from {test_file}")
        
        result.triggered_passes = extract_triggered_passes(result.log_text)
        
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
