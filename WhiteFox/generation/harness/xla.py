from typing import Dict, List

from generation.harness.base import RANDOM_SEED, TestHarness
from generation.wf_logging import extract_triggered_passes


class TensorFlowXLAHarness(TestHarness):

    def get_execution_modes(self) -> List[str]:
        return ["naive", "xla", "autocluster"]

    def get_env_vars(self) -> Dict[str, str]:
        return {}

    def extract_triggered_passes(self, log_text: str) -> set:
        return extract_triggered_passes(log_text)

    def generate_wrapper_script(self, test_code_repr: str) -> str:
        return f"""
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
                    pass

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
    if "    def call" in code:
        code = code.replace("    def call", "    " + decorator + "\\n    def call")
    else:
        code = code.replace("  def call", "  " + decorator + "\\n  def call")
    return code


import os
old_xla_flags_env = os.environ.get('XLA_FLAGS', '')
old_tf_xla_flags_env = os.environ.get('TF_XLA_FLAGS', '')

xla_flags_parts = []
if old_xla_flags_env:
    xla_flags_parts.append(old_xla_flags_env)
if '--xla_dump_to=' not in old_xla_flags_env:
    xla_flags_parts.append('--xla_dump_to=/tmp/xla_dump')
if '--xla_dump_hlo_pass_re=' not in old_xla_flags_env:
    xla_flags_parts.append('--xla_dump_hlo_pass_re=.*')
os.environ['XLA_FLAGS'] = ' '.join(xla_flags_parts) if xla_flags_parts else ''

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
    
    try:
        test_globals_naive = {{
            '__name__': '__main__',
            'tf': tf,
            'np': np,
            'random': random,
        }}.copy()
        exec(test_code, test_globals_naive)
        result["compile_success_naive"] = True
        
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
    
    try:
        test_code_xla = add_decorator_inline(test_code, "@tf.function(jit_compile=True)")
        test_globals_xla = {{
            '__name__': '__main__',
            'tf': tf,
            'np': np,
            'random': random,
        }}.copy()
        exec(test_code_xla, test_globals_xla)
        result["compile_success_xla"] = True
        
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
    
    try:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        
        test_code_ac = add_decorator_inline(test_code, "@tf.function")
        test_globals_ac = {{
            '__name__': '__main__',
            'tf': tf,
            'np': np,
            'random': random,
        }}.copy()
        exec(test_code_ac, test_globals_ac)
        result["compile_success_autocluster"] = True
        
        m_ac = test_globals_ac[model_key]
        input_data_ac = test_globals_ac[input_data_key]
        
        if not isinstance(input_data_ac, (list, tuple)):
            input_data_ac = [input_data_ac]
        
        output_ac = m_ac(*input_data_ac)
        result["runtime_success_autocluster"] = True
        result["output_autocluster"] = _serialize_output(output_ac)
        
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
