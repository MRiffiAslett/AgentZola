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
import os
import re
import resource
import threading
import time
from pathlib import Path

_mem_limit_gb = int(os.environ.get('WHITEFOX_TEST_MEM_LIMIT_GB', '8'))
_mem_limit_bytes = _mem_limit_gb * 1024 ** 3

# 1. RLIMIT_AS: cap virtual address space.  Add 4 GB headroom for TF/XLA
#    shared libraries (~5-6 GB of mmap'd .so files) so TF can load while
#    still blocking pathological allocations (e.g. 40 GB tensor).
#    preexec_fn sets a coarser outer limit; this tightens it after exec.
_rlimit_as_bytes = (_mem_limit_gb + 4) * 1024 ** 3
try:
    resource.setrlimit(resource.RLIMIT_AS, (_rlimit_as_bytes, _rlimit_as_bytes))
except (ValueError, resource.error) as _e:
    print(f"WHITEFOX_WRAPPER: RLIMIT_AS={{_rlimit_as_bytes}} FAILED: {{_e}}", file=sys.stderr, flush=True)
_actual_soft, _actual_hard = resource.getrlimit(resource.RLIMIT_AS)
if _actual_soft != _rlimit_as_bytes:
    print(
        f"WHITEFOX_WRAPPER: RLIMIT_AS mismatch: wanted={{_rlimit_as_bytes}}, "
        f"got soft={{_actual_soft}} hard={{_actual_hard}}",
        file=sys.stderr, flush=True,
    )
try:
    resource.setrlimit(resource.RLIMIT_DATA, (_mem_limit_bytes, _mem_limit_bytes))
except (ValueError, resource.error):
    pass

# 2. oom_score_adj: make the OOM killer target this subprocess first,
#    protecting the parent generator (vLLM + state).  Also set in
#    preexec_fn, but repeat here in case the wrapper was launched
#    without preexec_fn.
try:
    with open('/proc/self/oom_score_adj', 'w') as _f:
        _f.write('1000')
except Exception:
    pass

def _mem_mb():
    rss = 0.0
    vsz = 0.0
    try:
        with open('/proc/self/status') as _f:
            for _line in _f:
                if _line.startswith('VmRSS:'):
                    rss = int(_line.split()[1]) / 1024.0
                elif _line.startswith('VmSize:'):
                    vsz = int(_line.split()[1]) / 1024.0
    except Exception:
        pass
    return rss, vsz

def _rss_mb():
    return _mem_mb()[0]

_RSS_LIMIT_MB = _mem_limit_gb * 1024
_VSZ_LIMIT_MB = _mem_limit_gb * 3 * 1024

# 3. Memory watchdog: polls RSS/VmSize every 0.1s and hard-kills if over limit.
#    RLIMIT_AS blocks new virtual mappings but existing ones (TF runtime
#    buffers, XLA temp arenas) can still dirty pages and grow RSS.  The
#    watchdog catches RSS growth that RLIMIT_AS cannot prevent.
_watchdog_stop = threading.Event()

def _mem_watchdog():
    while not _watchdog_stop.is_set():
        rss, vsz = _mem_mb()
        if rss > _RSS_LIMIT_MB:
            print(
                f"WHITEFOX_OOM_WATCHDOG: RSS {{rss:.0f}} MB > {{_RSS_LIMIT_MB}} MB — killing subprocess",
                file=sys.stderr, flush=True,
            )
            os._exit(137)
        if vsz > _VSZ_LIMIT_MB:
            print(
                f"WHITEFOX_OOM_WATCHDOG: VmSize {{vsz:.0f}} MB > {{_VSZ_LIMIT_MB}} MB — killing subprocess",
                file=sys.stderr, flush=True,
            )
            os._exit(137)
        _watchdog_stop.wait(0.1)

_watchdog_thread = threading.Thread(target=_mem_watchdog, daemon=True)
_watchdog_thread.start()

# 4. Static tensor-size pre-check: estimate total bytes from array/tensor
#    creation calls in the generated code and reject before exec().
_DTYPE_BYTES = {{
    'float64': 8, 'float32': 4, 'float16': 2, 'bfloat16': 2,
    'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1,
    'bool': 1, 'complex64': 8, 'complex128': 16,
    'tf.float64': 8, 'tf.float32': 4, 'tf.float16': 2, 'tf.bfloat16': 2,
    'tf.int64': 8, 'tf.int32': 4, 'tf.int16': 2, 'tf.int8': 1, 'tf.uint8': 1,
    'tf.bool': 1, 'tf.complex64': 8, 'tf.complex128': 16,
    'np.float64': 8, 'np.float32': 4, 'np.float16': 2,
    'np.int64': 8, 'np.int32': 4, 'np.int16': 2, 'np.int8': 1, 'np.uint8': 1,
}}
# Pattern A: shape in brackets/parens — tf.zeros([d,d]) / np.ones((d,d))
_TENSOR_SHAPE_RE = re.compile(
    r'(?:tf\\.(?:zeros|ones|random\\.(?:normal|uniform|truncated_normal)|fill|constant)'
    r'|np\\.(?:zeros|ones|random\\.(?:randn?|uniform|normal)|full|empty))'
    r'\\s*\\('
    r'[^)]*?'
    r'(?:shape\\s*=\\s*|size\\s*=\\s*)?'
    r'(?:\\[|\\()'
    r'([\\d,\\s]+)'
    r'(?:\\]|\\))',
)
# Pattern B: bare positional args — np.random.randn(d, d)
_TENSOR_BARE_RE = re.compile(
    r'np\\.random\\.(?:randn?|uniform|normal|standard_normal)'
    r'\\s*\\(\\s*'
    r'([\\d,\\s]+)'
    r'\\s*\\)',
)

def _estimate_tensor_bytes(code_str):
    total = 0
    for pat in (_TENSOR_SHAPE_RE, _TENSOR_BARE_RE):
        for m in pat.finditer(code_str):
            try:
                dims = [int(d.strip()) for d in m.group(1).split(',') if d.strip()]
                numel = 1
                for d in dims:
                    numel *= d
                total += numel * 4
            except (ValueError, OverflowError):
                total += _mem_limit_bytes
    return total

_MAX_STATIC_BYTES = _mem_limit_bytes // 2

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

    _est_bytes = _estimate_tensor_bytes(test_code)
    if _est_bytes > _MAX_STATIC_BYTES:
        _msg = (
            f"Static pre-check: estimated {{_est_bytes / 1e9:.1f}} GB tensor allocations "
            f"exceeds {{_MAX_STATIC_BYTES / 1e9:.1f}} GB limit — skipping test"
        )
        result["compile_error_naive"] = _msg
        result["compile_error_xla"] = _msg
        result["compile_error_autocluster"] = _msg
        raise MemoryError(_msg)
    
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
    
    _rss_after_naive = _rss_mb()
    if _rss_after_naive > _RSS_LIMIT_MB:
        _skip = f"RSS {{_rss_after_naive:.0f}} MB > {{_RSS_LIMIT_MB}} MB after naive; skipping xla+autocluster"
        result["compile_error_xla"] = _skip
        result["compile_error_autocluster"] = _skip
        raise MemoryError(_skip)

    try:
        # Re-seed before xla mode: naive's random.random() / np.random / tf.random
        # calls advanced the shared PRNG state.  Without re-seeding, xla draws
        # different values than naive — any code branching on random.random()
        # takes a different path, producing a false AllDiff oracle report.
        random.seed({RANDOM_SEED})
        np.random.seed({RANDOM_SEED})
        tf.random.set_seed({RANDOM_SEED})

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
    
    _rss_after_xla = _rss_mb()
    if _rss_after_xla > _RSS_LIMIT_MB:
        _skip = f"RSS {{_rss_after_xla:.0f}} MB > {{_RSS_LIMIT_MB}} MB after xla; skipping autocluster"
        result["compile_error_autocluster"] = _skip
        raise MemoryError(_skip)

    try:
        # Re-seed before autocluster mode (see xla block above for rationale).
        random.seed({RANDOM_SEED})
        np.random.seed({RANDOM_SEED})
        tf.random.set_seed({RANDOM_SEED})

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
    _watchdog_stop.set()
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
