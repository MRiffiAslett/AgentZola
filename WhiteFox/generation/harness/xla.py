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
# Do NOT tee Python-level stdout/stderr to the real fd 1 / fd 2.  Tests can
# easily print megabyte-scale tracebacks (e.g. TF's OOM error includes the
# full failing HLO) or even whole tensors; teeing them to original_stdout
# means the parent's subprocess.run(capture_output=True) absorbs all of it
# into RAM as a Python string, and CPython's allocator does not return
# those arenas to the OS — job 241012_0 grew the parent's RSS 1.3 GB ->
# 197 GB at a steady ~10 MB/s during AllReduceCombiner iter 12 and OOMed.
# Sending writes only to the in-process StringIO keeps the leak bounded
# inside the wrapper subprocess (which has RLIMIT_AS=10 GB) and the
# 8 KB log_text truncation in the finally block keeps the parent payload
# small.  The fd 1 / fd 2 streams are still used for the WHITEFOX_RESULT
# markers + JSON, which are printed after the finally block restores
# sys.stdout / sys.stderr to their originals.
sys.stdout = stdout_capture
sys.stderr = stderr_capture

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
    # IPC payload cap.  Job 244274_0 died because the previous
    # implementation only recursed one level: a test returning
    # `[tensor.numpy().tolist()]` (a list containing an already-flattened
    # nested-Python-list rather than a tf.Tensor) sailed past the per-tensor
    # elem cap and shipped the full nested structure to the parent.  With
    # 10 such results held in `_execute_tests_parallel.results` at iter
    # boundary × 3 modes the parent's RSS jumped 2 GB → 175 GB in one iter
    # of AllReduceCombiner and OOMed at 190 GB.
    #
    # The new policy: enforce a *total leaf-element budget* per call.  The
    # recursion descends through tf.Tensor, numpy.ndarray, list, tuple and
    # dict, decrementing a shared budget as it emits scalars.  Anything
    # past the budget is replaced with a "<truncated_at_N>" marker so the
    # diff oracle still sees a structurally-comparable value, but the
    # cross-process JSON cannot exceed ~(budget × ~15 bytes/float) ≈ a few
    # MB per mode regardless of how the test code wraps its output.
    _MAX_TOTAL_ELEMS = int(os.environ.get("WHITEFOX_MAX_OUTPUT_ELEMS", "10000"))
    _budget = [_MAX_TOTAL_ELEMS]

    def _try_keras_tensor_repr(v):
        try:
            if tf.keras.backend.is_keras_tensor(v):
                return (
                    "<KerasTensor shape="
                    + str(tuple(v.shape))
                    + " dtype="
                    + str(v.dtype)
                    + ">"
                )
        except Exception:
            pass
        return None

    def _walk(v):
        if _budget[0] <= 0:
            return "<truncated_at_" + str(_MAX_TOTAL_ELEMS) + ">"

        # Scalars: count as 1, return verbatim.
        if v is None or isinstance(v, (bool, int, float, str)):
            _budget[0] -= 1
            return v

        # tf.Tensor and np.ndarray: convert and recurse on the .tolist().
        # We re-enter _walk on the resulting nested list so the budget
        # applies even when the tensor is huge.  arr.flatten()[:budget]
        # keeps the conversion itself bounded for multi-GB tensors.
        if isinstance(v, tf.Tensor):
            try:
                arr = v.numpy()
            except Exception:
                # KerasTensor or symbolic tensor — emit a stable repr.
                kt = _try_keras_tensor_repr(v)
                return kt if kt is not None else "<TFTensor>"
        else:
            try:
                import numpy as _np_local
                if isinstance(v, _np_local.ndarray):
                    arr = v
                else:
                    arr = None
            except Exception:
                arr = None

        if arr is not None:
            if arr.size > _budget[0]:
                flat = arr.flatten()[: _budget[0]].tolist()
                _budget[0] = 0
                return flat + ["<truncated_at_" + str(_MAX_TOTAL_ELEMS) + ">"]
            _budget[0] -= int(arr.size)
            return arr.tolist()

        # Containers: descend, capping each level by remaining budget.
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                if _budget[0] <= 0:
                    out.append("<truncated_at_" + str(_MAX_TOTAL_ELEMS) + ">")
                    break
                out.append(_walk(item))
            return out
        if isinstance(v, dict):
            out = dict()
            for k, item in v.items():
                if _budget[0] <= 0:
                    out["__truncated__"] = "<truncated_at_" + str(_MAX_TOTAL_ELEMS) + ">"
                    break
                # Keys are usually short strings; if not, fall back to str().
                key = k if isinstance(k, (str, int, bool, float)) else str(k)
                out[key] = _walk(item)
            return out

        # Catch-all: KerasTensor (symbolic, no .numpy()) embeds a per-instance
        # counter in its str() that differs across modes — return a
        # counter-free representation so identical graphs serialise to the
        # same string.
        kt = _try_keras_tensor_repr(v)
        if kt is not None:
            return kt
        # Anything else (custom objects): str() with a hard cap so we don't
        # ship a 100 MB __repr__ across.
        s = str(v)
        if len(s) > 4096:
            s = s[:4096] + "...[truncated]"
        return s

    return _walk(output)


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

# Force absl/glog to write to fd 2 (the captured pipe) as well as its log
# file.  Without this, absl::InitializeLog() redirects LOG() messages to a
# /tmp/*.INFO file; WHITEFOX_PASS_START/END markers (logged via LOG(INFO) in
# the custom TF build) then only reach the parent when absl's SIGSEGV crash
# handler flushed the buffer — i.e. only for crashed tests (exit_code=-11).
# Normally-completing tests had triggered_passes={} even though XLA ran all
# passes, because the markers were in the log file, not in process.stderr.
# Setting GLOG_logtostderr=1 makes absl always mirror output to fd 2 so
# base.py's extract_triggered_passes(output) can find the markers regardless
# of exit code.  The _BoundedTail cap (default 1 MB) keeps worker RSS
# bounded; target passes run in the later XLA pipeline stages and reliably
# fall within the tail.
os.environ.setdefault('GLOG_logtostderr', '1')

try:
    import tensorflow as tf
    import numpy as np
    import random
    
    random.seed({RANDOM_SEED})
    np.random.seed({RANDOM_SEED})
    tf.random.set_seed({RANDOM_SEED})
    
    test_code = {test_code_repr}
    # Strip any user-supplied @tf.function(...) decorator immediately above
    # `def call` so all three modes (naive, xla, autocluster) start from the
    # same baseline.  Otherwise add_decorator_inline stacks a second decorator
    # on top of the user's, giving each mode an asymmetric ConcreteFunction
    # wrapper — for some models (e.g. tf.while_loop with overflow) this makes
    # naive return shape (0,) and xla return shape (4,), causing a false
    # "Length mismatch: 0 vs 4" AllDiff oracle report.
    import re as _re
    test_code = _re.sub(
        r'^[ \\t]*@tf\\.function(?:\\([^)]*\\))?[ \\t]*\\n(?=[ \\t]*def call\\b)',
        '',
        test_code,
        flags=_re.MULTILINE,
    )
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
    # With --xla_dump_hlo_pass_re=.* every test floods stderr with HLO-pass
    # markers (≥1 MB/test on opts like StochasticConvertDecomposer).  The
    # parent only ever uses log_text to regex out the pass-name set, so
    # extract that set here and send the parent the (≤~80-string) result;
    # ship back only a short tail of the raw log for debugging.  Without
    # this the parent's PyMalloc arena retained ~10s of MB per test and
    # anon RSS climbed to 190 GiB in job 240645_2 even after ca48f23.
    _log_combined = stdout_capture.getvalue() + stderr_capture.getvalue()
    result["triggered_passes"] = sorted(set(
        re.findall(r'WHITEFOX_PASS_START[^\\n]*\\bpass=([^\\s]+)', _log_combined)
    ))
    _MAX_LOG_CHARS = int(os.environ.get("WHITEFOX_MAX_LOG_CHARS", "8192"))
    if len(_log_combined) > _MAX_LOG_CHARS:
        _log_combined = _log_combined[-_MAX_LOG_CHARS:]
    result["log_text"] = _log_combined
    del _log_combined

    # Bulletproof IPC payload size: cap *every* string field in `result`,
    # not just log_text.  For collective opts like AllReduceCombiner, TF's
    # ResourceExhaustedError / UnimplementedError exception messages embed
    # the full failing HLO module — easily 10+ MB per error.  Without this
    # cap, three modes worth of `compile_error_<mode>` get JSON-serialized
    # at full size, sent to the parent, deserialized onto `mr.compile_error`,
    # and held on the result object for the iter's lifetime.  At 4 parallel
    # workers × 10 samples × 3 modes that's ~1 GB / iter held just in
    # error strings.  The wrapper-side cap means the parent never sees
    # more than ~80 KB of IPC payload per test regardless of what TF
    # decides to dump into an exception message.
    _MAX_FIELD_CHARS = int(os.environ.get("WHITEFOX_MAX_FIELD_CHARS",
                                          str(_MAX_LOG_CHARS)))
    for _k, _v in list(result.items()):
        if isinstance(_v, str) and len(_v) > _MAX_FIELD_CHARS:
            result[_k] = _v[-_MAX_FIELD_CHARS:] + "...[truncated]"

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
