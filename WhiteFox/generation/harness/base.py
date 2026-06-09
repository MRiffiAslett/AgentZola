import json
import logging
import os
import resource
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from domain.harness import ExecutionResult

logger = logging.getLogger(__name__)


class _BoundedTail:
    """Drain a subprocess pipe in a background thread and retain only the
    last `cap` bytes.

    Background:
        subprocess.run(capture_output=True) reads each pipe fully into a
        Python str/bytes object with no upper bound.  For tests where the
        wrapper emits megabytes of fd-2 output (TF FATAL stack traces,
        --xla_dump_hlo_pass_re=.* HLO dumps under collective opts, or a
        --xla_dump_to write that hit stderr) that buffer can reach
        gigabytes inside a single ProcessPoolExecutor worker — and 4
        workers holding several GB each, then pickling the contents back
        to the parent at iter boundary, pushed the parent past the cgroup
        limit and OOM-killed job 244274_0 mid-AllReduceCombiner.

        This reader keeps only the *tail* of each pipe in RAM, dropping
        older bytes on the fly, so worker RSS attributable to subprocess
        captures stays at most `cap` per stream regardless of how much
        the wrapper writes.  We keep the tail rather than the head because
        the parser looks for WHITEFOX_RESULT_START/END markers and the
        JSON payload, which the wrapper prints last.
    """

    def __init__(self, stream, cap: int = 1024 * 1024) -> None:
        self._cap = cap
        self._buf = bytearray()
        self._stream = stream
        self._total = 0
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read(65536)
                if not chunk:
                    return
                with self._lock:
                    self._total += len(chunk)
                    self._buf.extend(chunk)
                    excess = len(self._buf) - self._cap
                    if excess > 0:
                        del self._buf[:excess]
        except Exception:
            return

    def text(self) -> str:
        self._t.join()
        with self._lock:
            return self._buf.decode("utf-8", errors="replace")

    def total_bytes(self) -> int:
        return self._total


def _child_preexec() -> None:
    """preexec_fn for test subprocesses — runs after fork() before exec()
    so RLIMIT_AS applies before the interpreter loads libc/libpython."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:
        pass
    mem_gb = int(os.environ.get("WHITEFOX_TEST_MEM_LIMIT_GB", "8"))
    rlimit_bytes = mem_gb * 3 * (1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (rlimit_bytes, rlimit_bytes))
    except Exception as exc:
        print(
            f"WHITEFOX_PREEXEC: RLIMIT_AS={rlimit_bytes} FAILED: {exc}",
            file=sys.stderr, flush=True,
        )
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft != rlimit_bytes:
            print(
                f"WHITEFOX_PREEXEC: RLIMIT_AS mismatch: "
                f"wanted={rlimit_bytes}, got soft={soft} hard={hard}",
                file=sys.stderr, flush=True,
            )
    except Exception:
        pass

RANDOM_SEED = 42


class TestHarness(ABC):
    @abstractmethod
    def get_execution_modes(self) -> List[str]: ...

    @abstractmethod
    def generate_wrapper_script(self, test_code_repr: str) -> str: ...

    @abstractmethod
    def get_env_vars(self) -> Dict[str, str]: ...

    @abstractmethod
    def extract_triggered_passes(self, log_text: str) -> set: ...

    def execute_test_in_subprocess(
        self,
        test_file: Path,
        whitefox_logger=None,
        optimization_name: Optional[str] = None,
        iteration: Optional[int] = None,
        sample_idx: Optional[int] = None,
        timeout: int = 60,
    ) -> ExecutionResult:
        modes = self.get_execution_modes()
        result = ExecutionResult(test_file=test_file)

        for mode in modes:
            mode_result = result.get_mode(mode)
            mode_result.compile_success = False
            mode_result.runtime_success = False

        try:
            with open(test_file, "r") as f:
                test_code = f.read()
        except Exception as e:
            for mode in modes:
                result.get_mode(mode).compile_error = str(e)
            return result

        test_code_repr = repr(test_code)

        try:
            wrapper_script = self.generate_wrapper_script(test_code_repr)
        except Exception as e:
            for mode in modes:
                result.get_mode(
                    mode
                ).compile_error = f"Wrapper script creation failed: {str(e)}"
            return result

        try:
            env = os.environ.copy()
            env.update(self.get_env_vars())

            logger.debug(
                "[%s] Running subprocess for %s (timeout=%ds)",
                optimization_name, test_file.name, timeout,
            )

            # Use Popen + bounded-tail readers instead of subprocess.run
            # so worker RSS attributable to captured stdout/stderr is
            # bounded by _PIPE_CAP per stream regardless of how much the
            # wrapper writes.  See the _BoundedTail docstring for the
            # full rationale (job 244274_0 OOM).
            _PIPE_CAP = int(
                os.environ.get("WHITEFOX_PIPE_CAP_BYTES", str(1024 * 1024))
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", wrapper_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=_child_preexec,
            )
            out_reader = _BoundedTail(proc.stdout, cap=_PIPE_CAP)
            err_reader = _BoundedTail(proc.stderr, cap=_PIPE_CAP)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                # Surface this through the same path as the original code
                # by re-raising — the outer except handles it identically.
                raise

            stdout_text = out_reader.text()
            stderr_text = err_reader.text()
            process = subprocess.CompletedProcess(
                args=proc.args,
                returncode=proc.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
            )
            output = process.stdout + process.stderr

            logger.debug(
                "[%s] Subprocess exit_code=%d, stdout=%d chars (of %d emitted), "
                "stderr=%d chars (of %d emitted)",
                optimization_name, process.returncode,
                len(process.stdout), out_reader.total_bytes(),
                len(process.stderr), err_reader.total_bytes(),
            )

            log_text_from_json = None
            passes_from_json = None
            if "WHITEFOX_RESULT_START" in output and "WHITEFOX_RESULT_END" in output:
                start_idx = output.find("WHITEFOX_RESULT_START") + len(
                    "WHITEFOX_RESULT_START"
                )
                end_idx = output.find("WHITEFOX_RESULT_END")
                json_str = output[start_idx:end_idx].strip()
                try:
                    result_data = json.loads(json_str)

                    for mode in modes:
                        mr = result.get_mode(mode)
                        mr.compile_success = result_data.get(
                            f"compile_success_{mode}", False
                        )
                        mr.runtime_success = result_data.get(
                            f"runtime_success_{mode}", False
                        )
                        mr.compile_error = result_data.get(f"compile_error_{mode}")
                        mr.runtime_error = result_data.get(f"runtime_error_{mode}")
                        mr.output = result_data.get(f"output_{mode}")

                    log_text_from_json = result_data.get("log_text", "")
                    passes_from_json = result_data.get("triggered_passes")

                    mode_summary = {
                        m: ("OK" if result_data.get(f"runtime_success_{m}") else "FAIL")
                        for m in modes
                    }
                    logger.info(
                        "[%s] %s modes: %s",
                        optimization_name, test_file.name, mode_summary,
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "[%s] Failed to parse JSON result from %s",
                        optimization_name, test_file,
                    )
            else:
                logger.warning(
                    "[%s] No WHITEFOX_RESULT markers in output of %s "
                    "(exit_code=%d, output_len=%d)",
                    optimization_name, test_file.name,
                    process.returncode, len(output),
                )
                if process.returncode != 0 and process.stderr:
                    for line in process.stderr.strip().splitlines()[-3:]:
                        logger.warning("  stderr: %s", line)

            # Always cap log_text.  Previously the "log_text_from_json
            # truthy" branch fell through to `result.log_text = output`
            # whenever the wrapper's StringIO was empty (i.e. the test
            # produced no Python-level stdout/stderr).  After f7a6d57 that
            # is the *common* case, because TF's C++ logging — most
            # importantly the per-XLA-pass dump notifications emitted under
            # `--xla_dump_hlo_pass_re=.*` for collective opts like
            # AllReduceCombiner — writes directly to fd 2 via write(2,…)
            # and bypasses Python's sys.stderr entirely.  Those C++ writes
            # still land in the parent's `process.stderr` (and therefore in
            # `output`), and at 10s of MB per test × 4 parallel workers
            # they drove the parent's RSS from 1.4 GB → 188 GB in ~30 min
            # in job 242824_0 even with the TeeOutput leak plugged.
            _MAX_LOG_CHARS = 8192
            chosen_log = log_text_from_json if log_text_from_json else output
            result.log_text = (
                chosen_log[-_MAX_LOG_CHARS:]
                if len(chosen_log) > _MAX_LOG_CHARS
                else chosen_log
            )
            # chosen_log aliases `output` in the empty-JSON branch — release
            # the alias before we hit the regex pass below or `del output`
            # at the end leaves the original 100 MB string alive via this
            # local reference.
            del chosen_log

            # Always scan the raw subprocess output (C++ fd 2) for
            # WHITEFOX_PASS_START markers and merge with whatever the wrapper
            # extracted from its Python-level streams.
            #
            # History: the wrapper redirects sys.stderr to a StringIO so
            # Python-level output is captured, but TF/XLA's C++ runtime writes
            # pass notifications directly to fd 2 via write(2,...), bypassing
            # the Python redirect.  Those markers land in process.stderr (and
            # therefore in `output`) but NOT in the wrapper's StringIO, so
            # result["triggered_passes"] in the JSON is always [].
            #
            # Previously we fell back to scanning `output` only when
            # passes_from_json was None (subprocess crashed without markers).
            # After removing RLIMIT_AS from pool workers (f892797) tests
            # complete normally, passes_from_json is always [] (not None),
            # and the fallback never fired — Triggered=0 for all passes.
            #
            # The fix: always merge both sources.  `output` is already in
            # memory here and is del'd three lines below, so the regex scan
            # costs nothing extra in terms of peak RSS.
            passes_set = set(passes_from_json) if passes_from_json is not None else set()
            passes_set |= self.extract_triggered_passes(output)
            result.triggered_passes = passes_set

            if result.triggered_passes:
                logger.info(
                    "[%s] %s triggered passes: %s",
                    optimization_name, test_file.name,
                    sorted(result.triggered_passes),
                )

            # Explicitly release the (potentially 10s-of-MB) captured
            # stdout/stderr buffers before this worker picks up its next
            # test.  Without this, with 4 parallel workers each holding a
            # ~50 MB CompletedProcess + concatenated `output` while
            # waiting on the next subprocess.run, the parent's transient
            # working set grows monotonically and CPython's allocator
            # does not return the freed pages between iterations fast
            # enough to keep up with cgroup pressure.
            del output
            process.stdout = ""
            process.stderr = ""

        except subprocess.TimeoutExpired:
            for mode in modes:
                result.get_mode(mode).runtime_error = "Execution timeout"
            logger.warning(
                "[%s] Test %s timed out after %d seconds",
                optimization_name, test_file, timeout,
            )
        except Exception as e:
            for mode in modes:
                result.get_mode(mode).compile_error = str(e)
            logger.error(
                "[%s] Error executing %s: %s",
                optimization_name, test_file, e,
            )

        return result
