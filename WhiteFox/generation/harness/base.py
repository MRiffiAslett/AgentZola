import json
import logging
import os
import resource
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from domain.harness import ExecutionResult

logger = logging.getLogger(__name__)


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

            process = subprocess.run(
                [sys.executable, "-c", wrapper_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                preexec_fn=_child_preexec,
            )

            output = process.stdout + process.stderr

            logger.debug(
                "[%s] Subprocess exit_code=%d, stdout=%d chars, stderr=%d chars",
                optimization_name, process.returncode,
                len(process.stdout), len(process.stderr),
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

            if log_text_from_json is not None and log_text_from_json:
                result.log_text = log_text_from_json
            else:
                result.log_text = output

            # Prefer the pass set the wrapper already extracted (cheap IPC,
            # bounded size) over re-regexing the raw log here.  Falling back
            # to extract_triggered_passes covers older wrappers and the case
            # where JSON parsing failed and result.log_text holds raw output.
            if passes_from_json is not None:
                result.triggered_passes = set(passes_from_json)
            else:
                result.triggered_passes = self.extract_triggered_passes(result.log_text)

            if result.triggered_passes:
                logger.info(
                    "[%s] %s triggered passes: %s",
                    optimization_name, test_file.name,
                    sorted(result.triggered_passes),
                )

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
