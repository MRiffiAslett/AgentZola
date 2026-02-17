import json
import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from domain.harness import ExecutionResult

logger = logging.getLogger(__name__)

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

            process = subprocess.run(
                [sys.executable, "-c", wrapper_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = process.stdout + process.stderr

            log_text_from_json = None
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
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON result from {test_file}")

            if log_text_from_json is not None and log_text_from_json:
                result.log_text = log_text_from_json
            else:
                result.log_text = output

            result.triggered_passes = self.extract_triggered_passes(result.log_text)

        except subprocess.TimeoutExpired:
            for mode in modes:
                result.get_mode(mode).runtime_error = "Execution timeout"
            logger.warning(f"Test {test_file} timed out after {timeout} seconds")
        except Exception as e:
            for mode in modes:
                result.get_mode(mode).compile_error = str(e)
            logger.error(f"Error executing {test_file}: {e}")

        return result
