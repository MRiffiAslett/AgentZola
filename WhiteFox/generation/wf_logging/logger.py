import json
import logging
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from domain.bandit import OptimizationState, TriggeringTest


def extract_triggered_passes(log_text: str) -> Set[str]:
    passes = set()
    pattern = r"WHITEFOX_PASS_START[^\n]*\bpass=([^\s]+)"
    for match in re.finditer(pattern, log_text):
        pass_name = match.group(1)
        passes.add(pass_name)
    return passes


class WhiteFoxLogger:
    def __init__(self, log_dir: Path, base_logger: Optional[logging.Logger] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.source_dir = self.log_dir / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_file = self.source_dir / "prompts.json"
        self.cleaned_code_file = self.source_dir / "all_cleaned_code.json"
        self.bug_reports_file = self.source_dir / "bug_reports.json"
        self.bug_reports_filtered_file = self.source_dir / "bug_reports_filtered.json"
        self.diagnostic_file = self.source_dir / "execution_diagnostics.json"

        self.prompts_text_file = self.log_dir / "gen_prompts.log"
        self.cleaned_code_text_file = self.log_dir / "gen_code.log"

        self.prompts_data: Dict[str, List[Dict]] = {}
        self.cleaned_code_data: Dict[str, List[Dict]] = {}
        self.bug_reports_data: List[Dict] = []
        self.diagnostic_data: List[Dict] = []

        self.opt_stats: Dict[str, Dict[str, int]] = {}

        self.base_logger = base_logger or logging.getLogger(__name__)

        self._lock = threading.RLock()

    def trace(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        pass

    def clear_old_logs(self) -> None:
        all_files = [
            self.prompts_file,
            self.cleaned_code_file,
            self.bug_reports_file,
            self.bug_reports_filtered_file,
            self.diagnostic_file,
            self.prompts_text_file,
            self.cleaned_code_text_file,
        ]

        for log_file in all_files:
            if log_file.exists():
                log_file.unlink()
                if self.base_logger:
                    self.base_logger.debug(f"Cleared old log file: {log_file}")

        for data in [
            self.prompts_data,
            self.cleaned_code_data,
            self.bug_reports_data,
            self.diagnostic_data,
            self.opt_stats,
        ]:
            data.clear()

    @staticmethod
    def _get_default_stats() -> Dict[str, int]:
        return {
            "generated": 0,
            "triggered": 0,
        }

    def _ensure_stats_initialized(self, opt_key: str) -> None:
        if opt_key not in self.opt_stats:
            self.opt_stats[opt_key] = self._get_default_stats()

    def log_prompt(
        self,
        optimization_name: str,
        iteration: int,
        prompt_type: str,
        prompt_text: str,
        example_tests: Optional[List[TriggeringTest]] = None,
    ) -> None:
        with self._lock:
            if optimization_name not in self.prompts_data:
                self.prompts_data[optimization_name] = []

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "optimization": optimization_name,
                "iteration": iteration,
                "prompt_type": prompt_type,
                "prompt": prompt_text,
                "num_examples": len(example_tests) if example_tests else 0,
                "examples": [],
            }

            if example_tests:
                for test in example_tests:
                    example_info = {
                        "test_id": test.test_id,
                        "file_path": str(test.file_path),
                        "alpha": test.alpha,
                        "beta": test.beta,
                    }
                    log_entry["examples"].append(example_info)

            self.prompts_data[optimization_name].append(log_entry)
            self._write_prompts()

    def log_generated_code(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        raw_text: str,
        cleaned_code: str,
        parsed_code: Optional[str] = None,
    ) -> None:
        with self._lock:
            if optimization_name not in self.cleaned_code_data:
                self.cleaned_code_data[optimization_name] = []

            self.cleaned_code_data[optimization_name].append(
                {
                    "optimization": optimization_name,
                    "iteration": iteration,
                    "sample_idx": sample_idx,
                    "code": cleaned_code,
                }
            )

            self._ensure_stats_initialized(optimization_name)
            self.opt_stats[optimization_name]["generated"] += 1

            self._write_cleaned_code()

    def log_execution_result(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        test_file: Path,
        result: Any,
        pass_triggered: bool,
        pass_log_name: str,
    ) -> None:
        self._track_execution_stats(optimization_name, result, pass_triggered)

    def log_pass_detection_analysis(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        pass_log_name: str,
        log_text: str,
        triggered_passes: set,
        expected_pass: str,
        expected_passes: Optional[List[str]] = None,
    ) -> None:
        pass

    def log_state_update(
        self,
        optimization_name: str,
        iteration: int,
        before_state: OptimizationState,
        after_state: OptimizationState,
        num_triggered: int,
        num_not_triggered: int,
        new_triggering_tests: List[Path],
        example_tests_used: List[TriggeringTest],
    ) -> None:
        pass

    def log_error(
        self,
        optimization_name: str,
        iteration: Optional[int],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        self.base_logger.error(
            f"Error: {optimization_name} it{iteration} - {error_type}: {error_message}"
        )

    def log_bug_report(self, bug_report: Any) -> None:
        with self._lock:
            self.bug_reports_data.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "test_id": bug_report.test_id,
                    "oracle_type": bug_report.oracle_type,
                    "details": bug_report.details,
                    "test_file": str(bug_report.test_file),
                    "logs_file": str(bug_report.logs_file),
                }
            )
            self._write_bug_reports()

    def _write_json(self, file_path: Path, data: Any, **kwargs) -> None:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, **kwargs)

    def _write_prompts(self) -> None:
        self._write_json(self.prompts_file, self.prompts_data)
        self._write_prompts_readable()

    def _write_cleaned_code(self) -> None:
        self._write_json(
            self.cleaned_code_file, self.cleaned_code_data, ensure_ascii=False
        )
        self._write_code_readable()

    @staticmethod
    def _is_low_signal_report(entry: Dict) -> bool:

        if entry.get("oracle_type") == "AllDiff_Rand":
            return True

        num_diff = (entry.get("details") or {}).get("Num Diff")
        if not isinstance(num_diff, str):
            return False

        match = re.match(r"^Value mismatch:\s*(.+?)\s+vs\s+(.+)$", num_diff)
        if not match:
            return False

        lhs, rhs = match.group(1), match.group(2)

        def _represents_nan_or_inf(text: str) -> Optional[str]:

            cleaned = text.strip()
            cleaned = re.sub(
                r"^(?:tensor|tf\.Tensor|Tensor|array)\s*\(\s*", "", cleaned
            )
            cleaned = re.sub(r"\s*(?:,\s*dtype=[^)]+)?\)\s*$", "", cleaned)

            tokens_str = cleaned.replace("[", " ").replace("]", " ")
            tokens_str = tokens_str.replace("(", " ").replace(")", " ")
            tokens_str = tokens_str.replace(",", " ")
            tokens = tokens_str.split()

            if not tokens:
                return None

            kinds = set()
            for tok in tokens:
                low = tok.strip().lower()
                if low in ("nan", "nan.", "float('nan')"):
                    kinds.add("nan")
                elif low in (
                    "inf",
                    "+inf",
                    "-inf",
                    "inf.",
                    "-inf.",
                    "float('inf')",
                    "float('-inf')",
                ):
                    kinds.add("inf")
                else:
                    try:
                        val = float(low)
                        if math.isnan(val):
                            kinds.add("nan")
                        elif math.isinf(val):
                            kinds.add("inf")
                        else:
                            return None  # Regular number
                    except (ValueError, OverflowError):
                        return None  # Not a recognised special value

            if len(kinds) == 1:
                return kinds.pop()
            return None

        lhs_kind = _represents_nan_or_inf(lhs)
        rhs_kind = _represents_nan_or_inf(rhs)

        if lhs_kind is not None and lhs_kind == rhs_kind:
            return True

        return False

    def _write_bug_reports(self) -> None:
        self._write_json(self.bug_reports_file, self.bug_reports_data)
        self._write_bug_reports_filtered()

    def _write_bug_reports_filtered(self) -> None:
        filtered = [
            entry
            for entry in self.bug_reports_data
            if not self._is_low_signal_report(entry)
        ]
        self._write_json(self.bug_reports_filtered_file, filtered)

    def _write_diagnostics(self) -> None:
        self._write_json(self.diagnostic_file, self.diagnostic_data)

    def log_diagnostic(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        stage: str,
        status: str,
        details: Dict[str, Any],
        test_code: Optional[str] = None,
    ) -> None:
        with self._lock:
            diagnostic_entry = {
                "timestamp": datetime.now().isoformat(),
                "optimization": optimization_name,
                "iteration": iteration,
                "sample_idx": sample_idx,
                "stage": stage,
                "status": status,
                "details": details,
            }

            self.diagnostic_data.append(diagnostic_entry)
            self._write_diagnostics()

    @staticmethod
    def _unescape_text(text: str) -> str:
        if isinstance(text, str):
            return text.replace("\\n", "\n").replace("\\t", "\t")
        return text

    @staticmethod
    def _write_header(f, title: str) -> None:
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")

    @staticmethod
    def _write_section_header(f, opt_name: str) -> None:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"OPTIMIZATION: {opt_name}\n")
        f.write(f"{'=' * 80}\n\n")

    def _write_prompts_readable(self) -> None:
        with open(self.prompts_text_file, "w") as f:
            self._write_header(f, "WHITEFOX PROMPTS - READABLE FORMAT")

            for opt_name, prompts_list in self.prompts_data.items():
                self._write_section_header(f, opt_name)

                for prompt_entry in prompts_list:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Iteration: {prompt_entry['iteration']} | ")
                    f.write(f"Type: {prompt_entry['prompt_type']} | ")
                    f.write(f"Timestamp: {prompt_entry['timestamp']}\n")
                    f.write(f"Number of Examples: {prompt_entry['num_examples']}\n")
                    f.write(f"{'-' * 80}\n\n")

                    f.write("PROMPT TEXT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(self._unescape_text(prompt_entry["prompt"]))
                    f.write("\n" + "-" * 40 + "\n")

                    if prompt_entry.get("examples"):
                        f.write("\nEXAMPLES USED:\n")
                        for i, example in enumerate(prompt_entry["examples"], 1):
                            f.write(f"  {i}. Test ID: {example['test_id']}\n")
                            f.write(f"     File: {example['file_path']}\n")
                            f.write(
                                f"     Alpha: {example['alpha']}, Beta: {example['beta']}\n"
                            )
                    f.write("\n")

    def _write_code_readable(self) -> None:
        with open(self.cleaned_code_text_file, "w") as f:
            self._write_header(f, "WHITEFOX CLEANED CODE - READABLE FORMAT")

            for opt_name, code_list in self.cleaned_code_data.items():
                self._write_section_header(f, opt_name)

                for code_entry in code_list:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Iteration: {code_entry['iteration']} | ")
                    f.write(f"Sample: {code_entry['sample_idx']}\n")
                    f.write(f"{'-' * 80}\n\n")

                    f.write("CLEANED CODE:\n")
                    f.write("-" * 40 + "\n")
                    f.write(self._unescape_text(code_entry["code"]))
                    f.write("\n" + "-" * 40 + "\n\n")

    def _track_execution_stats(
        self, optimization_name: str, result: Any, pass_triggered: bool
    ) -> None:
        with self._lock:
            self._ensure_stats_initialized(optimization_name)
            stats = self.opt_stats[optimization_name]

            if pass_triggered:
                stats["triggered"] += 1

            for mode in result.modes:
                key = f"success_{mode}"
                if key not in stats:
                    stats[key] = 0
                mr = result.get_mode(mode)
                if mr.runtime_success:
                    stats[key] += 1

    def generate_run_summary(self, whitefox_state: Any) -> None:
        with self._lock:
            detailed_summary_file = self.log_dir / "run_summary_detailed.log"

            # Collect all mode keys across all optimizations
            mode_keys = []
            for stats in self.opt_stats.values():
                for key in stats:
                    if key.startswith("success_") and key not in mode_keys:
                        mode_keys.append(key)
            mode_keys.sort()

            with open(detailed_summary_file, "w") as f:
                self._write_header(f, "WHITEFOX DETAILED RUN SUMMARY")

                mode_labels = [k.replace("success_", "") for k in mode_keys]
                header = (
                    "Optimization                             | Created | Triggered"
                )
                for label in mode_labels:
                    header += f" | {label.capitalize()}"
                f.write(header + "\n")
                f.write("-" * max(80, len(header) + 5) + "\n\n")

                totals = {"generated": 0, "triggered": 0}
                for mk in mode_keys:
                    totals[mk] = 0

                for opt_name in sorted(whitefox_state.optimizations.keys()):
                    stats = self.opt_stats.get(opt_name, self._get_default_stats())

                    totals["generated"] += stats.get("generated", 0)
                    totals["triggered"] += stats.get("triggered", 0)

                    f.write(f"{opt_name:40s} | ")
                    f.write(f"{stats.get('generated', 0):7d} | ")
                    f.write(f"{stats.get('triggered', 0):9d}")

                    for mk in mode_keys:
                        val = stats.get(mk, 0)
                        totals[mk] += val
                        f.write(f" | {val:7d}")
                    f.write("\n")

                f.write("-" * max(80, len(header) + 5) + "\n")
                f.write(f"{'TOTAL':40s} | ")
                f.write(f"{totals['generated']:7d} | ")
                f.write(f"{totals['triggered']:9d}")
                for mk in mode_keys:
                    f.write(f" | {totals[mk]:7d}")
                f.write("\n")
                f.write("=" * max(80, len(header) + 5) + "\n")

            if self.base_logger:
                self.base_logger.debug(
                    f"Run summary updated at {detailed_summary_file}"
                )

    def flush(self) -> None:
        with self._lock:
            self._write_prompts()
            self._write_cleaned_code()
            self._write_bug_reports()
            self._write_diagnostics()
