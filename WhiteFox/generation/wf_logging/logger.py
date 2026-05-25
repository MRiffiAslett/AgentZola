import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from domain.bandit import OptimizationState, TriggeringTest
from generation.wf_logging.quality import (
    QUALITY_TABLE_ROWS,
    accumulate_quality_stats,
    classify_generation_quality,
    default_quality_stats,
)


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

        self.prompts_file = self.log_dir / "prompts.json"
        self.cleaned_code_file = self.log_dir / "all_cleaned_code.json"
        self.bug_reports_file = self.log_dir / "bug_reports.json"
        self.execution_results_file = self.log_dir / "execution_results.jsonl"
        self.generation_quality_file = self.log_dir / "generation_quality.log"

        self.prompts_data: Dict[str, List[Dict]] = {}
        self.cleaned_code_data: Dict[str, List[Dict]] = {}
        self.bug_reports_data: List[Dict] = []
        self.execution_results_data: List[Dict] = []

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
            self.execution_results_file,
            self.generation_quality_file,
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
            self.execution_results_data,
            self.opt_stats,
        ]:
            data.clear()

    @staticmethod
    def _get_default_stats() -> Dict[str, int]:
        return {
            "generated": 0,
            "triggered": 0,
            "valid": 0,
            "invalid": 0,
        }

    def _ensure_stats_initialized(self, opt_key: str) -> None:
        if opt_key not in self.opt_stats:
            stats = self._get_default_stats()
            stats.update(default_quality_stats())
            self.opt_stats[opt_key] = stats

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
                    "raw_llm_output": raw_text,
                    "cleaned_code": cleaned_code,
                    "parsed_code": parsed_code,
                }
            )

            self._ensure_stats_initialized(optimization_name)
            self.opt_stats[optimization_name]["generated"] += 1

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
        test_code = None
        try:
            test_code = test_file.read_text()
        except Exception:
            pass

        quality = classify_generation_quality(test_code, result=result)
        self._record_generation_quality(
            optimization_name,
            iteration,
            sample_idx,
            test_file,
            quality,
            result=result,
            pass_triggered=pass_triggered,
        )
        self._track_execution_stats(optimization_name, result, pass_triggered)

    def log_execution_failure(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        test_file: Path,
        worker_error: str,
    ) -> None:
        test_code = None
        try:
            test_code = test_file.read_text()
        except Exception:
            pass

        quality = classify_generation_quality(
            test_code, result=None, worker_error=worker_error
        )
        self._record_generation_quality(
            optimization_name,
            iteration,
            sample_idx,
            test_file,
            quality,
            worker_error=worker_error,
        )

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

    @staticmethod
    def _append_jsonl(file_path: Path, data: Any, **kwargs) -> None:
        """Append each top-level entry as a single JSON line."""
        with open(file_path, "a") as f:
            if isinstance(data, dict):
                for key, entries in data.items():
                    if isinstance(entries, list):
                        for entry in entries:
                            json.dump(entry, f, **kwargs)
                            f.write("\n")
                    else:
                        json.dump({key: entries}, f, **kwargs)
                        f.write("\n")
            elif isinstance(data, list):
                for entry in data:
                    json.dump(entry, f, **kwargs)
                    f.write("\n")
            else:
                json.dump(data, f, **kwargs)
                f.write("\n")

    def _write_prompts(self) -> None:
        self._append_jsonl(self.prompts_file, self.prompts_data)

    def _write_cleaned_code(self) -> None:
        self._append_jsonl(
            self.cleaned_code_file, self.cleaned_code_data, ensure_ascii=False
        )

    def _write_bug_reports(self) -> None:
        self._append_jsonl(self.bug_reports_file, self.bug_reports_data)

    def _write_execution_results(self) -> None:
        self._append_jsonl(self.execution_results_file, self.execution_results_data)

    def _record_generation_quality(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        test_file: Path,
        quality: Dict[str, Any],
        result: Any = None,
        pass_triggered: Optional[bool] = None,
        worker_error: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "test_file": str(test_file),
            "pass_triggered": pass_triggered,
            "quality": quality,
        }

        if worker_error:
            entry["worker_error"] = worker_error[:2000]

        if result is not None:
            entry["modes"] = {}
            for mode in result.modes:
                mr = result.get_mode(mode)
                entry["modes"][mode] = {
                    "compile_success": mr.compile_success,
                    "runtime_success": mr.runtime_success,
                }

        with self._lock:
            self.execution_results_data.append(entry)
            self._ensure_stats_initialized(optimization_name)
            accumulate_quality_stats(self.opt_stats[optimization_name], quality)

    @staticmethod
    def _pct(count: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{100.0 * count / total:.1f}%"

    def _write_quality_table(
        self,
        f,
        opt_names: List[str],
        *,
        title: str,
    ) -> None:
        self._write_header(f, title)
        f.write(
            f"{'Generated test category':40s} | {'Count':>7s} | {'%':>7s}\n"
        )
        f.write("-" * 60 + "\n\n")

        totals = default_quality_stats()
        generated_total = 0
        for name in opt_names:
            stats = self.opt_stats.get(name, {})
            generated_total += stats.get("generated", 0)
            for key in totals:
                totals[key] += stats.get(key, 0)

        for key, label in QUALITY_TABLE_ROWS:
            count = totals.get(key, 0)
            f.write(
                f"{label:40s} | {count:7d} | "
                f"{self._pct(count, generated_total):>7s}\n"
            )

        f.write("\n")
        f.write(f"Generated (denominator): {generated_total}\n")
        f.write(f"Executed:                {totals.get('executed', 0)}\n")
        f.write(f"Worker failures:         {totals.get('worker_failed', 0)}\n")
        f.write("=" * 60 + "\n")

        f.write("\nPer optimization (% of generated tests):\n\n")
        f.write(
            f"{'Optimization':40s} | {'Gen':>5s} | "
            f"{'Syntax':>6s} | {'Import':>6s} | {'Eager':>6s} | "
            f"{'XLA':>6s} | {'BadAPI':>6s} | {'Unsup':>6s} | {'T/O':>5s}\n"
        )
        f.write("-" * 110 + "\n\n")

        for name in sorted(opt_names):
            stats = self.opt_stats.get(name, self._get_default_stats())
            generated = stats.get("generated", 0)
            f.write(f"{name:40s} | {generated:5d} | ")
            for key in (
                "syntax_valid",
                "imports_successfully",
                "eager_executable",
                "xla_compilable",
                "invalid_tf_api",
                "unsupported_by_xla",
                "timeout",
            ):
                count = stats.get(key, 0)
                if key in ("invalid_tf_api", "unsupported_by_xla", "timeout"):
                    f.write(f"{count:6d} | ")
                else:
                    f.write(f"{self._pct(count, generated):>6s} | ")
            f.write("\n")
        f.write("=" * 110 + "\n")

    def _write_generation_quality_log(self, opt_names: List[str]) -> None:
        with open(self.generation_quality_file, "w") as f:
            self._write_quality_table(
                f,
                opt_names,
                title="GENERATION QUALITY DISTRIBUTION (pre-oracle)",
            )

    @staticmethod
    def _write_header(f, title: str) -> None:
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")

    def _track_execution_stats(
        self, optimization_name: str, result: Any, pass_triggered: bool
    ) -> None:
        with self._lock:
            self._ensure_stats_initialized(optimization_name)
            stats = self.opt_stats[optimization_name]

            if pass_triggered:
                stats["triggered"] += 1

            any_success = False
            for mode in result.modes:
                key = f"success_{mode}"
                if key not in stats:
                    stats[key] = 0
                mr = result.get_mode(mode)
                if mr.runtime_success:
                    stats[key] += 1
                    any_success = True

            if any_success:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1

    def generate_run_summary(
        self,
        opt_names: List[str],
        opt_states: Optional[Dict[str, Any]] = None,
        coverage_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            detailed_summary_file = self.log_dir / "run_summary_detailed.log"

            mode_keys: List[str] = []
            for stats in self.opt_stats.values():
                for key in stats:
                    if key.startswith("success_") and key not in mode_keys:
                        mode_keys.append(key)
            mode_keys.sort()

            with open(detailed_summary_file, "w") as f:
                self._write_header(f, "WHITEFOX DETAILED RUN SUMMARY")

                # --- Test generation table ---
                mode_labels = [k.replace("success_", "") for k in mode_keys]
                header = (
                    "Optimization                             "
                    "| Created |   Valid | Invalid | Triggered"
                )
                for label in mode_labels:
                    header += f" | {label.capitalize():>11s}"
                f.write(header + "\n")
                sep_len = max(95, len(header) + 5)
                f.write("-" * sep_len + "\n\n")

                totals = {
                    "generated": 0, "triggered": 0,
                    "valid": 0, "invalid": 0,
                }
                for mk in mode_keys:
                    totals[mk] = 0

                for name in sorted(opt_names):
                    stats = self.opt_stats.get(name, self._get_default_stats())

                    totals["generated"] += stats.get("generated", 0)
                    totals["triggered"] += stats.get("triggered", 0)
                    totals["valid"] += stats.get("valid", 0)
                    totals["invalid"] += stats.get("invalid", 0)

                    f.write(f"{name:40s} | ")
                    f.write(f"{stats.get('generated', 0):7d} | ")
                    f.write(f"{stats.get('valid', 0):7d} | ")
                    f.write(f"{stats.get('invalid', 0):7d} | ")
                    f.write(f"{stats.get('triggered', 0):9d}")
                    for mk in mode_keys:
                        val = stats.get(mk, 0)
                        totals[mk] += val
                        f.write(f" | {val:11d}")
                    f.write("\n")

                f.write("-" * sep_len + "\n")
                f.write(f"{'TOTAL':40s} | ")
                f.write(f"{totals['generated']:7d} | ")
                f.write(f"{totals['valid']:7d} | ")
                f.write(f"{totals['invalid']:7d} | ")
                f.write(f"{totals['triggered']:9d}")
                for mk in mode_keys:
                    f.write(f" | {totals[mk]:11d}")
                f.write("\n")
                f.write("=" * sep_len + "\n")

                f.write("\n")
                self._write_quality_table(
                    f,
                    opt_names,
                    title="GENERATION QUALITY DISTRIBUTION (pre-oracle)",
                )

                # --- Thompson sampling stats ---
                if opt_states:
                    f.write("\n")
                    self._write_header(f, "THOMPSON SAMPLING STATS")
                    f.write(
                        f"{'Optimization':40s} | {'Tests':>5s} | "
                        f"{'Avg Alpha':>9s} | {'Avg Beta':>9s} | "
                        f"{'Avg Theta':>9s}\n"
                    )
                    f.write("-" * 85 + "\n\n")

                    for name in sorted(opt_names):
                        os_ = opt_states.get(name)
                        if os_ is None or not os_.triggering_tests:
                            f.write(f"{name:40s} |     0 |       - |       - |       -\n")
                            continue
                        tests = list(os_.triggering_tests.values())
                        n = len(tests)
                        avg_a = sum(t.alpha for t in tests) / n
                        avg_b = sum(t.beta for t in tests) / n
                        avg_theta = avg_a / (avg_a + avg_b) if (avg_a + avg_b) > 0 else 0
                        f.write(
                            f"{name:40s} | {n:5d} | "
                            f"{avg_a:9.2f} | {avg_b:9.2f} | "
                            f"{avg_theta:9.4f}\n"
                        )
                    f.write("=" * 85 + "\n")

                # --- Coverage ---
                if coverage_data:
                    f.write("\n")
                    self._write_header(f, "COVERAGE")
                    xla_hit = coverage_data.get("xla_lines_hit", 0)
                    xla_total = coverage_data.get("xla_lines_total", 0)
                    xla_pct = (xla_hit / xla_total * 100) if xla_total else 0
                    all_hit = coverage_data.get("all_lines_hit", 0)
                    all_total = coverage_data.get("all_lines_total", 0)
                    f.write(f"XLA lines hit:    {xla_hit:>10,}\n")
                    f.write(f"XLA lines total:  {xla_total:>10,}\n")
                    f.write(f"XLA coverage:     {xla_pct:>9.2f}%\n")
                    f.write(f"All TF lines hit: {all_hit:>10,}\n")
                    f.write(f"All TF lines total:{all_total:>9,}\n")
                    f.write("=" * 40 + "\n")

            self._write_generation_quality_log(opt_names)

            if self.base_logger:
                self.base_logger.debug(
                    f"Run summary updated at {detailed_summary_file}"
                )
                self.base_logger.debug(
                    f"Generation quality log updated at {self.generation_quality_file}"
                )

    def flush(self) -> None:
        with self._lock:
            self._write_prompts()
            self._write_cleaned_code()
            self._write_bug_reports()
            self._write_execution_results()

    def flush_and_clear(self) -> None:
        """Flush all buffered data to disk and release the in-memory copies.

        Call between optimizations to prevent unbounded memory growth.
        ``opt_stats`` is preserved since the run summary needs it.
        """
        with self._lock:
            self._write_prompts()
            self._write_cleaned_code()
            self._write_bug_reports()
            self._write_execution_results()
            self.prompts_data.clear()
            self.cleaned_code_data.clear()
            self.bug_reports_data.clear()
            self.execution_results_data.clear()
