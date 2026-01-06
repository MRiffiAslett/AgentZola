"""
Comprehensive logging module for WhiteFox generation.

Provides structured logging with consolidated files for easier reading.
"""

import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_m = sys.modules.get('logging')
if not _m or not hasattr(_m, 'Logger'):
    if _m:
        sys.modules.pop('logging')
    _temp_path = sys.path[:]
    sys.path = [p for p in sys.path if 'generation' not in p.lower() or 'AgentZola' not in p]
    _spec = importlib.util.find_spec('logging')
    logging = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(logging)
    sys.modules['logging'] = logging
    sys.path[:] = _temp_path
else:
    logging = _m

from domain.bandit import OptimizationState, TriggeringTest


class WhiteFoxLogger:
    """
    Structured logger for WhiteFox generation pipeline.
    
    Consolidates logs into single files per optimization for easier reading.
    """
    
    def __init__(self, log_dir: Path, base_logger: Optional[logging.Logger] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create source directory for JSON files
        self.source_dir = self.log_dir / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON source files
        self.prompts_file = self.source_dir / "prompts.json"
        self.code_before_after_file = self.source_dir / "cleaned_text_before_and_after.json"
        self.cleaned_code_file = self.source_dir / "all_cleaned_code.json"
        self.execution_results_file = self.source_dir / "execution_results.json"
        self.pass_analysis_file = self.source_dir / "pass_detection_analysis.json"
        self.state_changes_file = self.source_dir / "state_changes.json"
        self.errors_file = self.source_dir / "errors.json"
        self.bug_reports_file = self.source_dir / "bug_reports.json"
        self.diagnostic_file = self.source_dir / "execution_diagnostics.json"
        
        # Readable text output files (in main logging directory)
        self.prompts_text_file = self.log_dir / "prompts_readable.log"
        self.cleaned_code_text_file = self.log_dir / "cleaned_code_readable.log"
        self.code_before_after_text_file = self.log_dir / "code_before_after_readable.log"
        
        self.prompts_data: Dict[str, List[Dict]] = {}
        self.code_before_after_data: Dict[str, List[Dict]] = {}
        self.cleaned_code_data: Dict[str, List[Dict]] = {}
        self.execution_results_data: Dict[str, List[Dict]] = {}
        self.pass_analysis_data: Dict[str, List[Dict]] = {}
        self.state_changes_data: Dict[str, List[Dict]] = {}
        self.errors_data: List[Dict] = []
        self.bug_reports_data: List[Dict] = []
        self.diagnostic_data: List[Dict] = []
        
        self.base_logger = base_logger or logging.getLogger(__name__)
    
    def clear_old_logs(self) -> None:
        """Clear all consolidated log files from previous runs."""
        # JSON source files
        log_files = [
            self.prompts_file,
            self.code_before_after_file,
            self.cleaned_code_file,
            self.execution_results_file,
            self.pass_analysis_file,
            self.state_changes_file,
            self.errors_file,
            self.bug_reports_file,
            self.diagnostic_file,
        ]
        
        # Readable text files
        text_files = [
            self.prompts_text_file,
            self.cleaned_code_text_file,
            self.code_before_after_text_file,
        ]
        
        for log_file in log_files + text_files:
            if log_file.exists():
                log_file.unlink()
                if self.base_logger:
                    self.base_logger.debug(f"Cleared old log file: {log_file}")
        
        # Clear old timestamped sanity check reports (keep only the current format)
        for pattern in [
            "sanity_check_report_*.json",
            "sanity_check_report_*.txt",
            "sanity_check_report_*.log",
            "sanity_check_report_latest.json",
            "sanity_check_report_latest.txt",
            "sanity_check_report_latest.log",
        ]:
            for old_file in self.log_dir.glob(pattern):
                old_file.unlink()
                if self.base_logger:
                    self.base_logger.debug(f"Cleared old sanity check file: {old_file}")
        
        # Also clear the current format files (they'll be recreated fresh)
        current_sanity_files = [
            self.log_dir / "sanity_check_report.json",
            self.log_dir / "sanity_check_report.txt",
            self.log_dir / "sanity_check_report.log",
        ]
        for sanity_file in current_sanity_files:
            if sanity_file.exists():
                sanity_file.unlink()
                if self.base_logger:
                    self.base_logger.debug(f"Cleared sanity check file: {sanity_file}")
        
        # Clear in-memory data structures to ensure fresh start
        self.prompts_data.clear()
        self.code_before_after_data.clear()
        self.cleaned_code_data.clear()
        self.execution_results_data.clear()
        self.pass_analysis_data.clear()
        self.state_changes_data.clear()
        self.errors_data.clear()
        self.bug_reports_data.clear()
        self.diagnostic_data.clear()
    
    def _get_opt_key(self, optimization_name: str) -> str:
        """Get key for optimization-specific data."""
        return optimization_name
    
    def log_prompt(
        self,
        optimization_name: str,
        iteration: int,
        prompt_type: str,
        prompt_text: str,
        example_tests: Optional[List[TriggeringTest]] = None
    ) -> None:
        """Log the prompt sent to the LLM (consolidated)."""
        opt_key = self._get_opt_key(optimization_name)
        if opt_key not in self.prompts_data:
            self.prompts_data[opt_key] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "prompt_type": prompt_type,
            "prompt": prompt_text,
            "num_examples": len(example_tests) if example_tests else 0,
            "examples": []
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
        
        self.prompts_data[opt_key].append(log_entry)
        self._write_prompts()
    
    def log_generated_code(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        raw_text: str,
        cleaned_code: str,
        cleaning_changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log generated code (consolidated into before/after and cleaned code files)."""
        opt_key = self._get_opt_key(optimization_name)
        
        if opt_key not in self.code_before_after_data:
            self.code_before_after_data[opt_key] = []
        
        self.code_before_after_data[opt_key].append({
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "raw_text": raw_text,
            "cleaned_code": cleaned_code,
        })
        
        if opt_key not in self.cleaned_code_data:
            self.cleaned_code_data[opt_key] = []
        
        self.cleaned_code_data[opt_key].append({
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "code": cleaned_code,
        })
        
        self._write_code_logs()
    
    def log_execution_result(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        test_file: Path,
        result: Any,
        pass_triggered: bool,
        pass_log_name: str
    ) -> None:
        """Log execution result (consolidated)."""
        opt_key = self._get_opt_key(optimization_name)
        if opt_key not in self.execution_results_data:
            self.execution_results_data[opt_key] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "test_file": str(test_file),
            "pass_triggered": pass_triggered,
            "triggered_passes": list(result.triggered_passes) if hasattr(result, 'triggered_passes') else [],
            "compile_success": {
                "naive": result.compile_success_naive,
                "xla": result.compile_success_xla,
                "autocluster": result.compile_success_autocluster,
            },
            "runtime_success": {
                "naive": result.runtime_success_naive,
                "xla": result.runtime_success_xla,
                "autocluster": result.runtime_success_autocluster,
            },
        }
        
        errors = {}
        if result.compile_error_naive:
            errors["compile_naive"] = result.compile_error_naive[:200]
        if result.compile_error_xla:
            errors["compile_xla"] = result.compile_error_xla[:200]
        if result.compile_error_autocluster:
            errors["compile_autocluster"] = result.compile_error_autocluster[:200]
        if result.runtime_error_naive:
            errors["runtime_naive"] = result.runtime_error_naive[:200]
        if result.runtime_error_xla:
            errors["runtime_xla"] = result.runtime_error_xla[:200]
        if result.runtime_error_autocluster:
            errors["runtime_autocluster"] = result.runtime_error_autocluster[:200]
        
        if errors:
            log_entry["errors"] = errors
        
        self.execution_results_data[opt_key].append(log_entry)
        self._write_execution_results()
    
    def log_pass_detection_analysis(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        pass_log_name: str,
        log_text: str,
        triggered_passes: set,
        expected_pass: str
    ) -> None:
        """Log pass detection analysis (consolidated, minimal summary)."""
        opt_key = self._get_opt_key(optimization_name)
        if opt_key not in self.pass_analysis_data:
            self.pass_analysis_data[opt_key] = []
        
        self.pass_analysis_data[opt_key].append({
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "expected_pass": expected_pass,
            "triggered": expected_pass in triggered_passes,
            "all_triggered_passes": list(triggered_passes),
        })
        
        self._write_pass_analysis()
    
    def log_state_update(
        self,
        optimization_name: str,
        iteration: int,
        before_state: OptimizationState,
        after_state: OptimizationState,
        num_triggered: int,
        num_not_triggered: int,
        new_triggering_tests: List[Path],
        example_tests_used: List[TriggeringTest]
    ) -> None:
        """Log bandit state update (consolidated, summary only)."""
        opt_key = self._get_opt_key(optimization_name)
        if opt_key not in self.state_changes_data:
            self.state_changes_data[opt_key] = []
        
        self.state_changes_data[opt_key].append({
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "tests_before": len(before_state.triggering_tests),
            "tests_after": len(after_state.triggering_tests),
            "triggered_this_iteration": num_triggered,
            "not_triggered_this_iteration": num_not_triggered,
            "new_tests_added": len(new_triggering_tests),
            "examples_used": len(example_tests_used),
        })
        
        self._write_state_changes()
    
    def log_error(
        self,
        optimization_name: str,
        iteration: Optional[int],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log errors (consolidated, only errors)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        }
        
        if exception:
            import traceback
            log_entry["traceback"] = traceback.format_exc()
            log_entry["exception_type"] = type(exception).__name__
        
        self.errors_data.append(log_entry)
        self._write_errors()
        
        self.base_logger.error(
            f"Error: {optimization_name} it{iteration} - {error_type}: {error_message}"
        )
    
    def log_bug_report(self, bug_report: Any) -> None:
        """Log bug report (consolidated)."""
        self.bug_reports_data.append({
            "timestamp": datetime.now().isoformat(),
            "test_id": bug_report.test_id,
            "optimizations_triggered": bug_report.optimizations_triggered,
            "oracle_type": bug_report.oracle_type,
            "details": bug_report.details,
            "test_file": str(bug_report.test_file),
            "logs_file": str(bug_report.logs_file),
        })
        self._write_bug_reports()
    
    def _write_prompts(self) -> None:
        """Write consolidated prompts file."""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts_data, f, indent=2)
        self._write_prompts_readable()
    
    def _write_code_logs(self) -> None:
        """Write consolidated code logs."""
        with open(self.code_before_after_file, 'w') as f:
            json.dump(self.code_before_after_data, f, indent=2, ensure_ascii=False)
        with open(self.cleaned_code_file, 'w') as f:
            json.dump(self.cleaned_code_data, f, indent=2, ensure_ascii=False)
        self._write_code_readable()
        self._write_code_before_after_readable()
    
    def _write_execution_results(self) -> None:
        """Write consolidated execution results."""
        with open(self.execution_results_file, 'w') as f:
            json.dump(self.execution_results_data, f, indent=2)
    
    def _write_pass_analysis(self) -> None:
        """Write consolidated pass analysis."""
        with open(self.pass_analysis_file, 'w') as f:
            json.dump(self.pass_analysis_data, f, indent=2)
    
    def _write_state_changes(self) -> None:
        """Write consolidated state changes."""
        with open(self.state_changes_file, 'w') as f:
            json.dump(self.state_changes_data, f, indent=2)
    
    def _write_errors(self) -> None:
        """Write consolidated errors (only errors, no empty entries)."""
        with open(self.errors_file, 'w') as f:
            json.dump(self.errors_data, f, indent=2)
    
    def _write_bug_reports(self) -> None:
        """Write consolidated bug reports."""
        with open(self.bug_reports_file, 'w') as f:
            json.dump(self.bug_reports_data, f, indent=2)
    
    def _write_diagnostics(self) -> None:
        """Write diagnostic logs."""
        with open(self.diagnostic_file, 'w') as f:
            json.dump(self.diagnostic_data, f, indent=2)
    
    def log_diagnostic(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        stage: str,
        status: str,
        details: Dict[str, Any],
        test_code: Optional[str] = None
    ) -> None:
        """Log diagnostic information for execution tracking."""
        opt_key = self._get_opt_key(optimization_name)
        
        diagnostic_entry = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "stage": stage,  # e.g., "code_read", "exec_initial", "xla_exec", "pass_detection"
            "status": status,  # e.g., "success", "failure", "skipped"
            "details": details,
        }
        
        if test_code:
            diagnostic_entry["test_code"] = test_code[:1000]  # First 1000 chars
        
        self.diagnostic_data.append(diagnostic_entry)
        self._write_diagnostics()
    
    def _write_prompts_readable(self) -> None:
        """Write prompts in a human-readable format with actual newlines."""
        with open(self.prompts_text_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WHITEFOX PROMPTS - READABLE FORMAT\n")
            f.write("=" * 80 + "\n\n")
            
            for opt_name, prompts_list in self.prompts_data.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"OPTIMIZATION: {opt_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                for prompt_entry in prompts_list:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Iteration: {prompt_entry['iteration']} | ")
                    f.write(f"Type: {prompt_entry['prompt_type']} | ")
                    f.write(f"Timestamp: {prompt_entry['timestamp']}\n")
                    f.write(f"Number of Examples: {prompt_entry['num_examples']}\n")
                    f.write(f"{'-' * 80}\n\n")
                    
                    f.write("PROMPT TEXT:\n")
                    f.write("-" * 40 + "\n")
                    # Convert escaped newlines to actual newlines
                    prompt_text = prompt_entry['prompt']
                    if isinstance(prompt_text, str):
                        # Handle JSON-escaped newlines
                        prompt_text = prompt_text.replace('\\n', '\n').replace('\\t', '\t')
                    f.write(prompt_text)
                    f.write("\n" + "-" * 40 + "\n")
                    
                    if prompt_entry.get('examples'):
                        f.write("\nEXAMPLES USED:\n")
                        for i, example in enumerate(prompt_entry['examples'], 1):
                            f.write(f"  {i}. Test ID: {example['test_id']}\n")
                            f.write(f"     File: {example['file_path']}\n")
                            f.write(f"     Alpha: {example['alpha']}, Beta: {example['beta']}\n")
                    f.write("\n")
    
    def _write_code_readable(self) -> None:
        """Write cleaned code in a human-readable format with actual newlines."""
        with open(self.cleaned_code_text_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WHITEFOX CLEANED CODE - READABLE FORMAT\n")
            f.write("=" * 80 + "\n\n")
            
            for opt_name, code_list in self.cleaned_code_data.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"OPTIMIZATION: {opt_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                for code_entry in code_list:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Iteration: {code_entry['iteration']} | ")
                    f.write(f"Sample: {code_entry['sample_idx']}\n")
                    f.write(f"{'-' * 80}\n\n")
                    
                    f.write("CLEANED CODE:\n")
                    f.write("-" * 40 + "\n")
                    # Convert escaped newlines to actual newlines
                    code_text = code_entry['code']
                    if isinstance(code_text, str):
                        code_text = code_text.replace('\\n', '\n').replace('\\t', '\t')
                    f.write(code_text)
                    f.write("\n" + "-" * 40 + "\n\n")
    
    def _write_code_before_after_readable(self) -> None:
        """Write before/after code comparison in a human-readable format."""
        with open(self.code_before_after_text_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WHITEFOX CODE BEFORE/AFTER CLEANING - READABLE FORMAT\n")
            f.write("=" * 80 + "\n\n")
            
            for opt_name, code_list in self.code_before_after_data.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"OPTIMIZATION: {opt_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                for code_entry in code_list:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Iteration: {code_entry['iteration']} | ")
                    f.write(f"Sample: {code_entry['sample_idx']} | ")
                    f.write(f"Timestamp: {code_entry['timestamp']}\n")
                    f.write(f"{'-' * 80}\n\n")
                    
                    f.write("RAW TEXT (from LLM):\n")
                    f.write("-" * 40 + "\n")
                    raw_text = code_entry['raw_text']
                    if isinstance(raw_text, str):
                        raw_text = raw_text.replace('\\n', '\n').replace('\\t', '\t')
                    f.write(raw_text)
                    f.write("\n" + "-" * 40 + "\n\n")
                    
                    f.write("CLEANED CODE:\n")
                    f.write("-" * 40 + "\n")
                    cleaned_code = code_entry['cleaned_code']
                    if isinstance(cleaned_code, str):
                        cleaned_code = cleaned_code.replace('\\n', '\n').replace('\\t', '\t')
                    f.write(cleaned_code)
                    f.write("\n" + "-" * 40 + "\n\n")
    
    def flush(self) -> None:
        """Flush all consolidated logs to disk."""
        self._write_prompts()
        self._write_code_logs()
        self._write_execution_results()
        self._write_pass_analysis()
        self._write_state_changes()
        self._write_errors()
        self._write_bug_reports()
        self._write_diagnostics()
