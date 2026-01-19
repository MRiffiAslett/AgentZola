"""
Comprehensive logging module for WhiteFox generation.

Provides structured logging with consolidated files for easier reading.
"""

import json
import sys
import importlib.util
import threading
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
        self.cleaned_code_file = self.source_dir / "all_cleaned_code.json"
        self.bug_reports_file = self.source_dir / "bug_reports.json"
        self.diagnostic_file = self.source_dir / "execution_diagnostics.json"
        
        # Readable text output files (in main logging directory)
        self.prompts_text_file = self.log_dir / "gen_prompts.log"
        self.cleaned_code_text_file = self.log_dir / "gen_code.log"
        self.execution_trace_file = self.log_dir / "execution_trace.log"
        
        self.prompts_data: Dict[str, List[Dict]] = {}
        self.cleaned_code_data: Dict[str, List[Dict]] = {}
        self.bug_reports_data: List[Dict] = []
        self.diagnostic_data: List[Dict] = []
        
        # Thread-safety lock for trace (only method that truly needs it for parallel execution)
        self._trace_lock = threading.Lock()
        
        # Initialize execution trace
        self._init_execution_trace()
        
        self.base_logger = base_logger or logging.getLogger(__name__)
    
    def _init_execution_trace(self) -> None:
        """Initialize execution trace log."""
        with open(self.execution_trace_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WHITEFOX EXECUTION TRACE\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("This log shows each step as it executes in real-time.\n")
            f.write("Use this to debug if WhiteFox gets stuck.\n\n")
            f.write("-" * 80 + "\n\n")
    
    def trace(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Write to execution trace log immediately (unbuffered).
        
        This log is written in real-time so you can see progress even if
        the process crashes or hangs.
        Thread-safe for parallel execution.
        """
        with self._trace_lock:
            with open(self.execution_trace_file, 'a') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                if details:
                    for key, value in details.items():
                        f.write(f"  {key}: {value}\n")
                f.flush()  # Ensure immediate write to disk
    
    def clear_old_logs(self) -> None:
        """Clear all consolidated log files from previous runs."""
        # JSON source files
        log_files = [
            self.prompts_file,
            self.cleaned_code_file,
            self.bug_reports_file,
            self.diagnostic_file,
        ]
        
        # Readable text files
        text_files = [
            self.prompts_text_file,
            self.cleaned_code_text_file,
        ]
        
        for log_file in log_files + text_files:
            if log_file.exists():
                log_file.unlink()
                if self.base_logger:
                    self.base_logger.debug(f"Cleared old log file: {log_file}")
        
        # Clear in-memory data structures to ensure fresh start
        self.prompts_data.clear()
        self.cleaned_code_data.clear()
        self.bug_reports_data.clear()
        self.diagnostic_data.clear()
        
        # Reinitialize execution trace
        self._init_execution_trace()
    
    def _get_opt_key(self, optimization_name: str) -> str:
        """Get key for optimization-specific data."""
        return optimization_name
    
    def _ensure_opt_list_exists(self, data_dict: Dict[str, List], opt_key: str) -> None:
        """Ensure an empty list exists for an optimization in a data dictionary."""
        if opt_key not in data_dict:
            data_dict[opt_key] = []
    
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
        self._ensure_opt_list_exists(self.prompts_data, opt_key)
        
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
        """Log generated code (consolidated into cleaned code file)."""
        opt_key = self._get_opt_key(optimization_name)
        self._ensure_opt_list_exists(self.cleaned_code_data, opt_key)
        
        self.cleaned_code_data[opt_key].append({
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "code": cleaned_code,
        })
        
        self._write_cleaned_code()
    
    
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
        """Log bandit state update - no-op (state changes logging removed)."""
        pass
    
    def log_error(
        self,
        optimization_name: str,
        iteration: Optional[int],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log errors - base logger only (errors.json logging removed)."""
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
    
    def _write_cleaned_code(self) -> None:
        """Write consolidated cleaned code."""
        with open(self.cleaned_code_file, 'w') as f:
            json.dump(self.cleaned_code_data, f, indent=2, ensure_ascii=False)
        self._write_code_readable()
    
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
        """Log diagnostic information for execution tracking.
        
        Only logs successes and failures (with error details).
        Written immediately so available even if process crashes.
        
        Stages: "exec_initial", "xla_exec"
        Status: "success" or "failure"
        """
        opt_key = self._get_opt_key(optimization_name)
        
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
        # Write immediately so logs are available even if process crashes
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
    
    def generate_run_summary_detailed(self, whitefox_state: Any) -> None:
        """Generate detailed run summary with optimization statistics."""
        detailed_summary_file = self.log_dir / "run_summary_detailed.log"
        
        # Write detailed summary with execution mode breakdown
        with open(detailed_summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WHITEFOX DETAILED RUN SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write("Format: Optimization | Triggering Tests\n")
            f.write("-" * 80 + "\n\n")
            
            total_triggering = 0
            
            for opt_name in sorted(whitefox_state.optimizations.keys()):
                opt_state = whitefox_state.optimizations[opt_name]
                num_triggering = len(opt_state.triggering_tests)
                total_triggering += num_triggering
                
                f.write(f"{opt_name:40s} | {num_triggering:7d}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':40s} | {total_triggering:7d}\n")
            f.write("=" * 80 + "\n")
        
        if self.base_logger:
            self.base_logger.debug(f"Detailed run summary updated at {detailed_summary_file}")
    
    def flush(self) -> None:
        """Flush all consolidated logs to disk."""
        self._write_prompts()
        self._write_cleaned_code()
        self._write_bug_reports()
        self._write_diagnostics()
