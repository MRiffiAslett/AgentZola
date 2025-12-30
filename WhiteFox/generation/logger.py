"""
Comprehensive logging module for WhiteFox generation.

Provides structured logging to help debug:
- Code generation issues
- Few-shot example selection
- Pass triggering detection
- State management
- Error tracking
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.bandit import OptimizationState, TriggeringTest
from generation.spec import OptimizationSpec


class WhiteFoxLogger:
    """
    Structured logger for WhiteFox generation pipeline.
    
    Logs to:
    - Main log file (standard Python logging)
    - Detailed JSON logs per optimization/iteration
    - Prompt logs
    - Code generation logs
    - Execution result logs
    """
    
    def __init__(self, log_dir: Path, base_logger: Optional[logging.Logger] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different log types
        self.prompts_dir = self.log_dir / "prompts"
        self.code_dir = self.log_dir / "generated_code"
        self.execution_dir = self.log_dir / "execution_results"
        self.state_dir = self.log_dir / "state_changes"
        self.errors_dir = self.log_dir / "errors"
        
        for dir_path in [self.prompts_dir, self.code_dir, self.execution_dir, 
                        self.state_dir, self.errors_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.base_logger = base_logger or logging.getLogger(__name__)
    
    def log_prompt(
        self,
        optimization_name: str,
        iteration: int,
        prompt_type: str,  # "base" or "feedback"
        prompt_text: str,
        example_tests: Optional[List[TriggeringTest]] = None
    ) -> Path:
        """Log the prompt sent to the LLM."""
        log_data = {
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
                    "total_generated_from": test.total_generated_from,
                    "total_triggers_from": test.total_triggers_from,
                }
                try:
                    example_info["code"] = test.file_path.read_text()
                except Exception as e:
                    example_info["code_error"] = str(e)
                
                log_data["examples"].append(example_info)
        
        log_file = self.prompts_dir / f"{optimization_name}-it{iteration}-{prompt_type}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.base_logger.info(
            f"Logged prompt: {optimization_name} it{iteration} {prompt_type} "
            f"({len(example_tests) if example_tests else 0} examples)"
        )
        
        return log_file
    
    def log_generated_code(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        raw_text: str,
        cleaned_code: str,
        cleaning_changes: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Log generated code (raw and cleaned)."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "raw_text": raw_text,
            "cleaned_code": cleaned_code,
            "cleaning_changes": cleaning_changes or {},
            "raw_length": len(raw_text),
            "cleaned_length": len(cleaned_code),
        }
        
        log_file = self.code_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_file
    
    def log_execution_result(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        test_file: Path,
        result: Any,  # ExecutionResult
        pass_triggered: bool,
        pass_log_name: str
    ) -> Path:
        """Log execution result and pass triggering."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "test_file": str(test_file),
            "pass_log_name": pass_log_name,
            "pass_triggered": pass_triggered,
            "triggered_passes": list(result.triggered_passes) if hasattr(result, 'triggered_passes') else [],
            "compile_success_naive": result.compile_success_naive,
            "compile_success_xla": result.compile_success_xla,
            "compile_success_autocluster": result.compile_success_autocluster,
            "runtime_success_naive": result.runtime_success_naive,
            "runtime_success_xla": result.runtime_success_xla,
            "runtime_success_autocluster": result.runtime_success_autocluster,
            "compile_error_naive": result.compile_error_naive,
            "compile_error_xla": result.compile_error_xla,
            "compile_error_autocluster": result.compile_error_autocluster,
            "runtime_error_naive": result.runtime_error_naive,
            "runtime_error_xla": result.runtime_error_xla,
            "runtime_error_autocluster": result.runtime_error_autocluster,
            "log_text_length": len(result.log_text) if hasattr(result, 'log_text') else 0,
        }
        
        # Log snippet of errors for quick inspection
        if result.compile_error_naive:
            error_snippet = result.compile_error_naive[:500] if len(result.compile_error_naive) > 500 else result.compile_error_naive
            log_data["compile_error_snippet"] = error_snippet
        
        log_file = self.execution_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_file
    
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
    ) -> Path:
        """Log bandit state update after an iteration."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "before": {
                "num_triggering_tests": len(before_state.triggering_tests),
                "triggering_test_ids": list(before_state.triggering_tests.keys()),
            },
            "after": {
                "num_triggering_tests": len(after_state.triggering_tests),
                "triggering_test_ids": list(after_state.triggering_tests.keys()),
            },
            "iteration_stats": {
                "num_triggered": num_triggered,
                "num_not_triggered": num_not_triggered,
                "num_new_triggering_tests": len(new_triggering_tests),
                "new_test_files": [str(p) for p in new_triggering_tests],
            },
            "examples_used": [
                {
                    "test_id": t.test_id,
                    "file_path": str(t.file_path),
                    "alpha": t.alpha,
                    "beta": t.beta,
                }
                for t in example_tests_used
            ],
        }
        
        # Log alpha/beta updates for examples used
        alpha_beta_updates = []
        for example_test in example_tests_used:
            if example_test.test_id in after_state.triggering_tests:
                updated_test = after_state.triggering_tests[example_test.test_id]
                alpha_beta_updates.append({
                    "test_id": example_test.test_id,
                    "alpha_before": example_test.alpha,
                    "beta_before": example_test.beta,
                    "alpha_after": updated_test.alpha,
                    "beta_after": updated_test.beta,
                })
        log_data["alpha_beta_updates"] = alpha_beta_updates
        
        log_file = self.state_dir / f"{optimization_name}-it{iteration}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.base_logger.info(
            f"Logged state update: {optimization_name} it{iteration} - "
            f"{len(before_state.triggering_tests)} -> {len(after_state.triggering_tests)} tests, "
            f"{num_triggered} triggered, {len(new_triggering_tests)} new"
        )
        
        return log_file
    
    def log_error(
        self,
        optimization_name: str,
        iteration: Optional[int],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> Path:
        """Log errors with full context."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        }
        
        if exception:
            import traceback
            log_data["traceback"] = traceback.format_exc()
            log_data["exception_type"] = type(exception).__name__
        
        error_file = self.errors_dir / f"{optimization_name}-it{iteration or 'unknown'}-{error_type}.json"
        with open(error_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.base_logger.error(
            f"Logged error: {optimization_name} it{iteration} - {error_type}: {error_message}"
        )
        
        return error_file
    
    def log_pass_detection_analysis(
        self,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        pass_log_name: str,
        log_text: str,
        triggered_passes: set,
        expected_pass: str
    ) -> Path:
        """Log detailed analysis of pass detection."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization": optimization_name,
            "iteration": iteration,
            "sample_idx": sample_idx,
            "pass_log_name": pass_log_name,
            "expected_pass": expected_pass,
            "triggered_passes": list(triggered_passes),
            "expected_pass_triggered": expected_pass in triggered_passes,
            "log_text_length": len(log_text),
            "log_text_snippet": log_text[:2000] if len(log_text) > 2000 else log_text,
        }
        
        # Search for pass-related patterns in log
        import re
        pass_patterns = [
            r'WHITEFOX_PASS_START[^\n]*',
            r'pass=([^\s]+)',
            r'all-reduce',
            r'all-reduce-simplifier',
        ]
        
        found_patterns = []
        for pattern in pass_patterns:
            matches = re.findall(pattern, log_text, re.IGNORECASE)
            if matches:
                found_patterns.append({
                    "pattern": pattern,
                    "matches": matches[:10]  # Limit to first 10
                })
        
        log_data["found_patterns"] = found_patterns
        
        log_file = self.execution_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}-pass-analysis.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_file

