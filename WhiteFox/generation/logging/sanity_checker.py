"""
Sanity checker for WhiteFox implementation.

Generates concise report with key metrics and red flags.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.bandit import WhiteFoxState
from generation.code_cleaner import validate_tensorflow_apis
from generation.prompts import estimate_tokens


class SanityChecker:
    """Concise sanity checker with key metrics."""
    
    def __init__(self, logging_dir: Path, state: Optional[WhiteFoxState] = None):
        self.logging_dir = Path(logging_dir)
        self.state = state
        self.metrics = {}
        self.red_flags = []
        self.warnings = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect key metrics from consolidated log files."""
        metrics = {
            "total_optimizations": 0,
            "optimizations_with_triggering_tests": 0,
            "total_triggering_tests": 0,
            "total_prompts": 0,
            "total_vllm_runs": 0,
            "total_iterations": 0,
            "total_samples": 0,
            "total_bugs": 0,
            "total_errors": 0,
            "pass_trigger_rate": 0.0,
            "avg_bugs_per_prompt": 0.0,
        }
        
        if self.state:
            metrics["total_optimizations"] = len(self.state.optimizations)
            for opt_state in self.state.optimizations.values():
                num_tests = len(opt_state.triggering_tests)
                metrics["total_triggering_tests"] += num_tests
                if num_tests > 0:
                    metrics["optimizations_with_triggering_tests"] += 1
        
        prompts_file = self.logging_dir / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r') as f:
                    prompts_data = json.load(f)
                for opt_prompts in prompts_data.values():
                    metrics["total_prompts"] += len(opt_prompts)
                    metrics["total_vllm_runs"] += len(opt_prompts)
                    for prompt in opt_prompts:
                        metrics["total_iterations"] = max(metrics["total_iterations"], prompt.get("iteration", 0) + 1)
            except Exception as e:
                self.warnings.append(f"Could not read prompts.json: {e}")
        
        execution_file = self.logging_dir / "execution_results.json"
        if execution_file.exists():
            try:
                with open(execution_file, 'r') as f:
                    exec_data = json.load(f)
                triggered_count = 0
                total_executions = 0
                for opt_results in exec_data.values():
                    for result in opt_results:
                        total_executions += 1
                        if result.get("pass_triggered", False):
                            triggered_count += 1
                metrics["total_samples"] = total_executions
                if total_executions > 0:
                    metrics["pass_trigger_rate"] = triggered_count / total_executions
            except Exception as e:
                self.warnings.append(f"Could not read execution_results.json: {e}")
        
        bugs_file = self.logging_dir / "bug_reports.json"
        if bugs_file.exists():
            try:
                with open(bugs_file, 'r') as f:
                    bugs_data = json.load(f)
                metrics["total_bugs"] = len(bugs_data)
            except Exception as e:
                self.warnings.append(f"Could not read bug_reports.json: {e}")
        
        errors_file = self.logging_dir / "errors.json"
        if errors_file.exists():
            try:
                with open(errors_file, 'r') as f:
                    errors_data = json.load(f)
                    metrics["total_errors"] = len(errors_data)
            except Exception as e:
                self.warnings.append(f"Could not read errors.json: {e}")
        
        if metrics["total_prompts"] > 0:
            metrics["avg_bugs_per_prompt"] = metrics["total_bugs"] / metrics["total_prompts"]
        
        return metrics
    
    def check_api_validation(self) -> Dict[str, Any]:
        """Quick API validation check."""
        invalid_code = "strategy = tf.distribute.experimental.collective_all_reduce_strategy()"
        is_valid, errors = validate_tensorflow_apis(invalid_code)
        if is_valid or len(errors) == 0:
            self.red_flags.append("API validation not detecting invalid APIs")
            return {"status": "FAIL", "message": "API validation broken"}
        return {"status": "PASS", "message": "API validation working"}
    
    def check_prompt_limiting(self) -> Dict[str, Any]:
        """Quick prompt limiting check."""
        tokens = estimate_tokens("test")
        if tokens == 0:
            return {"status": "WARN", "message": "Token estimation using fallback"}
        return {"status": "PASS", "message": "Token estimation working"}
    
    def identify_red_flags(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify red flags from metrics."""
        flags = []
        
        if metrics["total_optimizations"] == 0:
            flags.append("No optimizations loaded")
        
        if metrics["optimizations_with_triggering_tests"] == 0 and metrics["total_prompts"] > 10:
            flags.append("No optimizations have triggering tests after many prompts - few-shot learning not working")
        
        if metrics["pass_trigger_rate"] == 0.0 and metrics["total_samples"] > 20:
            flags.append("Zero pass trigger rate after many samples - generation not effective")
        
        if metrics["total_errors"] > metrics["total_samples"] * 0.5:
            flags.append(f"High error rate: {metrics['total_errors']} errors for {metrics['total_samples']} samples")
        
        if metrics["total_prompts"] == 0:
            flags.append("No prompts generated - system may not be running")
        
        return flags
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run checks and generate concise report."""
        self.metrics = self.collect_metrics()
        
        api_check = self.check_api_validation()
        prompt_check = self.check_prompt_limiting()
        
        self.red_flags.extend(self.identify_red_flags(self.metrics))
        
        overall_status = "PASS"
        if self.red_flags or api_check["status"] == "FAIL":
            overall_status = "FAIL"
        elif self.warnings or prompt_check["status"] == "WARN":
            overall_status = "WARN"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "metrics": self.metrics,
            "checks": {
                "api_validation": api_check,
                "prompt_limiting": prompt_check,
            },
            "red_flags": self.red_flags,
            "warnings": self.warnings,
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save report with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.logging_dir / f"sanity_check_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        latest_file = self.logging_dir / "sanity_check_report_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def format_report_for_chat(self, report: Dict[str, Any]) -> str:
        """Format concise report for quick reading."""
        lines = []
        lines.append("=" * 80)
        lines.append("WHITEFOX SANITY CHECK REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Status: {report['overall_status']}")
        lines.append("")
        
        lines.append("KEY METRICS:")
        m = report["metrics"]
        lines.append(f"  Optimizations: {m['total_optimizations']} total, {m['optimizations_with_triggering_tests']} with triggering tests")
        lines.append(f"  Triggering Tests: {m['total_triggering_tests']}")
        lines.append(f"  Prompts Run: {m['total_prompts']}")
        lines.append(f"  vLLM Runs: {m['total_vllm_runs']}")
        lines.append(f"  Iterations: {m['total_iterations']}")
        lines.append(f"  Samples Generated: {m['total_samples']}")
        lines.append(f"  Pass Trigger Rate: {m['pass_trigger_rate']:.1%}")
        lines.append(f"  Bugs Found: {m['total_bugs']} (avg {m['avg_bugs_per_prompt']:.2f} per prompt)")
        lines.append(f"  Errors: {m['total_errors']}")
        lines.append("")
        
        lines.append("CHECKS:")
        for check_name, check_result in report["checks"].items():
            status = check_result["status"]
            symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "!"
            lines.append(f"  {symbol} {check_name}: {status} - {check_result['message']}")
        lines.append("")
        
        if report["red_flags"]:
            lines.append("🚨 RED FLAGS:")
            for flag in report["red_flags"]:
                lines.append(f"  ✗ {flag}")
            lines.append("")
        
        if report["warnings"]:
            lines.append("WARNINGS:")
            for warning in report["warnings"]:
                lines.append(f"  ! {warning}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


def run_sanity_check(logging_dir: Path, state: Optional[WhiteFoxState] = None):
    """Run sanity check and save reports."""
    checker = SanityChecker(logging_dir, state)
    report = checker.run_all_checks()
    
    json_file = checker.save_report(report)
    
    text_report = checker.format_report_for_chat(report)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = logging_dir / f"sanity_check_report_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write(text_report)
    
    latest_text_file = logging_dir / "sanity_check_report_latest.txt"
    with open(latest_text_file, 'w') as f:
        f.write(text_report)
    
    return json_file, text_file
