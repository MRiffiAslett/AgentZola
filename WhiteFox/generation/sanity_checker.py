"""
Sanity checker for WhiteFox implementation.

Verifies that all components are working correctly and generates a comprehensive
report that can be easily shared for debugging.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.bandit import WhiteFoxState, OptimizationState
from generation.code_cleaner import clean_generated_code, extract_code_from_markdown
from generation.api_validator import validate_tensorflow_apis
from generation.prompts import estimate_tokens, build_feedback_prompt, build_base_prompt


class SanityChecker:
    """Comprehensive sanity checker for WhiteFox system."""
    
    def __init__(self, logging_dir: Path, state: Optional[WhiteFoxState] = None):
        self.logging_dir = Path(logging_dir)
        self.state = state
        self.checks = []
        self.errors = []
        self.warnings = []
        
    def check_code_cleaning(self) -> Dict[str, Any]:
        """Verify code cleaning logic is working correctly."""
        result = {
            "name": "Code Cleaning",
            "status": "PASS",
            "details": []
        }
        
        # Test 1: Markdown extraction
        test_code = "```python\nimport tensorflow as tf\nclass Model(tf.keras.Model):\n    pass\n```\nThis is explanatory text."
        cleaned = extract_code_from_markdown(test_code)
        if "explanatory text" in cleaned:
            result["status"] = "FAIL"
            result["details"].append("FAIL: Markdown extraction includes trailing text")
            self.errors.append("Code cleaning includes explanatory text after markdown")
        else:
            result["details"].append("PASS: Markdown extraction removes trailing text")
        
        # Test 2: Import addition
        code_no_import = "class Model(tf.keras.Model):\n    def call(self, x):\n        return tf.reduce_sum(x)"
        cleaned = clean_generated_code(code_no_import)
        if "import tensorflow as tf" not in cleaned:
            result["status"] = "FAIL"
            result["details"].append("FAIL: Auto-import not working")
            self.errors.append("Code cleaning not adding missing imports")
        else:
            result["details"].append("PASS: Auto-import working correctly")
        
        # Test 3: Code with imports already present
        code_with_import = "import tensorflow as tf\nclass Model(tf.keras.Model):\n    pass"
        cleaned = clean_generated_code(code_with_import)
        if cleaned.count("import tensorflow as tf") > 1:
            result["status"] = "FAIL"
            result["details"].append("FAIL: Duplicate imports being added")
            self.errors.append("Code cleaning adding duplicate imports")
        else:
            result["details"].append("PASS: No duplicate imports")
        
        self.checks.append(result)
        return result
    
    def check_api_validation(self) -> Dict[str, Any]:
        """Verify API validation is working correctly."""
        result = {
            "name": "API Validation",
            "status": "PASS",
            "details": []
        }
        
        # Test 1: Invalid API detection
        invalid_code = "strategy = tf.distribute.experimental.collective_all_reduce_strategy()"
        is_valid, errors = validate_tensorflow_apis(invalid_code)
        if is_valid or len(errors) == 0:
            result["status"] = "FAIL"
            result["details"].append("FAIL: Invalid API not detected")
            self.errors.append("API validation not detecting invalid APIs")
        else:
            result["details"].append(f"PASS: Invalid API detected - {errors[0]}")
        
        # Test 2: Valid code passes
        valid_code = "import tensorflow as tf\nx = tf.constant([1, 2, 3])\ny = tf.reduce_sum(x)"
        is_valid, errors = validate_tensorflow_apis(valid_code)
        if not is_valid:
            result["status"] = "FAIL"
            result["details"].append("FAIL: Valid code incorrectly flagged")
            self.errors.append("API validation flagging valid code")
        else:
            result["details"].append("PASS: Valid code passes validation")
        
        # Test 3: Multiple invalid APIs
        multi_invalid = "s1 = tf.distribute.experimental.collective_all_reduce_strategy()\ns2 = tf.distribute.CollectiveCommunicator()"
        is_valid, errors = validate_tensorflow_apis(multi_invalid)
        if len(errors) < 2:
            result["status"] = "WARN"
            result["details"].append("WARN: Not all invalid APIs detected")
            self.warnings.append("API validation may miss some invalid APIs")
        else:
            result["details"].append(f"PASS: Multiple invalid APIs detected ({len(errors)} errors)")
        
        self.checks.append(result)
        return result
    
    def check_prompt_limiting(self) -> Dict[str, Any]:
        """Verify prompt length limiting is working."""
        result = {
            "name": "Prompt Length Limiting",
            "status": "PASS",
            "details": []
        }
        
        # Test token estimation
        test_text = "This is a test string with some content."
        tokens = estimate_tokens(test_text)
        if tokens == 0:
            result["status"] = "WARN"
            result["details"].append("WARN: Token estimation may not be working (tiktoken not installed?)")
            self.warnings.append("Token estimation using fallback method")
        else:
            result["details"].append(f"PASS: Token estimation working (estimated {tokens} tokens)")
        
        # Check if build_feedback_prompt respects limits
        # This would require actual spec and tests, so we'll check the function exists
        try:
            from generation.spec import OptimizationSpec
            # Just verify the function signature is correct
            import inspect
            sig = inspect.signature(build_feedback_prompt)
            if 'max_model_len' in sig.parameters:
                result["details"].append("PASS: build_feedback_prompt accepts max_model_len parameter")
            else:
                result["status"] = "FAIL"
                result["details"].append("FAIL: build_feedback_prompt missing max_model_len parameter")
                self.errors.append("Prompt limiting not properly integrated")
        except Exception as e:
            result["status"] = "WARN"
            result["details"].append(f"WARN: Could not verify prompt limiting integration: {e}")
        
        self.checks.append(result)
        return result
    
    def check_state_management(self) -> Dict[str, Any]:
        """Verify state management is working correctly."""
        result = {
            "name": "State Management",
            "status": "PASS",
            "details": []
        }
        
        if not self.state:
            result["status"] = "WARN"
            result["details"].append("WARN: No state provided for checking")
            return result
        
        # Check state structure
        if not hasattr(self.state, 'optimizations'):
            result["status"] = "FAIL"
            result["details"].append("FAIL: State missing optimizations")
            self.errors.append("State structure incorrect")
        else:
            result["details"].append(f"PASS: State has {len(self.state.optimizations)} optimizations")
        
        # Check triggering tests structure
        total_triggering_tests = 0
        optimizations_with_tests = 0
        for opt_name, opt_state in self.state.optimizations.items():
            if hasattr(opt_state, 'triggering_tests'):
                num_tests = len(opt_state.triggering_tests)
                total_triggering_tests += num_tests
                if num_tests > 0:
                    optimizations_with_tests += 1
                    # Check test structure
                    for test_id, test in opt_state.triggering_tests.items():
                        if not hasattr(test, 'alpha') or not hasattr(test, 'beta'):
                            result["status"] = "FAIL"
                            result["details"].append(f"FAIL: Triggering test {test_id} missing alpha/beta")
                            self.errors.append("Triggering test structure incomplete")
        
        result["details"].append(f"PASS: {optimizations_with_tests} optimizations have triggering tests")
        result["details"].append(f"INFO: Total triggering tests: {total_triggering_tests}")
        
        if total_triggering_tests == 0:
            result["status"] = "WARN"
            result["details"].append("WARN: No triggering tests found - system may not be learning")
            self.warnings.append("No triggering tests accumulated - few-shot learning not working")
        
        self.checks.append(result)
        return result
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Verify logging file structure is correct."""
        result = {
            "name": "File Structure",
            "status": "PASS",
            "details": []
        }
        
        required_dirs = [
            "prompts",
            "generated_code",
            "execution_results",
            "state_changes",
            "errors",
        ]
        
        for dir_name in required_dirs:
            dir_path = self.logging_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                result["details"].append(f"PASS: {dir_name}/ directory exists")
            else:
                result["status"] = "FAIL"
                result["details"].append(f"FAIL: {dir_name}/ directory missing")
                self.errors.append(f"Missing logging directory: {dir_name}")
        
        # Check state file location
        state_file = self.logging_dir / "whitefox_state.json"
        if state_file.exists():
            result["details"].append("PASS: whitefox_state.json in logging directory")
        else:
            result["status"] = "WARN"
            result["details"].append("WARN: whitefox_state.json not found in logging directory")
            self.warnings.append("State file may be in old location")
        
        # Check generated outputs
        generated_outputs = self.logging_dir / "generated-outputs"
        if generated_outputs.exists():
            result["details"].append("PASS: generated-outputs/ in logging directory")
        else:
            result["status"] = "WARN"
            result["details"].append("WARN: generated-outputs/ not found in logging directory")
        
        self.checks.append(result)
        return result
    
    def check_recent_activity(self) -> Dict[str, Any]:
        """Check recent activity to verify system is running."""
        result = {
            "name": "Recent Activity",
            "status": "PASS",
            "details": []
        }
        
        # Check for recent log files
        import time
        now = time.time()
        recent_threshold = 3600  # 1 hour
        
        recent_files = {
            "prompts": 0,
            "generated_code": 0,
            "execution_results": 0,
        }
        
        for log_type in recent_files.keys():
            log_dir = self.logging_dir / log_type
            if log_dir.exists():
                for log_file in log_dir.glob("*.json"):
                    if (now - log_file.stat().st_mtime) < recent_threshold:
                        recent_files[log_type] += 1
        
        total_recent = sum(recent_files.values())
        if total_recent > 0:
            result["details"].append(f"PASS: {total_recent} recent log files found")
            for log_type, count in recent_files.items():
                if count > 0:
                    result["details"].append(f"  - {log_type}: {count} files")
        else:
            result["status"] = "WARN"
            result["details"].append("WARN: No recent activity detected")
            self.warnings.append("No recent log files - system may not be running")
        
        self.checks.append(result)
        return result
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check quality of generated code samples."""
        result = {
            "name": "Generated Code Quality",
            "status": "PASS",
            "details": []
        }
        
        code_dir = self.logging_dir / "generated_code"
        if not code_dir.exists():
            result["status"] = "WARN"
            result["details"].append("WARN: No generated_code directory to check")
            return result
        
        # Sample recent generated code files
        code_files = list(code_dir.glob("*.json"))[:10]  # Check up to 10 files
        
        if len(code_files) == 0:
            result["status"] = "WARN"
            result["details"].append("WARN: No generated code files found")
            return result
        
        issues_found = {
            "markdown_in_cleaned": 0,
            "missing_imports": 0,
            "invalid_apis": 0,
            "trailing_text": 0,
        }
        
        for code_file in code_files:
            try:
                with open(code_file, 'r') as f:
                    data = json.load(f)
                
                raw_text = data.get("raw_text", "")
                cleaned_code = data.get("cleaned_code", "")
                
                # Check for markdown in cleaned code
                if "```" in cleaned_code:
                    issues_found["markdown_in_cleaned"] += 1
                
                # Check for trailing explanatory text
                if len(cleaned_code) > len(raw_text) * 1.5:  # Suspiciously long
                    issues_found["trailing_text"] += 1
                
                # Check for missing imports
                if "tf." in cleaned_code and "import tensorflow" not in cleaned_code:
                    issues_found["missing_imports"] += 1
                
                # Check for invalid APIs
                is_valid, _ = validate_tensorflow_apis(cleaned_code)
                if not is_valid:
                    issues_found["invalid_apis"] += 1
                    
            except Exception as e:
                result["details"].append(f"WARN: Error checking {code_file.name}: {e}")
        
        if sum(issues_found.values()) > 0:
            result["status"] = "WARN"
            result["details"].append(f"WARN: Found issues in {sum(issues_found.values())} files:")
            for issue, count in issues_found.items():
                if count > 0:
                    result["details"].append(f"  - {issue}: {count} files")
                    self.warnings.append(f"Code quality issue: {issue} in {count} files")
        else:
            result["details"].append(f"PASS: Checked {len(code_files)} files, no issues found")
        
        self.checks.append(result)
        return result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all sanity checks and generate report."""
        self.checks = []
        self.errors = []
        self.warnings = []
        
        # Run all checks
        self.check_code_cleaning()
        self.check_api_validation()
        self.check_prompt_limiting()
        self.check_state_management()
        self.check_file_structure()
        self.check_recent_activity()
        self.check_code_quality()
        
        # Generate summary
        total_checks = len(self.checks)
        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        failed = sum(1 for c in self.checks if c["status"] == "FAIL")
        warned = sum(1 for c in self.checks if c["status"] == "WARN")
        
        overall_status = "PASS"
        if failed > 0:
            overall_status = "FAIL"
        elif warned > 0:
            overall_status = "WARN"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": total_checks,
                "passed": passed,
                "failed": failed,
                "warnings": warned,
                "errors": len(self.errors),
                "warnings_count": len(self.warnings),
            },
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save sanity check report to file."""
        report_file = self.logging_dir / "sanity_check_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        return report_file
    
    def format_report_for_chat(self, report: Dict[str, Any]) -> str:
        """Format report as plain text for easy pasting in chat."""
        lines = []
        lines.append("=" * 80)
        lines.append("WHITEFOX SANITY CHECK REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Overall Status: {report['overall_status']}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Total Checks: {report['summary']['total_checks']}")
        lines.append(f"  Passed: {report['summary']['passed']}")
        lines.append(f"  Failed: {report['summary']['failed']}")
        lines.append(f"  Warnings: {report['summary']['warnings']}")
        lines.append(f"  Errors: {report['summary']['errors']}")
        lines.append("")
        
        # Detailed checks
        lines.append("DETAILED CHECKS:")
        for check in report['checks']:
            status_symbol = "✓" if check['status'] == "PASS" else "✗" if check['status'] == "FAIL" else "!"
            lines.append(f"  {status_symbol} {check['name']}: {check['status']}")
            for detail in check['details']:
                lines.append(f"    - {detail}")
            lines.append("")
        
        # Errors
        if report['errors']:
            lines.append("ERRORS:")
            for error in report['errors']:
                lines.append(f"  ✗ {error}")
            lines.append("")
        
        # Warnings
        if report['warnings']:
            lines.append("WARNINGS:")
            for warning in report['warnings']:
                lines.append(f"  ! {warning}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def run_sanity_check(logging_dir: Path, state: Optional[WhiteFoxState] = None) -> Path:
    """
    Run comprehensive sanity check and save report.
    
    Returns path to both JSON and text report files.
    """
    checker = SanityChecker(logging_dir, state)
    report = checker.run_all_checks()
    
    # Save JSON report
    json_file = checker.save_report(report)
    
    # Save text report for easy pasting
    text_report = checker.format_report_for_chat(report)
    text_file = logging_dir / "sanity_check_report.txt"
    with open(text_file, 'w') as f:
        f.write(text_report)
    
    return json_file, text_file

