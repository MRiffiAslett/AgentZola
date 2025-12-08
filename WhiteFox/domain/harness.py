"""
Harness data classes for execution results and bug reports.

Contains data structures for tracking test execution results and bug reports
from the WhiteFox fuzzing system.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExecutionResult:
    """Result of executing a test program.
    
    Tracks three execution modes matching WhiteFox original:
    - naive: Standard TensorFlow execution
    - xla: XLA compilation with jit_compile=True
    - autocluster: XLA autoclustering
    """
    test_file: Path
    # Compile success flags for each mode
    compile_success_naive: bool
    compile_success_xla: bool
    compile_success_autocluster: bool
    # Runtime success flags for each mode
    runtime_success_naive: bool
    runtime_success_xla: bool
    runtime_success_autocluster: bool
    # Compile errors for each mode
    compile_error_naive: Optional[str] = None
    compile_error_xla: Optional[str] = None
    compile_error_autocluster: Optional[str] = None
    # Runtime errors for each mode
    runtime_error_naive: Optional[str] = None
    runtime_error_xla: Optional[str] = None
    runtime_error_autocluster: Optional[str] = None
    # Outputs for each mode
    output_naive: Optional[Any] = None
    output_xla: Optional[Any] = None
    output_autocluster: Optional[Any] = None
    # Allowed error flags (errors that don't count as real failures)
    allowed_error_naive: bool = False
    allowed_error_xla: bool = False
    allowed_error_autocluster: bool = False
    log_text: str = ""
    triggered_passes: set = None
    
    def __post_init__(self):
        if self.triggered_passes is None:
            self.triggered_passes = set()


@dataclass
class BugReport:
    """Bug report for a test that triggered an oracle condition."""
    test_id: str
    optimizations_triggered: List[str]
    oracle_type: str  # e.g. "miscompilation", "compile_crash_optimized", ...
    details: Dict[str, Any]  # serialized info: exception messages, shapes, etc.
    test_file: Path
    logs_file: Path
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "optimizations_triggered": self.optimizations_triggered,
            "oracle_type": self.oracle_type,
            "details": self.details,
            "test_file": str(self.test_file),
            "logs_file": str(self.logs_file),
        }
    
    def save(self, file_path: Path) -> None:
        """Save bug report to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

