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
    test_file: Path
    compile_success_naive: bool
    compile_success_xla: bool
    compile_success_autocluster: bool
    runtime_success_naive: bool
    runtime_success_xla: bool
    runtime_success_autocluster: bool
    compile_error_naive: Optional[str] = None
    compile_error_xla: Optional[str] = None
    compile_error_autocluster: Optional[str] = None
    runtime_error_naive: Optional[str] = None
    runtime_error_xla: Optional[str] = None
    runtime_error_autocluster: Optional[str] = None
    output_naive: Optional[Any] = None
    output_xla: Optional[Any] = None
    output_autocluster: Optional[Any] = None
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
    test_id: str
    optimizations_triggered: List[str]
    oracle_type: str
    details: Dict[str, Any]
    test_file: Path
    logs_file: Path
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "optimizations_triggered": self.optimizations_triggered,
            "oracle_type": self.oracle_type,
            "details": self.details,
            "test_file": str(self.test_file),
            "logs_file": str(self.logs_file),
        }
    
    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

