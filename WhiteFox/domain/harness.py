import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModeResult:
    compile_success: bool = False
    runtime_success: bool = False
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None
    output: Optional[Any] = None
    allowed_error: bool = False


@dataclass
class ExecutionResult:
    test_file: Path
    mode_results: Dict[str, ModeResult] = field(default_factory=dict)
    log_text: str = ""
    triggered_passes: set = None

    def __post_init__(self):
        if self.triggered_passes is None:
            self.triggered_passes = set()

    def get_mode(self, mode_name: str) -> ModeResult:
        if mode_name not in self.mode_results:
            self.mode_results[mode_name] = ModeResult()
        return self.mode_results[mode_name]

    @property
    def modes(self) -> List[str]:
        return list(self.mode_results.keys())

    @property
    def compile_success_naive(self) -> bool:
        return self.get_mode("naive").compile_success

    @compile_success_naive.setter
    def compile_success_naive(self, value: bool):
        self.get_mode("naive").compile_success = value

    @property
    def compile_success_xla(self) -> bool:
        return self.get_mode("xla").compile_success

    @compile_success_xla.setter
    def compile_success_xla(self, value: bool):
        self.get_mode("xla").compile_success = value

    @property
    def compile_success_autocluster(self) -> bool:
        return self.get_mode("autocluster").compile_success

    @compile_success_autocluster.setter
    def compile_success_autocluster(self, value: bool):
        self.get_mode("autocluster").compile_success = value

    @property
    def runtime_success_naive(self) -> bool:
        return self.get_mode("naive").runtime_success

    @runtime_success_naive.setter
    def runtime_success_naive(self, value: bool):
        self.get_mode("naive").runtime_success = value

    @property
    def runtime_success_xla(self) -> bool:
        return self.get_mode("xla").runtime_success

    @runtime_success_xla.setter
    def runtime_success_xla(self, value: bool):
        self.get_mode("xla").runtime_success = value

    @property
    def runtime_success_autocluster(self) -> bool:
        return self.get_mode("autocluster").runtime_success

    @runtime_success_autocluster.setter
    def runtime_success_autocluster(self, value: bool):
        self.get_mode("autocluster").runtime_success = value

    @property
    def compile_error_naive(self) -> Optional[str]:
        return self.get_mode("naive").compile_error

    @compile_error_naive.setter
    def compile_error_naive(self, value: Optional[str]):
        self.get_mode("naive").compile_error = value

    @property
    def compile_error_xla(self) -> Optional[str]:
        return self.get_mode("xla").compile_error

    @compile_error_xla.setter
    def compile_error_xla(self, value: Optional[str]):
        self.get_mode("xla").compile_error = value

    @property
    def compile_error_autocluster(self) -> Optional[str]:
        return self.get_mode("autocluster").compile_error

    @compile_error_autocluster.setter
    def compile_error_autocluster(self, value: Optional[str]):
        self.get_mode("autocluster").compile_error = value

    @property
    def runtime_error_naive(self) -> Optional[str]:
        return self.get_mode("naive").runtime_error

    @runtime_error_naive.setter
    def runtime_error_naive(self, value: Optional[str]):
        self.get_mode("naive").runtime_error = value

    @property
    def runtime_error_xla(self) -> Optional[str]:
        return self.get_mode("xla").runtime_error

    @runtime_error_xla.setter
    def runtime_error_xla(self, value: Optional[str]):
        self.get_mode("xla").runtime_error = value

    @property
    def runtime_error_autocluster(self) -> Optional[str]:
        return self.get_mode("autocluster").runtime_error

    @runtime_error_autocluster.setter
    def runtime_error_autocluster(self, value: Optional[str]):
        self.get_mode("autocluster").runtime_error = value

    @property
    def output_naive(self) -> Optional[Any]:
        return self.get_mode("naive").output

    @output_naive.setter
    def output_naive(self, value: Optional[Any]):
        self.get_mode("naive").output = value

    @property
    def output_xla(self) -> Optional[Any]:
        return self.get_mode("xla").output

    @output_xla.setter
    def output_xla(self, value: Optional[Any]):
        self.get_mode("xla").output = value

    @property
    def output_autocluster(self) -> Optional[Any]:
        return self.get_mode("autocluster").output

    @output_autocluster.setter
    def output_autocluster(self, value: Optional[Any]):
        self.get_mode("autocluster").output = value

    @property
    def allowed_error_naive(self) -> bool:
        return self.get_mode("naive").allowed_error

    @allowed_error_naive.setter
    def allowed_error_naive(self, value: bool):
        self.get_mode("naive").allowed_error = value

    @property
    def allowed_error_xla(self) -> bool:
        return self.get_mode("xla").allowed_error

    @allowed_error_xla.setter
    def allowed_error_xla(self, value: bool):
        self.get_mode("xla").allowed_error = value

    @property
    def allowed_error_autocluster(self) -> bool:
        return self.get_mode("autocluster").allowed_error

    @allowed_error_autocluster.setter
    def allowed_error_autocluster(self, value: bool):
        self.get_mode("autocluster").allowed_error = value


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
