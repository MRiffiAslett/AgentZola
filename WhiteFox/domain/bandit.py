"""
Bandit data classes for Thompson Sampling state management.

Contains the data structures for managing bandit state for each optimization,
including triggering tests and their Beta distribution parameters (alpha, beta).
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..generation.prompts import OptimizationSpec


@dataclass
class TriggeringTest:
    test_id: str
    optimization_name: str
    file_path: Path
    alpha: float
    beta: float
    total_generated_from: int
    total_triggers_from: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "optimization_name": self.optimization_name,
            "file_path": str(self.file_path),
            "alpha": self.alpha,
            "beta": self.beta,
            "total_generated_from": self.total_generated_from,
            "total_triggers_from": self.total_triggers_from,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TriggeringTest":
        return cls(
            test_id=data["test_id"],
            optimization_name=data["optimization_name"],
            file_path=Path(data["file_path"]),
            alpha=data["alpha"],
            beta=data["beta"],
            total_generated_from=data["total_generated_from"],
            total_triggers_from=data["total_triggers_from"],
        )


@dataclass
class OptimizationState:
    spec: "OptimizationSpec"
    triggering_tests: Dict[str, TriggeringTest] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "internal_name": self.spec.internal_name,
            "pass_log_name": self.spec.pass_log_name,
            "triggering_tests": {
                test_id: test.to_dict()
                for test_id, test in self.triggering_tests.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict, spec: "OptimizationSpec") -> "OptimizationState":
        state = cls(spec=spec)
        for test_id, test_data in data.get("triggering_tests", {}).items():
            state.triggering_tests[test_id] = TriggeringTest.from_dict(test_data)
        return state


@dataclass
class WhiteFoxState:
    optimizations: Dict[str, OptimizationState]
    config_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_path": self.config_path,
            "optimizations": {
                opt_name: opt_state.to_dict()
                for opt_name, opt_state in self.optimizations.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict, specs: Dict[str, "OptimizationSpec"]) -> "WhiteFoxState":
        optimizations = {}
        for opt_name, opt_data in data.get("optimizations", {}).items():
            if opt_name in specs:
                optimizations[opt_name] = OptimizationState.from_dict(opt_data, specs[opt_name])
            else:
                continue
        
        return cls(
            optimizations=optimizations,
            config_path=data.get("config_path"),
        )
    
    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: Path, specs: Dict[str, "OptimizationSpec"]) -> "WhiteFoxState":
        if not file_path.exists():
            return cls(optimizations={})
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data, specs)

