"""
Thompson Sampling bandit state management.

Manages the bandit state for each optimization, including triggering tests
and their Beta distribution parameters (alpha, beta).
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import json

import numpy as np

if TYPE_CHECKING:
    from .spec import OptimizationSpec


@dataclass
class TriggeringTest:
    """Metadata for a test that triggers an optimization."""
    test_id: str  # e.g. UUID or "AllGatherBroadcastReorder-000123"
    optimization_name: str  # internal_name
    file_path: Path  # where the .py program lives
    alpha: float  # Beta param (successes+1)
    beta: float  # Beta param (failures+1)
    total_generated_from: int  # how many child tests this has helped generate
    total_triggers_from: int  # among those, how many triggered this optimization
    
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
        """Create from dictionary."""
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
    """State for a single optimization's bandit."""
    spec: "OptimizationSpec"  # type: ignore
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
    def from_dict(cls, data: dict, spec: "OptimizationSpec") -> "OptimizationState":  # type: ignore
        """Create from dictionary."""
        state = cls(spec=spec)
        for test_id, test_data in data.get("triggering_tests", {}).items():
            state.triggering_tests[test_id] = TriggeringTest.from_dict(test_data)
        return state


@dataclass
class WhiteFoxState:
    """Global state for WhiteFox fuzzing system."""
    optimizations: Dict[str, OptimizationState]
    config_path: Optional[str] = None  # path to config file for reference
    
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
    def from_dict(cls, data: dict, specs: Dict[str, "OptimizationSpec"]) -> "WhiteFoxState":  # type: ignore
        """Create from dictionary, using specs to reconstruct OptimizationState."""
        optimizations = {}
        for opt_name, opt_data in data.get("optimizations", {}).items():
            if opt_name in specs:
                optimizations[opt_name] = OptimizationState.from_dict(opt_data, specs[opt_name])
            else:
                # Skip optimizations that no longer exist in specs
                continue
        
        return cls(
            optimizations=optimizations,
            config_path=data.get("config_path"),
        )
    
    def save(self, file_path: Path) -> None:
        """Save state to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: Path, specs: Dict[str, "OptimizationSpec"]) -> "WhiteFoxState":
        """Load state from JSON file."""
        if not file_path.exists():
            # Return empty state
            return cls(optimizations={})
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data, specs)


def select_examples_thompson_sampling(
    optimization_state: OptimizationState,
    num_examples: int
) -> List[TriggeringTest]:
    """
    Select example tests using Thompson Sampling.
    
    For each TriggeringTest, sample theta ~ Beta(alpha, beta),
    then select the top N tests by sampled theta.
    
    Args:
        optimization_state: State containing triggering tests.
        num_examples: Number of examples to select (N).
        
    Returns:
        List of selected TriggeringTest instances, sorted by sampled theta (descending).
    """
    if not optimization_state.triggering_tests:
        return []
    
    tests = list(optimization_state.triggering_tests.values())
    
    # Sample theta for each test
    sampled_thetas = []
    for test in tests:
        theta = np.random.beta(test.alpha, test.beta)
        sampled_thetas.append((theta, test))
    
    # Sort by theta descending
    sampled_thetas.sort(key=lambda x: x[0], reverse=True)
    
    # Select top N
    selected = [test for _, test in sampled_thetas[:num_examples]]
    
    return selected


def create_new_triggering_test(
    optimization_name: str,
    file_path: Path,
    alpha: float = 1.0,
    beta: float = 1.0
) -> TriggeringTest:
    """
    Create a new TriggeringTest with initial Beta parameters.
    
    Args:
        optimization_name: Internal name of the optimization.
        file_path: Path to the test file.
        alpha: Initial alpha parameter (default 1.0 for uniform prior).
        beta: Initial beta parameter (default 1.0 for uniform prior).
        
    Returns:
        New TriggeringTest instance.
    """
    test_id = f"{optimization_name}-{uuid.uuid4().hex[:8]}"
    
    return TriggeringTest(
        test_id=test_id,
        optimization_name=optimization_name,
        file_path=file_path,
        alpha=alpha,
        beta=beta,
        total_generated_from=0,
        total_triggers_from=0,
    )


def update_bandit_after_generation(
    optimization_state: OptimizationState,
    example_tests: List[TriggeringTest],
    num_triggered: int,
    num_not_triggered: int,
    new_triggering_tests: List[Path]
) -> None:
    """
    Update bandit state after generating and testing a batch.
    
    Updates:
    - Example tests: alpha += num_triggered, beta += num_not_triggered
    - New triggering tests: initialized with average alpha/beta from examples
    
    Args:
        optimization_state: State to update.
        example_tests: Tests that were used as examples in this iteration.
        num_triggered: Number of new tests that triggered the optimization.
        num_not_triggered: Number of new tests that did not trigger.
        new_triggering_tests: Paths to newly discovered triggering tests.
    """
    batch_size = num_triggered + num_not_triggered
    
    # Update example tests
    for example_test in example_tests:
        if example_test.test_id in optimization_state.triggering_tests:
            test = optimization_state.triggering_tests[example_test.test_id]
            test.alpha += num_triggered
            test.beta += num_not_triggered
            test.total_generated_from += batch_size
            test.total_triggers_from += num_triggered
    
    # Compute average alpha/beta from example tests (for initializing new tests)
    if example_tests:
        avg_alpha = sum(t.alpha for t in example_tests) / len(example_tests)
        avg_beta = sum(t.beta for t in example_tests) / len(example_tests)
    else:
        # Default if no examples yet
        avg_alpha = 1.0
        avg_beta = 1.0
    
    # Initialize new triggering tests
    for test_path in new_triggering_tests:
        new_test = create_new_triggering_test(
            optimization_state.spec.internal_name,
            test_path,
            alpha=avg_alpha,
            beta=avg_beta,
        )
        optimization_state.triggering_tests[new_test.test_id] = new_test

