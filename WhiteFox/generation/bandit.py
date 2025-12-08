"""
Thompson Sampling bandit functions.

Contains functions for bandit operations: selecting examples, creating tests,
and updating bandit state after generation.
"""

import uuid
from pathlib import Path
from typing import List

import numpy as np

from ..domain.bandit import OptimizationState, TriggeringTest


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

