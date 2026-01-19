#!/usr/bin/env python3
"""
Test script to simulate WhiteFox state initialization and see what optimizations would be processed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tomllib
from WhiteFox.generation.prompts import load_optimization_specs
from WhiteFox.domain.bandit import WhiteFoxState, OptimizationState

def simulate_state_initialization():
    """Simulate the state initialization process."""
    
    # Read config
    config_path = Path(__file__).parent / "xilo_xla" / "config" / "generator.toml"
    print(f"Reading config from: {config_path}\n")
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    # Get settings
    optimizations_dir = Path(__file__).parent / Path(config["generation"]["optimizations_dir"])
    optimizations_list = config["generation"].get("optimizations", None)
    
    print("="*70)
    print("STEP 1: Loading optimization specs")
    print("="*70)
    specs = load_optimization_specs(optimizations_dir, optimizations_list)
    print(f"✓ Loaded {len(specs)} optimization specifications\n")
    
    print("="*70)
    print("STEP 2: Initializing WhiteFox state")
    print("="*70)
    state = WhiteFoxState(optimizations={})
    
    for opt_name, spec in specs.items():
        state.optimizations[opt_name] = OptimizationState(spec=spec)
    
    print(f"✓ Initialized state with {len(state.optimizations)} optimizations\n")
    
    print("="*70)
    print("STEP 3: What would be processed (simulating iteration)")
    print("="*70)
    
    # Simulate what happens in generate_whitefox
    only_optimizations = None  # Set this to a list to test filtering
    
    if only_optimizations:
        print(f"Filtering enabled: only_optimizations = {only_optimizations}\n")
    else:
        print(f"No filtering: will process all optimizations\n")
    
    processed_count = 0
    skipped_count = 0
    
    for opt_state in state.optimizations.values():
        opt_name = opt_state.spec.internal_name
        
        # This is the filtering logic from _run_single_optimization
        if only_optimizations and opt_name not in only_optimizations:
            skipped_count += 1
            print(f"  [SKIP] {opt_name}")
            continue
        
        processed_count += 1
        print(f"  [PROCESS] {opt_name}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total optimizations in state: {len(state.optimizations)}")
    print(f"Would process: {processed_count}")
    print(f"Would skip: {skipped_count}")
    
    if processed_count == len(state.optimizations):
        print("\n✓ ALL optimizations would be processed!")
    else:
        print(f"\n⚠ WARNING: Only {processed_count}/{len(state.optimizations)} would be processed!")

if __name__ == "__main__":
    simulate_state_initialization()

