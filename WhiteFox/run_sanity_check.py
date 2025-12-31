#!/usr/bin/env python3
"""
Standalone script to run WhiteFox sanity check.

Usage:
    python run_sanity_check.py [--logging-dir PATH]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generation.sanity_checker import run_sanity_check
from domain.bandit import WhiteFoxState
from generation.spec import load_optimization_specs


def main():
    parser = argparse.ArgumentParser(
        description="Run WhiteFox sanity check"
    )
    parser.add_argument(
        "--logging-dir",
        type=Path,
        default=None,
        help="Path to logging directory (default: WhiteFox/logging)"
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to whitefox_state.json (default: logging_dir/whitefox_state.json)"
    )
    
    args = parser.parse_args()
    
    # Determine logging directory
    if args.logging_dir:
        logging_dir = Path(args.logging_dir)
    else:
        # Find WhiteFox directory
        cwd = Path.cwd()
        if cwd.name == "WhiteFox":
            logging_dir = cwd / "logging"
        else:
            current = cwd
            project_root = None
            while current != current.parent:
                if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                    project_root = current / "WhiteFox"
                    break
                current = current.parent
            
            if project_root:
                logging_dir = project_root / "logging"
            else:
                logging_dir = Path("WhiteFox/logging")
    
    if not logging_dir.exists():
        print(f"Error: Logging directory not found: {logging_dir}")
        sys.exit(1)
    
    # Load state if available
    state = None
    if args.state_file:
        state_file = Path(args.state_file)
    else:
        state_file = logging_dir / "whitefox_state.json"
    
    if state_file.exists():
        try:
            # Need to load specs to load state
            # For now, just try to load state
            with open(state_file, 'r') as f:
                import json
                state_data = json.load(f)
            # Create minimal state for checking
            from domain.bandit import WhiteFoxState
            state = WhiteFoxState(optimizations={})
            # We'll check structure even without full spec loading
        except Exception as e:
            print(f"Warning: Could not load state file: {e}")
    
    # Run sanity check
    print(f"Running sanity check on: {logging_dir}")
    try:
        json_file, text_file = run_sanity_check(logging_dir, state)
        print(f"\nSanity check complete!")
        print(f"  JSON report: {json_file}")
        print(f"  Text report: {text_file}")
        print(f"\nTo view the report, run:")
        print(f"  cat {text_file}")
    except Exception as e:
        print(f"Error running sanity check: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

