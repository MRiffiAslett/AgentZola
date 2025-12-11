"""
WhiteFox: White-Box Compiler Fuzzing Empowered by Large Language Models.

CLI entry point for WhiteFox compiler fuzzer.
"""

import sys
import traceback
from pathlib import Path
from pprint import pprint

from AgentZola.WhiteFox.generation.generator import StarCoderGenerator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WhiteFox: White-Box Compiler Fuzzing using VLLM with TOML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file (required)"
    )
    parser.add_argument(
        "--only-opt",
        type=str,
        default=None,
        help="Comma-separated list of optimization names to process"
    )
    
    args = parser.parse_args()
    
    print("Configuration:")
    pprint(vars(args))
    
    try:
        generator = StarCoderGenerator.from_config_file(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    only_optimizations = None
    if args.only_opt:
        only_optimizations = [opt.strip() for opt in args.only_opt.split(",")]
        print(f"Processing only optimizations: {only_optimizations}")
    
    try:
        generator.generate_whitefox(only_optimizations=only_optimizations)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving state...")
        state_file = Path(generator.config.paths.bandit_state_file or "whitefox_state.json")
        if hasattr(generator, 'whitefox_state'):
            generator.whitefox_state.save(state_file)
            print(f"State saved to {state_file}")
        sys.exit(0)
    except Exception as e:
        print(f"Error during WhiteFox fuzzing: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
