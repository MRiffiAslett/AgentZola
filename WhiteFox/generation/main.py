"""
WhiteFox: White-Box Compiler Fuzzing Empowered by Large Language Models.

CLI entry point for WhiteFox compiler fuzzer.
"""

import sys
import traceback
from pathlib import Path
from pprint import pprint

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generation.generator import StarCoderGenerator


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
        print("\nInterrupted by user. Saving state and generating summary...")
        logging_dir = generator._get_logging_dir() if hasattr(generator, '_get_logging_dir') else None
        if logging_dir:
            source_dir = logging_dir / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            state_file = source_dir / "whitefox_state.json"
        else:
            state_file = Path(generator.config.paths.bandit_state_file or "whitefox_state.json")
        if hasattr(generator, 'whitefox_state'):
            generator.whitefox_state.save(state_file)
            print(f"State saved to {state_file}")
            
            # Generate run summary with partial results
            try:
                from generation.logging import WhiteFoxLogger
                whitefox_logger = WhiteFoxLogger(logging_dir, generator.logger)
                whitefox_logger.generate_run_summary(generator.whitefox_state)
                print(f"Run summary saved to {logging_dir / 'run_summary_detailed.log'}")
            except Exception as summary_error:
                print(f"Warning: Could not generate run summary: {summary_error}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error during WhiteFox fuzzing: {e}", file=sys.stderr)
        traceback.print_exc()
        
        # Try to generate run summary with partial results before exiting
        try:
            logging_dir = generator._get_logging_dir() if hasattr(generator, '_get_logging_dir') else None
            if logging_dir and hasattr(generator, 'whitefox_state'):
                from generation.logging import WhiteFoxLogger
                whitefox_logger = WhiteFoxLogger(logging_dir, generator.logger)
                whitefox_logger.generate_run_summary(generator.whitefox_state)
                print(f"Run summary saved to {logging_dir / 'run_summary_detailed.log'}", file=sys.stderr)
        except Exception as summary_error:
            print(f"Warning: Could not generate run summary: {summary_error}", file=sys.stderr)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
