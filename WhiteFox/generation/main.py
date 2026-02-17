import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generation.generator import StarCoderGenerator

# Default config paths per SUT (relative to project root)
_DEFAULT_CONFIG = {
    "xla": "xilo_xla/config/generator.toml",
    "inductor": "xilo_inductor/config/generator.toml",
    "tflite": "xilo_tflite/config/generator.toml",
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sut",
        type=str,
        choices=sorted(_DEFAULT_CONFIG),
        default="xla",
        help="Only used to pick the default --config when --config is omitted.",
    )
    parser.add_argument(
        "--only-opt",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.config is None:
        args.config = project_root / _DEFAULT_CONFIG[args.sut]
        print(f"Using default config for SUT '{args.sut}': {args.config}")

    generator = StarCoderGenerator.from_config_file(args.config)

    only_optimizations = None
    if args.only_opt:
        only_optimizations = [opt.strip() for opt in args.only_opt.split(",")]
        print(f"Processing only optimizations: {only_optimizations}")

    try:
        generator.generate_whitefox(only_optimizations=only_optimizations)
    except Exception as e:
        print(f"Error during WhiteFox fuzzing: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
