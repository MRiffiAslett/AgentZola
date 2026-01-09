#!/usr/bin/env python3
"""
Diagnostic script to check WhiteFox optimization loading.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tomllib
from WhiteFox.generation.prompts import load_optimization_specs

def main():
    # Read config
    config_path = Path(__file__).parent / "xilo_xla" / "config" / "generator.toml"
    print(f"Reading config from: {config_path}")
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    # Get optimization settings
    optimizations_dir = Path(__file__).parent / Path(config["generation"]["optimizations_dir"])
    optimizations_list = config["generation"].get("optimizations", None)
    
    print(f"\nOptimizations directory: {optimizations_dir}")
    print(f"Optimizations directory exists: {optimizations_dir.exists()}")
    
    if optimizations_list:
        print(f"\nOptimizations specified in config: {len(optimizations_list)}")
        print("First 10:", optimizations_list[:10])
    else:
        print("\nNo optimizations list specified - will load all .txt files")
    
    # Count actual files
    if optimizations_dir.exists():
        txt_files = list(optimizations_dir.glob("*.txt"))
        print(f"\nActual .txt files in directory: {len(txt_files)}")
        print("First 10 files:", [f.stem for f in sorted(txt_files)[:10]])
    else:
        print(f"\nERROR: Directory does not exist!")
        return
    
    # Try loading specs
    print("\n" + "="*70)
    print("Attempting to load optimization specs...")
    print("="*70)
    
    try:
        specs = load_optimization_specs(optimizations_dir, optimizations_list)
        print(f"\n✓ Successfully loaded {len(specs)} optimization specifications:")
        for i, opt_name in enumerate(sorted(specs.keys()), 1):
            spec = specs[opt_name]
            print(f"  {i:2d}. {opt_name:40s} -> pass: {spec.pass_log_name}")
        
        # Check for any missing files
        if optimizations_list:
            missing = []
            for opt_name in optimizations_list:
                txt_file = optimizations_dir / f"{opt_name}.txt"
                if not txt_file.exists():
                    missing.append(opt_name)
            
            if missing:
                print(f"\n⚠ WARNING: {len(missing)} optimizations from config have no .txt file:")
                for opt_name in missing:
                    print(f"  - {opt_name}")
            else:
                print(f"\n✓ All {len(optimizations_list)} optimizations from config have corresponding .txt files")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        return
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

