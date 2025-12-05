"""
Optimization specification and metadata.

Defines OptimizationSpec and related data structures for tracking
XLA optimization passes and their requirement prompts.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class OptimizationSpec:
    """Specification for a targeted XLA optimization."""
    internal_name: str  # e.g. "AllGatherBroadcastReorder"
    pass_log_name: str  # e.g. "all-gather-broadcast-reorder" (matches pass= in logs)
    requirement_prompt_path: Path  # path to .txt file
    requirement_text: str  # contents of the requirement .txt


def camel_to_kebab(name: str) -> str:
    """
    Convert CamelCase to kebab-case.
    
    Examples:
        AllGatherBroadcastReorder -> all-gather-broadcast-reorder
        BatchDotSimplification -> batch-dot-simplification
    """
    # Insert hyphens before uppercase letters (except the first one)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    # Insert hyphens before uppercase letters that follow lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()


# Override mapping for cases where CamelCase -> kebab-case doesn't match actual pass names
# These are manually maintained based on actual TensorFlow XLA log output
PASS_NAME_OVERRIDES: Dict[str, str] = {
    # Underscores vs hyphens mismatches
    "BroadcastCanonicalizer": "broadcast_canonicalizer",
    "DotDecomposer": "dot_decomposer",
    "StochasticConvertDecomposer": "stochastic_convert_decomposer",
    "TreeReductionRewriter": "tree_reduction_rewriter",
    "ZeroSizedHloElimination": "zero_sized_hlo_elimination",
    # Add more overrides here as needed when testing reveals additional mismatches
}


def load_optimization_specs(
    req_dir: Path, 
    optimizations: Optional[List[str]] = None
) -> Dict[str, OptimizationSpec]:
    """
    Load optimization specifications from requirement prompt directory.
    
    Args:
        req_dir: Directory containing *.txt requirement prompt files.
        optimizations: Optional list of optimization names to load. If None, loads all .txt files.
        
    Returns:
        Dictionary mapping internal_name -> OptimizationSpec.
    """
    specs = {}
    
    if not req_dir.exists():
        raise FileNotFoundError(f"Requirement directory not found: {req_dir}")
    
    # If optimizations list is provided, use it; otherwise scan directory
    if optimizations is not None:
        optimization_set = set(optimizations)
        for opt_name in optimizations:
            txt_file = req_dir / f"{opt_name}.txt"
            if not txt_file.exists():
                raise FileNotFoundError(f"Requirement file not found: {txt_file}")
            
            # Read requirement text
            requirement_text = txt_file.read_text()
            
            # Derive pass_log_name: use override if available, otherwise convert CamelCase to kebab-case
            pass_log_name = PASS_NAME_OVERRIDES.get(opt_name, camel_to_kebab(opt_name))
            
            spec = OptimizationSpec(
                internal_name=opt_name,
                pass_log_name=pass_log_name,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            
            specs[opt_name] = spec
    else:
        # Legacy behavior: scan directory for all .txt files
        for txt_file in sorted(req_dir.glob("*.txt")):
            internal_name = txt_file.stem
            
            # Read requirement text
            requirement_text = txt_file.read_text()
            
            # Derive pass_log_name: use override if available, otherwise convert CamelCase to kebab-case
            pass_log_name = PASS_NAME_OVERRIDES.get(internal_name, camel_to_kebab(internal_name))
            
            spec = OptimizationSpec(
                internal_name=internal_name,
                pass_log_name=pass_log_name,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            
            specs[internal_name] = spec
    
    return specs

