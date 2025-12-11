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
    internal_name: str
    pass_log_name: str
    requirement_prompt_path: Path
    requirement_text: str


def camel_to_kebab(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()


PASS_NAME_OVERRIDES: Dict[str, str] = {
    "BroadcastCanonicalizer": "broadcast_canonicalizer",
    "DotDecomposer": "dot_decomposer",
    "StochasticConvertDecomposer": "stochastic_convert_decomposer",
    "TreeReductionRewriter": "tree_reduction_rewriter",
    "ZeroSizedHloElimination": "zero_sized_hlo_elimination",
}


def load_optimization_specs(
    req_dir: Path, 
    optimizations: Optional[List[str]] = None
) -> Dict[str, OptimizationSpec]:
    specs = {}
    
    if not req_dir.exists():
        raise FileNotFoundError(f"Requirement directory not found: {req_dir}")
    
    if optimizations is not None:
        optimization_set = set(optimizations)
        for opt_name in optimizations:
            txt_file = req_dir / f"{opt_name}.txt"
            if not txt_file.exists():
                raise FileNotFoundError(f"Requirement file not found: {txt_file}")
            
            requirement_text = txt_file.read_text()
            
            pass_log_name = PASS_NAME_OVERRIDES.get(opt_name, camel_to_kebab(opt_name))
            
            spec = OptimizationSpec(
                internal_name=opt_name,
                pass_log_name=pass_log_name,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            
            specs[opt_name] = spec
    else:
        for txt_file in sorted(req_dir.glob("*.txt")):
            internal_name = txt_file.stem
            
            requirement_text = txt_file.read_text()
            
            pass_log_name = PASS_NAME_OVERRIDES.get(internal_name, camel_to_kebab(internal_name))
            
            spec = OptimizationSpec(
                internal_name=internal_name,
                pass_log_name=pass_log_name,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            
            specs[internal_name] = spec
    
    return specs

