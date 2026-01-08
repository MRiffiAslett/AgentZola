"""
Prompt templates and optimization specifications for WhiteFox generation.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from domain.bandit import TriggeringTest


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


def build_base_prompt(spec: OptimizationSpec) -> str:
    """Build base prompt using requirement text from txt file."""
    return f"""{spec.requirement_text}

Tensor shape safety: Before emitting any TensorFlow operation, explicitly reason about and validate tensor ranks, shapes, and element counts, ensuring all reshape, broadcast, split, tile, and convolution operations are mathematically compatible.

Code-only output constraint: Output strictly valid, executable Python code only—do not include comments, explanations, markdown, examples, or natural-language text inside the code region.

Runable code: Ensure that the code is ready to run with the neccesary import statements. Make sure to incldue  dummy data of some sort to probe the function. 

Do not produce any markdown or natural language text in the entire output.

Expected Output Format for the entire output:
```python
<code>
```

"""

def build_feedback_prompt(
    spec: OptimizationSpec, 
    example_tests: List[TriggeringTest]
) -> str:
    """Build feedback prompt with successful examples."""
    feedback_instruction = """Please generate different valid TensorFlow model that satisfies requirements below.
You should only use public TensorFlow APIs. The new model MUST be semantically different from
the examples shown below (no trivial renames or copy-paste)."""
    
    # Format examples
    examples = []
    for test in example_tests:
        try:
            test_content = test.file_path.read_text()
            examples.append(f"# Example\n{test_content}")
        except Exception:
            continue
    
    examples_section = "\n\n".join(examples)
    
    prompt = f"""

{spec.requirement_text}

Tensor shape safety: Before emitting any TensorFlow operation, explicitly reason about and validate tensor ranks, shapes, and element counts, ensuring all reshape, broadcast, split, tile, and convolution operations are mathematically compatible.

Code-only output constraint: Output strictly valid, executable Python code only—do not include comments, explanations, markdown, examples, or natural-language text inside the code region.

Runable code: Ensure that the code is ready to run with the neccesary import statements. Make sure to incldue  dummy data of some sort to probe the function. 

Do not produce any markdown or natural language text in the entire output.

Expected Output Format for the entire output:
```python
<code>
```

{feedback_instruction}

{examples_section}

"""
    
    return prompt

