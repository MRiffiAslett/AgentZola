"""
Prompt templates and optimization specifications for WhiteFox generation.

Provides base and feedback prompt templates for generating TensorFlow-XLA test programs,
and manages optimization specifications and requirement prompts.
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

try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None


BASE_PROMPT_TEMPLATE = """### Please generate one valid TensorFlow model that satisfies requirements below.
You should only use public TensorFlow APIs.

# Description
{requirement_text}

# Model
"""


FEEDBACK_PROMPT_TEMPLATE = """### Please generate different valid TensorFlow model that satisfies requirements below.
You should only use public TensorFlow APIs. The new model MUST be semantically different from
the examples shown below (no trivial renames or copy-paste).

# Description
{requirement_text}

{examples_section}

# Model
"""


def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    if ENCODING:
        try:
            return len(ENCODING.encode(text))
        except Exception:
            pass
    return len(text) // 4


def format_example_section(
    triggering_tests: List[TriggeringTest], 
    max_tokens: Optional[int] = None
) -> str:
    """
    Format example section, truncating if necessary to stay within token limit.
    
    Args:
        triggering_tests: List of triggering tests to include
        max_tokens: Maximum tokens allowed (None = no limit)
    """
    examples = []
    current_tokens = 0
    
    for test in triggering_tests:
        try:
            test_content = test.file_path.read_text()
            example_text = f"# Model\n{test_content}"
            example_tokens = estimate_tokens(example_text)
            
            if max_tokens is not None:
                if current_tokens + example_tokens > max_tokens:
                    remaining = max_tokens - current_tokens
                    if remaining > 100:
                        chars_to_keep = remaining * 4
                        truncated = test_content[:chars_to_keep]
                        example_text = f"# Model\n{truncated}\n# ... (truncated)"
                        examples.append(example_text)
                    break
            
            examples.append(example_text)
            current_tokens += example_tokens
            
        except Exception as e:
            continue
    
    return "\n\n".join(examples)


def build_base_prompt(spec: OptimizationSpec) -> str:
    return BASE_PROMPT_TEMPLATE.format(requirement_text=spec.requirement_text)


def build_feedback_prompt(
    spec: OptimizationSpec, 
    example_tests: List[TriggeringTest],
    max_model_len: int = 4096,
    reserved_tokens: int = 500
) -> str:
    """
    Build feedback prompt with examples, ensuring it doesn't exceed max_model_len.
    
    Args:
        spec: Optimization specification
        example_tests: List of example tests
        max_model_len: Maximum model length in tokens
        reserved_tokens: Tokens to reserve for prompt template and generation
    """
    base_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        requirement_text=spec.requirement_text,
        examples_section=""
    )
    base_tokens = estimate_tokens(base_prompt)
    available_tokens = max_model_len - base_tokens - reserved_tokens
    
    examples_section = format_example_section(example_tests, max_tokens=available_tokens)
    
    return FEEDBACK_PROMPT_TEMPLATE.format(
        requirement_text=spec.requirement_text,
        examples_section=examples_section,
    )

