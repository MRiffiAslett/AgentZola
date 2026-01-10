"""
Prompt templates and optimization specifications for WhiteFox generation.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from domain.bandit import TriggeringTest


# Single-shot example for few-shot prompting
SINGLE_SHOT_EXAMPLE = """
<example>
import tensorflow as tf

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = SimpleModel()
x = tf.random.normal([32, 20])
output = model(x)
print(output.shape)
</example>
""".strip()


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


def parse_generated_code(response: str) -> str:
    """
    Parse generated code from LLM response, handling various formats.
    
    Attempts to extract pure Python code from:
    - Markdown code blocks (```python ... ```)
    - XML-like tags (<example>...</example>)
    - Plain code with explanations
    
    Args:
        response: Raw LLM response
        
    Returns:
        Cleaned Python code string
    """
    # Strip leading/trailing whitespace
    response = response.strip()
    
    # Try to extract from markdown code blocks
    markdown_pattern = r'```(?:python)?\s*\n(.*?)```'
    markdown_match = re.search(markdown_pattern, response, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()
    
    # Try to extract from <example> tags
    example_pattern = r'<example>\s*\n?(.*?)</example>'
    example_match = re.search(example_pattern, response, re.DOTALL)
    if example_match:
        return example_match.group(1).strip()
    
    # Try to extract from <code> tags
    code_pattern = r'<code>\s*\n?(.*?)</code>'
    code_match = re.search(code_pattern, response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no special markers found, look for import tensorflow as first valid line
    lines = response.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            start_idx = i
            break
    
    # Find end: stop at common explanation markers
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if any(line.startswith(marker) for marker in ['Note:', 'Explanation:', '---', '###']):
            end_idx = i
            break
    
    # Return the code section
    if start_idx < len(lines):
        return '\n'.join(lines[start_idx:end_idx]).strip()
    
    # Fallback: return as-is
    return response


def build_base_prompt(spec: OptimizationSpec, include_example: bool = True) -> str:
    """Build base prompt using requirement text from txt file."""
    
    example_section = ""
    if include_example:
        example_section = f"""
Here is the expected output format. Your output should be wrapped in <example> tags like this:

{SINGLE_SHOT_EXAMPLE}

"""
    
    return f"""

You are generating a one  TensorFlow Python program.

Hard requirements:
- Output ONLY valid Python code (no comments, no markdown, no explanations).
- The program must run end-to-end without errors.
- Import TensorFlow explicitly and use real TensorFlow APIs.
- Create all inputs using mocked/synthetic data (e.g. tf.random, constants).
- Execute a real computation (model call, loss, gradient, or tensor ops).
- Force execution via print(), .numpy(), or returned values.



{example_section}

"""

def build_feedback_prompt(
    spec: OptimizationSpec, 
    example_tests: List[TriggeringTest],
    include_example: bool = True
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
    
    format_section = ""
    if include_example:
        format_section = f"""
OUTPUT FORMAT (MANDATORY):
Wrap your code in <example> tags like this:

{SINGLE_SHOT_EXAMPLE}

"""
    
    prompt = f"""

{spec.requirement_text}
You are a code-generation engine.

HARD OUTPUT CONTRACT (MANDATORY):
- Output ONLY syntactically valid Python code.
- Do NOT include comments, explanations, markdown, backticks, or any natural-language text outside the tags.
- Do NOT include leading or trailing whitespace outside the code.
- The output will be executed directly via python.
- Any non-code token will cause immediate failure.

{format_section}

SEMANTIC CONSTRAINTS:
- TensorFlow 2.x only.
- Build at least one tf.keras.Model.
- Create dummy input tensors and execute a forward pass.
- Ensure all operations are mathematically shape-safe.

FAILURE MODE:
- If any rule cannot be satisfied, output nothing.

TASK:
Generate the program.


{feedback_instruction}

{examples_section}

"""
    
    return prompt

