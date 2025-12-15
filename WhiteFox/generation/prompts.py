"""
Prompt templates for WhiteFox generation.

Provides base and feedback prompt templates for generating TensorFlow-XLA test programs.
"""

from pathlib import Path
from typing import List

from generation.spec import OptimizationSpec
from domain.bandit import TriggeringTest


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


def format_example_section(triggering_tests: List[TriggeringTest]) -> str:
    examples = []
    for test in triggering_tests:
        try:
            test_content = test.file_path.read_text()
            examples.append(f"# Model\n{test_content}")
        except Exception as e:
            continue
    
    return "\n\n".join(examples)


def build_base_prompt(spec: OptimizationSpec) -> str:
    return BASE_PROMPT_TEMPLATE.format(requirement_text=spec.requirement_text)


def build_feedback_prompt(spec: OptimizationSpec, example_tests: List[TriggeringTest]) -> str:
    examples_section = format_example_section(example_tests)
    
    return FEEDBACK_PROMPT_TEMPLATE.format(
        requirement_text=spec.requirement_text,
        examples_section=examples_section,
    )

