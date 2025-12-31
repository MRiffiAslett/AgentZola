"""
Prompt templates for WhiteFox generation.

Provides base and feedback prompt templates for generating TensorFlow-XLA test programs.
"""

from pathlib import Path
from typing import List, Optional

from generation.spec import OptimizationSpec
from domain.bandit import TriggeringTest

# Try to get encoding for token counting
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
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
    # Fallback: rough estimate (1 token ≈ 4 characters)
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
            
            # Check if adding this example would exceed limit
            if max_tokens is not None:
                if current_tokens + example_tokens > max_tokens:
                    # Try to include partial example if there's room
                    remaining = max_tokens - current_tokens
                    if remaining > 100:  # Only if meaningful space remains
                        # Truncate the example
                        chars_to_keep = remaining * 4  # rough conversion
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
    reserved_tokens: int = 500  # Reserve tokens for prompt template and generation
) -> str:
    """
    Build feedback prompt with examples, ensuring it doesn't exceed max_model_len.
    
    Args:
        spec: Optimization specification
        example_tests: List of example tests
        max_model_len: Maximum model length in tokens
        reserved_tokens: Tokens to reserve for prompt template and generation
    """
    # Calculate available tokens for examples
    base_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        requirement_text=spec.requirement_text,
        examples_section=""
    )
    base_tokens = estimate_tokens(base_prompt)
    available_tokens = max_model_len - base_tokens - reserved_tokens
    
    # Format examples with token limit
    examples_section = format_example_section(example_tests, max_tokens=available_tokens)
    
    return FEEDBACK_PROMPT_TEMPLATE.format(
        requirement_text=spec.requirement_text,
        examples_section=examples_section,
    )

