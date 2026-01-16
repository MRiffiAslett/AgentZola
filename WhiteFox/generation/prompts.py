"""
Prompt templates and optimization specifications for WhiteFox generation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from domain.bandit import TriggeringTest

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSpec:
    internal_name: str
    pass_log_names: List[str]  # List of possible pass names from instrumentation
    requirement_prompt_path: Path
    requirement_text: str
    
    @property
    def pass_log_name(self) -> str:
        """Backward compatibility: return the first pass log name."""
        return self.pass_log_names[0] if self.pass_log_names else ""
    
    def matches_any_pass(self, triggered_passes: set) -> bool:
        """Check if any of the pass log names match the triggered passes."""
        return bool(set(self.pass_log_names) & triggered_passes)


def load_optimization_specs(
    req_dir: Path, 
    optimizations: Optional[List[str]] = None,
    pass_name_aliases: Optional[Dict[str, List[str]]] = None
) -> Dict[str, OptimizationSpec]:
    """
    Load optimization specs from requirement directory.
    
    Args:
        req_dir: Directory containing requirement .txt files
        optimizations: Optional list of optimization names to load (if None, loads all .txt files)
        pass_name_aliases: Optional mapping from optimization names to pass log name aliases.
                          Pass name aliases should be loaded from config file. If None, optimization
                          names will be used as-is.
    """
    specs = {}
    
    # Use config mapping if provided, otherwise use optimization names as-is
    aliases_mapping = pass_name_aliases or {}
    
    if not req_dir.exists():
        raise FileNotFoundError(f"Requirement directory not found: {req_dir}")
    
    if optimizations is not None:
        for opt_name in optimizations:
            txt_file = req_dir / f"{opt_name}.txt"
            if not txt_file.exists():
                raise FileNotFoundError(f"Requirement file not found: {txt_file}")
            
            requirement_text = txt_file.read_text()
            pass_log_names = aliases_mapping.get(opt_name)
            if pass_log_names is None:
                logger.warning(
                    f"Optimization '{opt_name}' not found in pass name aliases mapping. "
                    f"Using optimization name as-is. Consider adding it to the mapping."
                )
                pass_log_names = [opt_name]
            
            spec = OptimizationSpec(
                internal_name=opt_name,
                pass_log_names=pass_log_names,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            specs[opt_name] = spec
    else:
        for txt_file in sorted(req_dir.glob("*.txt")):
            internal_name = txt_file.stem
            requirement_text = txt_file.read_text()
            pass_log_names = aliases_mapping.get(internal_name)
            if pass_log_names is None:
                logger.warning(
                    f"Optimization '{internal_name}' not found in pass name aliases mapping. "
                    f"Using optimization name as-is. Consider adding it to the mapping."
                )
                pass_log_names = [internal_name]
            
            spec = OptimizationSpec(
                internal_name=internal_name,
                pass_log_names=pass_log_names,
                requirement_prompt_path=txt_file,
                requirement_text=requirement_text,
            )
            specs[internal_name] = spec
    
    return specs



def build_base_prompt(spec: OptimizationSpec) -> str:
    """Build base prompt using requirement text from txt file."""
    
    return f"""
{spec.requirement_text}
"""

def build_feedback_prompt(
    spec: OptimizationSpec, 
    example_tests: List[TriggeringTest],
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

{feedback_instruction}

{examples_section}

"""
    
    return prompt

