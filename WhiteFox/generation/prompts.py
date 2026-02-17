import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from domain.bandit import TriggeringTest

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSpec:
    internal_name: str
    pass_log_names: List[str]
    requirement_prompt_path: Path
    requirement_text: str

    @property
    def pass_log_name(self) -> str:
        return self.pass_log_names[0] if self.pass_log_names else ""

    def matches_any_pass(self, triggered_passes: set) -> bool:
        return bool(set(self.pass_log_names) & triggered_passes)


def _create_optimization_spec(
    opt_name: str, txt_file: Path, aliases_mapping: Dict[str, List[str]]
) -> OptimizationSpec:

    requirement_text = txt_file.read_text()
    pass_log_names = aliases_mapping.get(opt_name)
    if pass_log_names is None:
        logger.warning(
            f"Optimization '{opt_name}' not found in pass name aliases mapping. "
            f"Using optimization name as-is. Consider adding it to the mapping."
        )
        pass_log_names = [opt_name]

    return OptimizationSpec(
        internal_name=opt_name,
        pass_log_names=pass_log_names,
        requirement_prompt_path=txt_file,
        requirement_text=requirement_text,
    )


def load_optimization_specs(
    req_dir: Path,
    optimizations: Optional[List[str]] = None,
    pass_name_aliases: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, OptimizationSpec]:
    specs = {}

    aliases_mapping = pass_name_aliases or {}

    if not req_dir.exists():
        raise FileNotFoundError(f"Requirement directory not found: {req_dir}")

    if optimizations is not None:
        for opt_name in optimizations:
            txt_file = req_dir / f"{opt_name}.txt"
            if not txt_file.exists():
                raise FileNotFoundError(f"Requirement file not found: {txt_file}")

            specs[opt_name] = _create_optimization_spec(
                opt_name, txt_file, aliases_mapping
            )
    else:
        for txt_file in sorted(req_dir.glob("*.txt")):
            internal_name = txt_file.stem
            specs[internal_name] = _create_optimization_spec(
                internal_name, txt_file, aliases_mapping
            )

    return specs


def build_base_prompt(spec: OptimizationSpec) -> str:
    opt_name = spec.internal_name

    return (
        f"### Please generate one valid TensorFlow model that satisfies requirements below.\n"
        f"You should only use public TensorFlow APIs. The model can be used as the input "
        f"to trigger the optimization pass `{opt_name}` in TensorFlow XLA.\n"
        f"\n"
        f"# Description\n"
        f"{spec.requirement_text.strip()}\n"
        f"\n"
        f"# Model"
    )


def build_feedback_prompt(
    spec: OptimizationSpec,
    example_tests: List[TriggeringTest],
    feedback_instruction: str,
) -> str:
    opt_name = spec.internal_name

    examples = []
    for test in example_tests:
        try:
            test_content = test.file_path.read_text()
            examples.append(f"# Model begins\n{test_content.strip()}\n# Model ends")
        except Exception:
            continue

    examples_section = "\n\n".join(examples)

    return (
        f"### Please generate a valid TensorFlow model with public TensorFlow APIs. "
        f"The model can trigger the optimization pass `{opt_name}` in TensorFlow XLA. "
        f"Additionally, please generate valid input tensors for the model and pass to the model.\n"
        f"\n"
        f"# Description\n"
        f"{spec.requirement_text.strip()}\n"
        f"\n"
        f"{examples_section}\n"
        f"\n"
        f"# Model begins"
    )
