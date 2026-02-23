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


def _make_spec(
    name: str, txt_file: Path, aliases: Dict[str, List[str]]
) -> OptimizationSpec:
    text = txt_file.read_text()
    log_names = aliases.get(name)
    if log_names is None:
        logger.warning(f"No pass alias for '{name}', using as-is.")
        log_names = [name]
    return OptimizationSpec(
        internal_name=name,
        pass_log_names=log_names,
        requirement_prompt_path=txt_file,
        requirement_text=text,
    )


def load_optimization_specs(
    req_dir: Path,
    optimizations: Optional[List[str]] = None,
    pass_name_aliases: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, OptimizationSpec]:
    specs: Dict[str, OptimizationSpec] = {}
    aliases = pass_name_aliases or {}

    if not req_dir.exists():
        raise FileNotFoundError(f"Requirement directory not found: {req_dir}")

    if optimizations is not None:
        for name in optimizations:
            txt = req_dir / f"{name}.txt"

            specs[name] = _make_spec(name, txt, aliases)
    else:
        for txt in sorted(req_dir.glob("*.txt")):
            specs[txt.stem] = _make_spec(txt.stem, txt, aliases)

    return specs


def build_base_prompt(
    spec: OptimizationSpec,
    prompt_style: str = "paper",
    seed_qa: str = "",
) -> str:
    target_q = spec.requirement_text.strip()

    if prompt_style == "stacked" and seed_qa:
        return f"{seed_qa}\n\n{target_q}\n\n# Model"

    return f"{target_q}\n\n# Model"


def build_feedback_prompt(
    spec: OptimizationSpec,
    example_tests: List[TriggeringTest],
    feedback_instruction: str = "",
    prompt_style: str = "paper",
    seed_qa: str = "",
) -> str:
    target_q = spec.requirement_text.strip()

    examples: List[str] = []
    for test in example_tests:
        try:
            code = test.file_path.read_text().strip()
            if code:
                examples.append(f"# Model begins\n{code}\n# Model ends")
        except Exception:
            continue

    body = "\n\n".join(examples)

    parts: List[str] = []

    if prompt_style == "stacked" and seed_qa:
        parts.append(seed_qa)

    if feedback_instruction:
        parts.append(feedback_instruction)

    parts.append(target_q)

    if body:
        parts.append(body)

    parts.append("# Model begins")

    return "\n\n".join(parts)
