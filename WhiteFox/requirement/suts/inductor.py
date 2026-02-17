from typing import Dict, Tuple

from requirement.base.base import RequirementGenerator


class InductorRequirementGenerator(RequirementGenerator):

    def get_system_message(self) -> str:
        return "You are a source code analyzer for PyTorch."

    def get_instruction_template(self) -> str:
        return (
            "### Please generate a valid PyTorch model example with public PyTorch APIs "
            "meets the specified requirements.\n\n"
            "# Description\n"
        )

    def generate_prompts(
        self,
        opt_spec_path: str,
        template_path: str,
        outdir: str,
        use_mini: bool = False,
        fallback_dir: str = None,
    ) -> Tuple[Dict[str, str], set]:
        raise NotImplementedError(
            "PyTorch Inductor requirement generation is not yet implemented."
        )
