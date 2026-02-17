from typing import Dict, Tuple

from requirement.base.base import RequirementGenerator
from requirement.prompt_gen import Src2NLTFXLA, generate_requirement_prompts


class XLARequirementGenerator(RequirementGenerator):

    def get_system_message(self) -> str:
        return "You are a source code analyzer for TensorFlow XLA."

    def get_instruction_template(self) -> str:
        return (
            "### Please generate one valid TensorFlow model that satisfies requirements below.\n"
            "You should only use public TensorFlow APIs. The model can be used as the input "
            "to trigger the optimization pass `{opt_name}` in TensorFlow XLA.\n\n"
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
        return generate_requirement_prompts(
            optpath=opt_spec_path,
            template_path=template_path,
            outdir=outdir,
            use_mini=use_mini,
            fallback_dir=fallback_dir,
        )
