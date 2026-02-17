from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple


class RequirementGenerator(ABC):

    @abstractmethod
    def get_system_message(self) -> str:
        ...

    @abstractmethod
    def get_instruction_template(self) -> str:
        ...

    @abstractmethod
    def generate_prompts(
        self,
        opt_spec_path: str,
        template_path: str,
        outdir: str,
        use_mini: bool = False,
        fallback_dir: str = None,
    ) -> Tuple[Dict[str, str], set]:
        ...
