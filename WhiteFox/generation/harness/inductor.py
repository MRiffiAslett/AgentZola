from typing import Dict, List

from generation.harness.base import TestHarness


class PyTorchInductorHarness(TestHarness):

    def get_execution_modes(self) -> List[str]:
        return ["eager", "jit", "inductor"]

    def get_env_vars(self) -> Dict[str, str]:
        return {
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        }

    def extract_triggered_passes(self, log_text: str) -> set:
        raise NotImplementedError(
            "PyTorch Inductor pass extraction is not yet implemented."
        )

    def generate_wrapper_script(self, test_code_repr: str) -> str:
        raise NotImplementedError(
            "PyTorch Inductor wrapper script is not yet implemented."
        )
