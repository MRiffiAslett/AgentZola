from typing import Dict, List

from generation.harness.base import TestHarness


class TensorFlowLiteHarness(TestHarness):

    def get_execution_modes(self) -> List[str]:
        return ["reference", "tflite"]

    def get_env_vars(self) -> Dict[str, str]:
        return {}

    def extract_triggered_passes(self, log_text: str) -> set:
        raise NotImplementedError(
            "TensorFlow Lite pass extraction is not yet implemented."
        )

    def generate_wrapper_script(self, test_code_repr: str) -> str:
        raise NotImplementedError(
            "TensorFlow Lite wrapper script is not yet implemented."
        )
