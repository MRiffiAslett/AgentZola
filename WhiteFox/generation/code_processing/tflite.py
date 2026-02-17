from typing import List, Tuple

from generation.code_processing.base import CodeParser


class TensorFlowLiteCodeParser(CodeParser):

    DEFAULT_TENSOR_VALUE = "tf.constant([1.0, 2.0, 3.0, 4.0])"

    def split_func_tensor(self, code: str) -> Tuple[str, str, List[str], str]:
        raise NotImplementedError(
            "TensorFlow Lite code parsing is not yet implemented."
        )

    def preprocessing(self, code: str) -> str:
        raise NotImplementedError(
            "TensorFlow Lite code preprocessing is not yet implemented."
        )

    def process_code(self, code: str) -> str:
        raise NotImplementedError(
            "TensorFlow Lite code processing is not yet implemented."
        )
