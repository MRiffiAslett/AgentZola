from typing import List, Tuple

from generation.code_processing.base import CodeParser


class PyTorchCodeParser(CodeParser):

    DEFAULT_TENSOR_VALUE = "torch.randn(4)"

    def split_func_tensor(self, code: str) -> Tuple[str, str, List[str], str]:
        raise NotImplementedError(
            "PyTorch code parsing is not yet implemented."
        )

    def preprocessing(self, code: str) -> str:
        raise NotImplementedError(
            "PyTorch code preprocessing is not yet implemented."
        )

    def process_code(self, code: str) -> str:
        raise NotImplementedError(
            "PyTorch code processing is not yet implemented."
        )
