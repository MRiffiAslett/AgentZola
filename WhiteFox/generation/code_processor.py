from generation.code_processing.tensorflow import TensorFlowCodeParser

TFCodeParserJIT = TensorFlowCodeParser

_default_parser = TensorFlowCodeParser()


def process_code(code: str) -> str:
    return _default_parser.process_code(code)
