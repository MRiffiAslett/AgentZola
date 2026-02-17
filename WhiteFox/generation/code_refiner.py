import re
from typing import Optional

from generation.code_processing.base import CodeParser
from generation.code_processing.tensorflow import TensorFlowCodeParser

_default_parser = TensorFlowCodeParser()


def parse_generated_code(generated_text: str) -> str:
    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(code_block_pattern, generated_text, re.DOTALL)

    if matches:
        code = matches[-1].strip()
    else:
        text = generated_text.strip()

        prefixes_to_remove = [
            "Here is the code:",
            "Here's the code:",
            "Here is a TensorFlow model:",
            "Here's a TensorFlow model:",
            "Here is a PyTorch model:",
            "Here's a PyTorch model:",
        ]

        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
                break

        code = text

    filtered_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            english_indicators = [
                "the optimization",
                "this optimization",
                "the model",
                "could return",
                "even though",
                "described above",
                "because it checks",
            ]
            is_english = any(
                indicator in stripped.lower() for indicator in english_indicators
            )
            if (
                is_english
                and "=" not in line
                and "def " not in line
                and "class " not in line
            ):
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def ensure_imports(code: str, framework: str = "TensorFlow") -> str:
    lines = code.split("\n")
    _fw = framework.lower()

    if _fw == "tensorflow":
        uses_tf = bool(re.search(r"\btf[\.\s]", code))
        uses_np = bool(re.search(r"\bnp[\.\s]", code))

        has_tf_import = re.search(r"import\s+tensorflow\s+as\s+tf", code, re.IGNORECASE)
        has_np_import = re.search(r"import\s+numpy\s+as\s+np", code, re.IGNORECASE)

        imports_to_add = []
        if uses_tf and not has_tf_import:
            imports_to_add.append("import tensorflow as tf")
        if uses_np and not has_np_import:
            imports_to_add.append("import numpy as np")
    elif _fw == "pytorch":
        uses_torch = bool(re.search(r"\btorch[\.\s]", code))
        uses_nn = bool(re.search(r"\bnn[\.\s]", code))
        uses_np = bool(re.search(r"\bnp[\.\s]", code))

        has_torch_import = re.search(r"import\s+torch\b", code, re.IGNORECASE)
        has_nn_import = re.search(r"import\s+torch\.nn\s+as\s+nn", code, re.IGNORECASE)
        has_np_import = re.search(r"import\s+numpy\s+as\s+np", code, re.IGNORECASE)

        imports_to_add = []
        if uses_torch and not has_torch_import:
            imports_to_add.append("import torch")
        if uses_nn and not has_nn_import:
            imports_to_add.append("import torch.nn as nn")
        if uses_np and not has_np_import:
            imports_to_add.append("import numpy as np")
    else:
        uses_tf = bool(re.search(r"\btf[\.\s]", code))
        uses_np = bool(re.search(r"\bnp[\.\s]", code))

        has_tf_import = re.search(r"import\s+tensorflow\s+as\s+tf", code, re.IGNORECASE)
        has_np_import = re.search(r"import\s+numpy\s+as\s+np", code, re.IGNORECASE)

        imports_to_add = []
        if uses_tf and not has_tf_import:
            imports_to_add.append("import tensorflow as tf")
        if uses_np and not has_np_import:
            imports_to_add.append("import numpy as np")

    if imports_to_add:
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                insert_idx = i
                break

        for imp in reversed(imports_to_add):
            lines.insert(insert_idx, imp)

        code = "\n".join(lines)

    return code


def refine_generated_code(
    generated_text: str,
    framework: str = "TensorFlow",
    parser: Optional[CodeParser] = None,
) -> str:
    parsed_code = parse_generated_code(generated_text)

    code_with_imports = ensure_imports(parsed_code, framework=framework)

    p = parser or _default_parser
    processed_code = p.process_code(code_with_imports)

    return processed_code
