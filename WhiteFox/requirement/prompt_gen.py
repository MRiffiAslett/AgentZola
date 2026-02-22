import json
from pathlib import Path
from typing import Dict, List

EXEMPLAR_OPT_NAME = "ReshapeReshapeForwarding"


def strip_cpp_boilerplate(code: str) -> str:
    lines = code.split("\n")

    last_include_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#include"):
            last_include_idx = i

    if last_include_idx == -1:
        return code

    remaining = lines[last_include_idx + 1 :]

    while remaining and not remaining[0].strip():
        remaining.pop(0)

    return "\n".join(remaining)


class Optim:
    def get_opts(self):
        return self.opts.keys()

    def _src_code(self, code_paths: List[str], use_mini: bool = False) -> str:
        code = ""
        for code_path in code_paths:
            if use_mini:
                code_path = code_path.replace("-full", "-mini")

            file_path = Path(code_path)

            raw = file_path.read_text()
            code += strip_cpp_boilerplate(raw) + "\n"
        return code


class Src2TestTFLite(Optim):
    def __init__(self, file_path: str):
        super().__init__()
        self.opts = json.loads(Path(file_path).read_text())

    def _get_hint_code(self, hint: Dict, use_mini: bool = False) -> str:
        return self._src_code(hint["codes"], use_mini)


class Src2NLTFLite(Src2TestTFLite):
    PLACEHOLDER_TFLITE_OPTIMIZATION_NAME = "PLACEHOLDER_TFLITE_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"

    def get_prompt(self, template: str, opt: str, use_mini: bool = False) -> str:
        opt_info = self.opts[opt]
        code = ""
        for hint in opt_info["hints"]:
            code += self._get_hint_code(hint, use_mini)

        code_formatted = f"```cpp\n{code.strip()}\n```"
        prompt = template.replace(self.PLACEHOLDER_TFLITE_OPTIMIZATION_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, code_formatted)
        return prompt


class Src2NLTFXLA(Src2NLTFLite):
    PLACEHOLDER_TFLITE_OPTIMIZATION_NAME = "PLACEHOLDER_TFXLA_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_TARGET_LINE = "PLACEHOLDER_TARGET_LINE"
    PLACEHOLDER_FUNC_NAME = "PLACEHOLDER_FUNC_NAME"

    def format_source_code(self, code: str) -> str:
        return f"```cpp\n{code.strip()}\n```"

    def get_prompt(self, template: str, opt: str, use_mini: bool = False) -> str:
        opt_info = self.opts[opt]
        code = ""
        func_name = ""
        target_line = ""

        for hint in opt_info["hints"]:
            code += self._get_hint_code(hint, use_mini)
            func_name = hint.get("func", func_name)
            target_line = hint.get("target_line", target_line)

        if target_line and (target_line not in code):
            print(
                f"[WARNING] {opt} target line '{target_line}' does not exist in code."
            )

        prompt = template.replace(self.PLACEHOLDER_TFLITE_OPTIMIZATION_NAME, opt)
        prompt = prompt.replace(
            self.PLACEHOLDER_SRC_CODE, self.format_source_code(code)
        )
        prompt = prompt.replace(self.PLACEHOLDER_TARGET_LINE, target_line)
        prompt = prompt.replace(self.PLACEHOLDER_FUNC_NAME, func_name)
        return prompt


def _build_exemplar_block(
    optim: Src2NLTFXLA,
    template: str,
    exemplar_description: str,
    use_mini: bool = False,
) -> str:
    exemplar_block = optim.get_prompt(template, EXEMPLAR_OPT_NAME, use_mini=use_mini)
    exemplar_block = exemplar_block.rstrip("\n") + "\n" + exemplar_description
    return exemplar_block


def generate_requirement_prompts(
    optpath: str,
    template_path: str,
    outdir: str,
    use_mini: bool = False,
    fallback_dir: str = None,
    exemplar_description_path: str = None,
) -> tuple[Dict[str, str], set]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)

    template = Path(template_path).read_text()

    if fallback_dir is None:
        optpath_parent = Path(optpath).parent
        fallback_dir = str(optpath_parent / "whitefox_original" / "req")

    fallback_path = Path(fallback_dir) if fallback_dir else None

    if exemplar_description_path is None:
        exemplar_description_path = str(
            Path(optpath).parent / "exemplar_description.txt"
        )
    exemplar_description = Path(exemplar_description_path).read_text().rstrip("\n")

    optim = Src2NLTFXLA(optpath)

    try:
        exemplar_block = _build_exemplar_block(
            optim, template, exemplar_description, use_mini=use_mini
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot build few-shot exemplar for '{EXEMPLAR_OPT_NAME}': {e}\n"
            "Ensure the source file exists in source-code-data/."
        ) from e

    prompts = {}
    skipped = []
    used_fallback = []

    for opt in optim.get_opts():
        try:
            target_block = optim.get_prompt(template, opt, use_mini=use_mini)

            stacked_prompt = exemplar_block + "\n\n" + target_block

            prompts[opt] = stacked_prompt

            output_file = outdir_path / f"{opt}.txt"
            output_file.write_text(stacked_prompt)
            print(f"Generated prompt for {opt} -> {output_file}")
        except FileNotFoundError:
            fallback_file = fallback_path / f"{opt}.txt" if fallback_path else None

            prompts[opt] = "[FALLBACK - No prompt needed]"
            print(
                f"[FALLBACK] {opt} -> Will use pre-existing description (skip prompt & GPT)"
            )
            used_fallback.append(opt)

    actual_generated = len(prompts) - len(used_fallback)

    return prompts, set(used_fallback)
