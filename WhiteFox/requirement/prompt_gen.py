

import json
from pathlib import Path
from typing import Dict, List


class Optim:
    
    def get_opts(self):
        return self.opts.keys()

    def _src_code(self, code_paths: List[str], use_mini: bool = False) -> str:
        
        code = ""
        for code_path in code_paths:
            if use_mini:
                code_path = code_path.replace("-full", "-mini")
            
            file_path = Path(code_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Source file not found: {code_path}")
            
            code += file_path.read_text() + "\n"
        return code


class Src2TestTFLite(Optim):
    
    def __init__(self, file_path: str):
        super().__init__()
        self.opts = json.loads(Path(file_path).read_text())

    def _get_hint_code(self, hint: Dict, use_mini: bool = False) -> str:
        """Extract hint code from hint dictionary."""
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
        """Format source code in markdown code block."""
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
            print(f"[WARNING] {opt} target line '{target_line}' does not exist in code.")

        prompt = template.replace(self.PLACEHOLDER_TFLITE_OPTIMIZATION_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, self.format_source_code(code))
        prompt = prompt.replace(self.PLACEHOLDER_TARGET_LINE, target_line)
        prompt = prompt.replace(self.PLACEHOLDER_FUNC_NAME, func_name)
        return prompt


def generate_requirement_prompts(
    optpath: str,
    template_path: str,
    outdir: str,
    use_mini: bool = False,
    fallback_dir: str = None
) -> tuple[Dict[str, str], set]:
    
    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)
    
    template = Path(template_path).read_text()
    
    # Set default fallback directory if not provided
    if fallback_dir is None:
        # Default to ../xilo_xla/artifacts/Prompts/req relative to optpath
        optpath_parent = Path(optpath).parent
        fallback_dir = str(optpath_parent / "Prompts" / "req")
    
    fallback_path = Path(fallback_dir) if fallback_dir else None
    
    optim = Src2NLTFXLA(optpath)
    prompts = {}
    skipped = []
    used_fallback = []
    
    for opt in optim.get_opts():
        try:
            prompt = optim.get_prompt(template, opt, use_mini=use_mini)
            prompts[opt] = prompt
            
            output_file = outdir_path / f"{opt}.txt"
            output_file.write_text(prompt)
            print(f"Generated prompt for {opt} -> {output_file}")
        except FileNotFoundError as e:
            fallback_file = fallback_path / f"{opt}.txt" if fallback_path else None
            
            if fallback_file and fallback_file.exists():
                
                prompts[opt] = "[FALLBACK - No prompt needed]"
                print(f"[FALLBACK] {opt} -> Will use pre-existing description (skip prompt & GPT)")
                used_fallback.append(opt)
            else:
                print(f"[SKIPPED] {opt}: {e}")
                skipped.append(opt)
                continue
    
    actual_generated = len(prompts) - len(used_fallback)
    
    return prompts, set(used_fallback)


