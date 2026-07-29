import ast
import os
import re
from typing import List, Tuple

import astunparse

from generation.code_processing.base import CodeParser

# Static pre-check mirroring harness/xla.py's wrapper-script guard against
# pathologically large literal tensor shapes (e.g. shape=[512,512,512,512,512]
# ~= 144 PB). That guard only protects the sandboxed test-execution subprocess
# (RLIMIT_AS + this same regex check there). split_func_tensor() below does
# its own exec()/eval() of fragments of the same raw generated code, but runs
# unsandboxed in the main long-lived process — a native crash there (observed:
# a SIGSEGV from an ~65 TB allocation attempt, job 266428 batch1) takes down
# the whole multi-hour run instead of just failing one test. Skip the exec/eval
# entirely for code matching this pattern; the sandboxed harness still handles
# the test safely and records a normal (not lost) result for it.
_TENSOR_SHAPE_RE = re.compile(
    r'(?:tf\.(?:zeros|ones|random\.(?:normal|uniform|truncated_normal)|fill|constant)'
    r'|np\.(?:zeros|ones|random\.(?:randn?|uniform|normal)|full|empty))'
    r'\s*\('
    r'[^)]*?'
    r'(?:shape\s*=\s*|size\s*=\s*)?'
    r'(?:\[|\()'
    r'([\d,\s]+)'
    r'(?:\]|\))',
)
_TENSOR_BARE_RE = re.compile(
    r'np\.random\.(?:randn?|uniform|normal|standard_normal)'
    r'\s*\(\s*'
    r'([\d,\s]+)'
    r'\s*\)',
)
_STATIC_ELEMENT_BYTES = 4
_MAX_STATIC_BYTES = (
    int(os.environ.get("WHITEFOX_TEST_MEM_LIMIT_GB", "8")) * 1024 ** 3
) // 2


def _estimate_tensor_bytes(code_str: str) -> int:
    total = 0
    for pat in (_TENSOR_SHAPE_RE, _TENSOR_BARE_RE):
        for m in pat.finditer(code_str):
            try:
                dims = [int(d.strip()) for d in m.group(1).split(",") if d.strip()]
                numel = 1
                for d in dims:
                    numel *= d
                total += numel * _STATIC_ELEMENT_BYTES
            except (ValueError, OverflowError):
                total += _MAX_STATIC_BYTES + 1
    return total


class _MultilineAssignTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [
                    ast.Assign(targets=[t], value=v)
                    for t, v in zip(node.targets[0].elts, node.value.elts)
                ]
        return node


class _TFAssignRemover(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(self._is_tf_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node)

    def _is_tf_attribute(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "tf":
                return True
            return self._is_tf_attribute(node.value)
        return False


class TensorFlowCodeParser(CodeParser):

    DEFAULT_TENSOR_VALUE = "tf.constant([1.0, 2.0, 3.0, 4.0])"

    def split_func_tensor(self, code: str) -> Tuple[str, str, List[str], str]:
        code = self.preprocessing(code)
        tree = ast.parse(code)

        class_init_args = []
        class_init_required_args = []
        class_init_code = ""
        class_code = ""
        class_name = ""
        class_forward_args = []
        class_forward_required_args = []
        tensors: List[str] = []
        tensor_inits = ""

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_code += astunparse.unparse(node) + "\n\n"
                class_name = node.name

                try:
                    init_method = next(
                        n
                        for n in ast.walk(node)
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    )
                    class_init_args = [arg.arg for arg in init_method.args.args[1:]]
                    defaults = init_method.args.defaults
                    class_init_required_args = class_init_args[
                        : len(class_init_args) - len(defaults)
                    ]
                except Exception:
                    pass

                try:
                    forward_method = next(
                        n
                        for n in ast.walk(node)
                        if isinstance(n, ast.FunctionDef) and n.name == "call"
                    )
                    class_forward_args = [
                        arg.arg for arg in forward_method.args.args[1:]
                    ]
                    defaults = forward_method.args.defaults
                    class_forward_required_args = class_forward_args[
                        : len(class_forward_args) - len(defaults)
                    ]
                except Exception:
                    pass

            elif isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    if isinstance(value.func, ast.Name) and value.func.id == class_name:
                        if len(value.args) >= len(class_init_required_args) and len(
                            value.args
                        ) <= len(class_init_args):
                            class_init_code = "m = " + astunparse.unparse(value) + "\n"
                        else:
                            class_init_code = ""
                        continue

                    try:
                        tgt = node.targets[0].id
                    except Exception:
                        continue

                    init_code = astunparse.unparse(node)
                    if tgt not in tensors:
                        for arg in ast.walk(value):
                            if isinstance(arg, ast.Name):
                                dep_code = self._find_name_in_tree(tree, arg.id)
                                if dep_code:
                                    init_code = dep_code + "\n" + init_code
                            elif isinstance(arg, ast.Starred):
                                if isinstance(arg.value, ast.Name):
                                    dep_code = self._find_name_in_tree(
                                        tree, arg.value.id
                                    )
                                    if dep_code:
                                        init_code = dep_code + "\n" + init_code

                    try:
                        import tensorflow as tf

                        exec(init_code)
                        if isinstance(eval(tgt), tf.Tensor):
                            tensors.append(tgt)
                            tensor_inits += init_code + "\n"
                        elif tgt in class_forward_args:
                            tensors.append(tgt)
                            tensor_inits += init_code + "\n"
                    except Exception:
                        pass

        class_init_args_code = ""
        for arg_name in class_init_required_args:
            class_init_args_code += (
                self._find_name_in_tree(tree, arg_name, use_default=True) + "\n"
            )

        if class_init_code != "":
            class_init_code = class_init_args_code + class_init_code
        else:
            class_init_code = class_init_args_code
            class_init_code += (
                f"\nm = {class_name}({', '.join(class_init_required_args)})\n"
            )

        class_code += "\n" + class_init_code

        if len(tensors) < len(class_forward_args):
            diff = len(class_forward_args) - len(tensors)
            for arg_name in class_forward_required_args:
                if arg_name not in tensors:
                    tensors.append(arg_name)
                    tensor_inits += f"{arg_name} = {self.DEFAULT_TENSOR_VALUE}\n"
                    diff -= 1
                    if diff == 0:
                        break

            while diff > 0:
                arg_name = f"arg_{len(tensors)}"
                tensors.append(arg_name)
                tensor_inits += f"{arg_name} = {self.DEFAULT_TENSOR_VALUE}\n"
                diff -= 1

        if len(tensors) > len(class_forward_args):
            tensors = tensors[: len(class_forward_args)]

        return class_code, class_name, tensors, tensor_inits

    def preprocessing(self, code: str) -> str:
        code = code.replace("\t", "    ")

        new_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert") or stripped.startswith("import"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        try:
            tree = ast.parse(code)

            transformer = _MultilineAssignTransformer()
            new_tree = transformer.visit(tree)

            tf_assign_remover = _TFAssignRemover()
            new_tree = tf_assign_remover.visit(new_tree)

            code = astunparse.unparse(new_tree)
        except Exception:
            pass

        return code

    def process_code(self, code: str) -> str:
        code = code.replace("__call__", "call")

        if _estimate_tensor_bytes(code) > _MAX_STATIC_BYTES:
            # split_func_tensor() below exec()s/eval()s fragments of this
            # code unsandboxed in the main process to detect tensor
            # variables — skip it entirely for a pathological shape literal
            # and fall back to unmodified code, same as the exception
            # fallback below. The sandboxed test-execution harness has its
            # own copy of this same check and will still handle this test
            # safely (and record a normal result for it).
            return code

        try:
            class_code, class_name, tensors, tensor_inits = self.split_func_tensor(code)
            code = (
                class_code
                + "\n"
                + tensor_inits
                + "\n"
                + f"input_data = [{', '.join(tensors)}]\n"
            )
        except Exception:
            pass

        return code

    @staticmethod
    def _find_name_in_tree(tree, arg_name: str, use_default: bool = False) -> str:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == arg_name:
                        return astunparse.unparse(node)

        if arg_name == "batch_size":
            return f"{arg_name} = 1"

        if use_default:
            return f"{arg_name} = {TensorFlowCodeParser.DEFAULT_TENSOR_VALUE}"
        else:
            return ""
