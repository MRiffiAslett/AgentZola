"""
Code processing for WhiteFox generated tests.
Transforms generated TensorFlow model code into executable test format.

This module implements the AST-based code transformation from the original WhiteFox,
including:
- Parsing class definitions
- Extracting tensor initializations
- Creating input_data variable
- Applying preprocessing transformations
"""

import ast
import astunparse
from typing import List, Tuple
import tensorflow as tf


class MultilineAssignTransformer(ast.NodeTransformer):
    """Transform multiline assignments into separate single assignments.
    
    Example: a, b = x, y  =>  a = x; b = y
    """
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [ast.Assign(targets=[t], value=v) 
                       for t, v in zip(node.targets[0].elts, node.value.elts)]
        return node


class TFAssignRemover(ast.NodeTransformer):
    """Remove TensorFlow global config assignments.
    
    Example: tf.config.x = y  =>  (removed)
    """
    def visit_Assign(self, node):
        if any(self.is_tf_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node)

    def is_tf_attribute(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'tf':
                return True
            return self.is_tf_attribute(node.value)
        return False


class TFCodeParserJIT:
    """Parse and transform TensorFlow test code into executable format."""
    
    def __init__(self) -> None:
        pass

    def split_func_tensor(self, code: str) -> Tuple[str, str, List[str], str]:
        """Split code into class definition, tensor inits, and input data.
        
        Returns:
            Tuple of (class_code, class_name, tensors, tensor_inits)
        """
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
        tensor_inits = ''
        
        # Extract class definition
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_code += astunparse.unparse(node) + "\n\n"
                class_name = node.name

                # Get __init__ arguments
                try:
                    init_method = next(
                        n for n in ast.walk(node) 
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    )
                    class_init_args = [arg.arg for arg in init_method.args.args[1:]]
                    defaults = init_method.args.defaults
                    class_init_required_args = class_init_args[:len(class_init_args) - len(defaults)]
                except Exception:
                    pass

                # Get call method arguments
                try:
                    forward_method = next(
                        n for n in ast.walk(node) 
                        if isinstance(n, ast.FunctionDef) and n.name == "call"
                    )
                    class_forward_args = [arg.arg for arg in forward_method.args.args[1:]]
                    defaults = forward_method.args.defaults
                    class_forward_required_args = class_forward_args[:len(class_forward_args) - len(defaults)]
                except Exception:
                    pass

            elif isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    # Check if this is class initialization
                    if isinstance(value.func, ast.Name) and value.func.id == class_name:
                        if len(value.args) >= len(class_init_required_args) and len(value.args) <= len(class_init_args):
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
                        # Collect dependencies
                        for arg in ast.walk(value):
                            if isinstance(arg, ast.Name):
                                dep_code = self.find_name_in_tree(tree, arg.id)
                                if dep_code:
                                    init_code = dep_code + '\n' + init_code
                            elif isinstance(arg, ast.Starred):
                                if isinstance(arg.value, ast.Name):
                                    dep_code = self.find_name_in_tree(tree, arg.value.id)
                                    if dep_code:
                                        init_code = dep_code + '\n' + init_code
                        
                        # Test if this creates a tensor
                        try:
                            exec(init_code)
                            if isinstance(eval(tgt), tf.Tensor):
                                tensors.append(tgt)
                                tensor_inits += init_code + '\n'
                            elif tgt in class_forward_args:
                                tensors.append(tgt)
                                tensor_inits += init_code + '\n'
                        except Exception:
                            pass

        # Build class initialization code
        class_init_args_code = ""
        for arg_name in class_init_required_args:
            class_init_args_code += self.find_name_in_tree(tree, arg_name, use_default=True) + "\n"
        
        if class_init_code != "":
            class_init_code = class_init_args_code + class_init_code
        else:
            class_init_code = class_init_args_code
            class_init_code += f"\nm = {class_name}({', '.join(class_init_required_args)})\n"
        
        class_code += "\n" + class_init_code

        # Ensure we have enough tensors for forward pass
        if len(tensors) < len(class_forward_args):
            diff = len(class_forward_args) - len(tensors)
            for arg_name in class_forward_required_args:
                if arg_name not in tensors:
                    tensors.append(arg_name)
                    tensor_inits += f"{arg_name} = 1\n"
                    diff -= 1
                    if diff == 0:
                        break

        if len(tensors) > len(class_forward_args):
            tensors = tensors[:len(class_forward_args)]

        return class_code, class_name, tensors, tensor_inits
    
    @staticmethod
    def preprocessing(code: str) -> str:
        """Apply preprocessing transformations to code."""
        # Replace tabs with spaces
        code = code.replace("\t", "    ")
        
        # Remove assert and import statements
        new_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert") or stripped.startswith("import"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        # Apply AST transformations
        try:
            tree = ast.parse(code)
            
            # Transform multiline assignments
            transformer = MultilineAssignTransformer()
            new_tree = transformer.visit(tree)

            # Remove TF global assignments
            tf_assign_remover = TFAssignRemover()
            new_tree = tf_assign_remover.visit(new_tree)
            
            code = astunparse.unparse(new_tree)
        except Exception:
            # If AST parsing fails, return code as-is
            pass
            
        return code

    @staticmethod
    def find_name_in_tree(tree, arg_name: str, use_default: bool = False) -> str:
        """Find assignment of a variable name in the AST."""
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == arg_name:
                        return astunparse.unparse(node)
        
        if arg_name == "batch_size":
            return f"{arg_name} = 1"
        
        if use_default:
            return f"{arg_name} = 1"
        else:
            return ""


def process_code(code: str) -> str:
    """Process generated TensorFlow model code into executable test format.
    
    This is the main entry point for code processing. It:
    1. Replaces __call__ with call
    2. Parses the class and extracts components
    3. Creates input_data variable
    4. Returns executable test code
    
    Args:
        code: Raw generated code string
        
    Returns:
        Processed executable code with input_data variable
    """
    code = code.replace('__call__', 'call')
    parser = TFCodeParserJIT()
    
    try:
        class_code, class_name, tensors, tensor_inits = parser.split_func_tensor(code)
        code = class_code + "\n" + tensor_inits + "\n" + f"input_data = [{', '.join(tensors)}]\n"
    except Exception:
        # If processing fails, return code as-is
        # This ensures we don't break on edge cases
        pass
    
    return code

