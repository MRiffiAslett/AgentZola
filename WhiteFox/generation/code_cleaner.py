"""
Code cleaning utilities.

Handles ensuring imports and cleaning generated code from LLM outputs.
"""

import re


def ensure_imports(code: str) -> str:
    """
    Ensure required imports are present in the code.
    
    Adds:
    - import tensorflow as tf (if tf is used but not imported)
    - import numpy as np (if np is used but not imported)
    """
    lines = code.split('\n')
    
    uses_tf = re.search(r'\btf\.', code) or 'tf.' in code or 'tf ' in code
    uses_np = re.search(r'\bnp\.', code) or 'np.' in code or 'np ' in code
    
    has_tf_import = re.search(r'import\s+tensorflow\s+as\s+tf', code, re.IGNORECASE)
    has_np_import = re.search(r'import\s+numpy\s+as\s+np', code, re.IGNORECASE)
    
    imports_to_add = []
    if uses_tf and not has_tf_import:
        imports_to_add.append('import tensorflow as tf')
    if uses_np and not has_np_import:
        imports_to_add.append('import numpy as np')
    
    if imports_to_add:
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                insert_idx = i
                break
        
        for imp in reversed(imports_to_add):
            lines.insert(insert_idx, imp)
        
        code = '\n'.join(lines)
    
    return code


