"""
Code cleaning and extraction utilities.

Handles extraction of Python code from markdown, ensuring imports, and cleaning
generated code from LLM outputs.
"""

import re


def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown code blocks.
    
    Handles:
    - ```python ... ```
    - ``` ... ```
    - Stops at first non-code line after markdown block
    """
    pattern = r'```(?:python)?\s*\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        code = matches[-1].strip()
    else:
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if not in_code and (stripped.startswith(('import ', 'from ', 'class ', 'def ', '#')) or 
                               '=' in stripped or stripped.startswith('tf.') or stripped.startswith('np.')):
                in_code = True
            
            if in_code:
                if (len(stripped) > 100 and 
                    not any(keyword in stripped for keyword in ['import', 'def', 'class', '=', '(', ')', '[', ']', 'tf.', 'np.', '#']) and
                    not stripped.startswith('#')):
                    break
                if stripped and not stripped.startswith(('#', 'import', 'from', 'class', 'def', ' ', '\t')) and '=' not in stripped:
                    if any(word in stripped.lower() for word in ['this', 'should', 'model', 'optimization', 'tensorflow', 'xla', 'trigger']):
                        if not any(char in stripped for char in ['(', ')', '[', ']', '=', '.']):
                            break
                
                code_lines.append(line)
        
        code = '\n'.join(code_lines).strip()
    
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('```'):
            continue
        stripped = line.strip()
        if (stripped and 
            len(stripped) > 80 and 
            not stripped.startswith(('#', 'import', 'from', 'class', 'def')) and
            not any(char in stripped for char in ['(', ')', '[', ']', '=', '.']) and
            any(word in stripped.lower() for word in ['this', 'should', 'model', 'optimization', 'tensorflow', 'xla', 'trigger', 'prototype', 'require'])):
            break
        cleaned_lines.append(line)
    
    code = '\n'.join(cleaned_lines)
    
    return code.strip()


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


def clean_generated_code(raw_text: str) -> str:
    """
    Clean and prepare generated code for execution.
    
    Steps:
    1. Extract code from markdown blocks
    2. Ensure required imports
    3. Basic cleanup
    
    Returns cleaned code ready for execution.
    """
    code = extract_code_from_markdown(raw_text)
    
    code = ensure_imports(code)
    
    code = code.rstrip() + '\n'
    
    return code
