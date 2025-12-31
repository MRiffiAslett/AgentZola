"""
Code cleaning and extraction utilities.

Handles extraction of Python code from markdown, ensuring imports, and cleaning
generated code from LLM outputs.
"""

import re
from typing import Optional


def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown code blocks.
    
    Handles:
    - ```python ... ```
    - ``` ... ```
    - Stops at first non-code line after markdown block
    """
    # Remove markdown code blocks
    # Pattern: ```python ... ``` or ``` ... ```
    pattern = r'```(?:python)?\s*\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last (usually largest) code block
        code = matches[-1].strip()
    else:
        # No markdown blocks, find where code likely ends
        # Look for common patterns that indicate end of code
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            # Start collecting when we see Python-like code
            if not in_code and (stripped.startswith(('import ', 'from ', 'class ', 'def ', '#')) or 
                               '=' in stripped or stripped.startswith('tf.') or stripped.startswith('np.')):
                in_code = True
            
            if in_code:
                # Stop at explanatory text (long lines without Python syntax)
                if (len(stripped) > 100 and 
                    not any(keyword in stripped for keyword in ['import', 'def', 'class', '=', '(', ')', '[', ']', 'tf.', 'np.', '#']) and
                    not stripped.startswith('#')):
                    break
                # Stop at markdown-style explanations
                if stripped and not stripped.startswith(('#', 'import', 'from', 'class', 'def', ' ', '\t')) and '=' not in stripped:
                    # Check if it looks like explanatory text
                    if any(word in stripped.lower() for word in ['this', 'should', 'model', 'optimization', 'tensorflow', 'xla', 'trigger']):
                        if not any(char in stripped for char in ['(', ')', '[', ']', '=', '.']):
                            break
                
                code_lines.append(line)
        
        code = '\n'.join(code_lines).strip()
    
    # Remove any remaining markdown artifacts
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip markdown formatting
        if line.strip().startswith('```'):
            continue
        # Stop at explanatory text that's clearly not code
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
    
    # Check if tf is used
    uses_tf = re.search(r'\btf\.', code) or 'tf.' in code or 'tf ' in code
    # Check if np is used
    uses_np = re.search(r'\bnp\.', code) or 'np.' in code or 'np ' in code
    
    # Check existing imports
    has_tf_import = re.search(r'import\s+tensorflow\s+as\s+tf', code, re.IGNORECASE)
    has_np_import = re.search(r'import\s+numpy\s+as\s+np', code, re.IGNORECASE)
    
    imports_to_add = []
    if uses_tf and not has_tf_import:
        imports_to_add.append('import tensorflow as tf')
    if uses_np and not has_np_import:
        imports_to_add.append('import numpy as np')
    
    if imports_to_add:
        # Find the first non-comment, non-empty line or insert at the beginning
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                insert_idx = i
                break
        
        # Insert imports
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
    # Step 1: Extract from markdown
    code = extract_code_from_markdown(raw_text)
    
    # Step 2: Ensure imports
    code = ensure_imports(code)
    
    # Step 3: Final cleanup - remove trailing whitespace, ensure newline at end
    code = code.rstrip() + '\n'
    
    return code

