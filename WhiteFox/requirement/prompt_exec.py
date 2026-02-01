import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI


class GPTModel:
    """Interface for OpenAI GPT models."""
    
    def __init__(
        self,
        api_key_file: str = "openai.key",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 300,
        system_message: str = None
    ):
       
        api_key_path = Path(api_key_file)
        if not api_key_path.exists():
            raise FileNotFoundError(
                f"API key file not found: {api_key_file}\n"
                "Please create this file with your OpenAI API key."
            )
        
        self.client = OpenAI(api_key=api_key_path.read_text().strip())
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_message = system_message or (
            "You are an expert in TensorFlow XLA compiler optimizations. "
            "Your task is to analyze optimization pass source code and describe "
            "the characteristics of TensorFlow models that would trigger these optimizations. "
            "Be concise, clear, and use code examples to illustrate patterns."
        )
    
    def process_response(self, response_text: str) -> str:
       
        return response_text.strip()
    
    def generate_requirement(
        self,
        prompt: str,
        n_samples: int = 1,
        max_retries: int = 3,
        retry_delay: int = 10
    ) -> Tuple[List[str], Dict]:
       
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                t_start = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    n=n_samples,
                    timeout=self.timeout
                )
                
                generation_time = time.time() - t_start
                
                descriptions = [
                    self.process_response(choice.message.content)
                    for choice in response.choices
                ]
                
                metadata = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "n_samples": n_samples,
                    "generation_time": generation_time,
                    "raw_response": response.model_dump()
                }
                
                return descriptions, metadata
                
            except Exception as e:
                retry_count += 1
                print(f"[ERROR] API call failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to generate requirement after {max_retries} attempts"
                    ) from e
    
    def batch_generate_requirements(
        self,
        prompts: Dict[str, str],
        output_dir: Path,
        n_samples: int = 1,
        skip_existing: bool = True,
        fallback_optimizations: set = None,
        fallback_dir: str = None,
        prefix_text: str = None
    ) -> Dict[str, Dict]:
       
        if fallback_optimizations is None:
            fallback_optimizations = set()
        
        fallback_path = Path(fallback_dir) if fallback_dir else None
        output_dir.mkdir(exist_ok=True, parents=True)
        results = {}
        
        total = len(prompts)
        for idx, (opt_name, prompt) in enumerate(prompts.items(), 1):
            print("=" * 80)
            print(f"[{idx}/{total}] Processing: {opt_name}")
            print("=" * 80)
            
            output_file = output_dir / f"{opt_name}.txt"
            
            if skip_existing and output_file.exists():
                print(f"  ⏭️  Skipping {opt_name} (already exists)")
                continue
            
            if opt_name in fallback_optimizations:
                if fallback_path and (fallback_path / f"{opt_name}.txt").exists():
                    fallback_file = fallback_path / f"{opt_name}.txt"
                    description = fallback_file.read_text()
                    
                    if prefix_text:
                        description = prefix_text + "\n\n" + description
                    
                    with open(output_file, "w") as f:
                        f.write(description)
                    
                    print(f"  📋 Using pre-existing description (no GPT call)")
                    print(f"  ✓ Saved: {output_file.name}")
                    
                    results[opt_name] = {
                        "status": "success",
                        "descriptions": [description],
                        "metadata": {"source": "fallback", "fallback_file": str(fallback_file)}
                    }
                    continue
                else:
                    print(f"  ⚠️  Marked as fallback but file not found, skipping")
                    results[opt_name] = {
                        "status": "failed",
                        "error": "Fallback file not found"
                    }
                    continue
            
            try:
                
                descriptions, metadata = self.generate_requirement(
                    prompt=prompt,
                    n_samples=n_samples
                )
                
                description = descriptions[0] if descriptions else ""
     
                if prefix_text:
                    description = prefix_text + "\n\n" + description
                
                with open(output_file, "w") as f:
                    f.write(description)
                print(f"  ✓ Saved: {output_file.name}")
                
                results[opt_name] = {
                    "status": "success",
                    "descriptions": descriptions,
                    "metadata": metadata
                }
                
                print(f"  ✓ Generated {len(descriptions)} description(s) in {metadata['generation_time']:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Failed to process {opt_name}: {e}")
                results[opt_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        

        results_file = output_dir / "generation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
