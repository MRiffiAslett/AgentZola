import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


class GPTClient:

    def __init__(
        self,
        api_key_file: str = "openai.key",
        model: str = "gpt-4",
        temperature: float = 1.0,
        max_tokens: int = 512,
        timeout: int = 300,
        system_message: str = "",
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
        self.system_message = system_message

    def process_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_requirement(
        self,
        prompt: str,
        n_samples: int = 1,
        max_retries: int = 3,
        retry_delay: int = 10,
    ) -> Tuple[List[str], Dict]:
        retry_count = 0

        while retry_count < max_retries:
            t_start = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                n=n_samples,
                timeout=self.timeout,
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
                "raw_response": response.model_dump(),
            }

            return descriptions, metadata

    def batch_generate_requirements(
        self,
        prompts: Dict[str, str],
        output_dir: Path,
        instruction_template: str = (
            "### Please generate one valid model that satisfies requirements below.\n\n"
            "# Description\n"
        ),
        n_samples: int = 1,
        skip_existing: bool = True,
        fallback_optimizations: set = None,
        fallback_dir: str = None,
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
                continue

            instruction = instruction_template.format(opt_name=opt_name)

            if opt_name in fallback_optimizations:
                fallback_file = fallback_path / f"{opt_name}.txt"
                description = fallback_file.read_text()

                full_prompt = instruction + description

                with open(output_file, "w") as f:
                    f.write(full_prompt)

                results[opt_name] = {
                    "status": "success",
                    "descriptions": [full_prompt],
                    "metadata": {
                        "source": "fallback",
                        "fallback_file": str(fallback_file),
                    },
                }
                continue

            descriptions, metadata = self.generate_requirement(
                prompt=prompt, n_samples=n_samples
            )

            description = descriptions[0] if descriptions else ""

            full_prompt = instruction + description

            with open(output_file, "w") as f:
                f.write(full_prompt)

            results[opt_name] = {
                "status": "success",
                "descriptions": [full_prompt],
                "metadata": metadata,
            }

            print(
                f"  ✓ Generated {len(descriptions)} description(s) in {metadata['generation_time']:.2f}s"
            )

        results_file = output_dir / "generation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results
