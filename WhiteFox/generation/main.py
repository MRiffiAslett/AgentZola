"""
Accelerate the generation of StarCoder programs using VLLM.

This module provides a generic interface for generating test programs
using VLLM with configuration loaded from TOML files.
The configuration file must be provided via --config argument.
"""

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

from pprint import pprint
from vllm import LLM, SamplingParams

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from AgentZola.WhiteFox.models.generation import GeneratorConfig

# TOML parsing: try tomllib (Python 3.11+), then tomli, then toml
try:
    import tomllib  # Python 3.11+
    TOML_LOAD = tomllib.load
except ImportError:
    try:
        import tomli
        TOML_LOAD = tomli.load
    except ImportError:
        try:
            import toml
            TOML_LOAD = toml.load
        except ImportError:
            raise ImportError(
                "No TOML parser found. Please install one of: tomli, toml, or use Python 3.11+"
            )


class StarCoderGenerator:
    """Generator for StarCoder programs using VLLM."""
    
    def __init__(self, config: GeneratorConfig):
        """
        Initialize the generator with configuration.
        
        Args:
            config: GeneratorConfig instance with all configuration parameters.
        """
        self.config = config
        self._setup_environment()
        self._setup_logging()
        self.llm = self._initialize_llm()
        
    @classmethod
    def from_config_file(cls, config_path: Path) -> "StarCoderGenerator":
        """
        Create generator from TOML configuration file.
        
        Args:
            config_path: Path to TOML configuration file.
            
        Returns:
            StarCoderGenerator instance.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file is invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load TOML file
        try:
            with open(config_path, "rb") as f:
                toml_data = TOML_LOAD(f)
        except (TypeError, AttributeError):
            # Fall back to text mode for older toml library
            with open(config_path, "r") as f:
                toml_data = TOML_LOAD(f)
        
        # Create config from TOML data
        try:
            config = GeneratorConfig.from_toml(toml_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration file {config_path}: {e}") from e
        
        return cls(config)
    
    def _setup_environment(self) -> None:
        """Set up environment variables for HuggingFace."""
        if self.config.paths.hf_home:
            os.environ["HF_HOME"] = os.environ.get("HF_HOME", self.config.paths.hf_home)
        
        if self.config.paths.hf_cache:
            os.environ["HF_CACHE"] = self.config.paths.hf_cache
    
    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            filename=self.config.paths.log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_llm(self) -> LLM:
        """Initialize the VLLM model."""
        hf_cache_dir = (
            self.config.paths.hf_cache 
            or os.environ.get("HF_CACHE") 
            or os.environ.get("HF_HOME", "~/.cache/huggingface")
        )
        
        return LLM(
            model=self.config.model.name,
            dtype=self.config.model.dtype,
            download_dir=hf_cache_dir,
            max_model_len=self.config.model.max_model_len,
            gpu_memory_utilization=self.config.model.gpu_memory_utilization,
            swap_space=self.config.model.swap_space,
        )
    
    def _load_prompts(self, prompt_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Load prompts from directory.
        
        Args:
            prompt_dir: Directory containing prompt files.
            
        Returns:
            Tuple of (prompts, filenames) lists.
        """
        prompts = []
        filenames = []
        
        for prompt_file in sorted(prompt_dir.glob("*.txt")):
            prompts.append(prompt_file.read_text())
            filenames.append(prompt_file.stem)
        
        self.logger.info(f"Loaded {len(prompts)} prompts from {prompt_dir}")
        print(f"Number of prompts: {len(prompts)}")
        
        return prompts, filenames
    
    def _create_sampling_params(self, num_samples: int) -> SamplingParams:
        """Create sampling parameters for generation."""
        return SamplingParams(
            n=num_samples,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            max_tokens=self.config.generation.max_tokens,
            stop=self.config.stopping.eof_strings,
            seed=random.randint(0, 10000)
        )
    
    def _save_generated_outputs(
        self,
        outputs: List,
        filenames: List[str],
        prompt_start_idx: int,
        sample_start_idx: int,
        output_dir: Path
    ) -> None:
        """Save generated outputs to files."""
        for i, output in enumerate(outputs):
            filename = filenames[prompt_start_idx + i]
            output_file_dir = output_dir / filename
            output_file_dir.mkdir(exist_ok=True, parents=True)
            
            for r, text_output in enumerate(output.outputs):
                generated_text = text_output.text
                output_file = output_file_dir / f"{filename}-{sample_start_idx + r}.py"
                output_file.write_text(generated_text)
    
    def generate(
        self, 
        prompt_dir: Optional[Path] = None, 
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Generate test programs from prompts.
        
        Args:
            prompt_dir: Directory containing prompts. If None, uses config value.
            output_dir: Directory for outputs. If None, uses config value.
        """
        prompt_dir = Path(prompt_dir or self.config.paths.prompt_dir)
        output_dir = Path(output_dir or self.config.paths.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        prompts, filenames = self._load_prompts(prompt_dir)        
        num_samples = self.config.generation.num_samples
        split_size = self.config.generation.split_size
        unit_num = self.config.generation.unit_num
        
        # Process prompts in batches
        for prompt_batch_start in range(0, len(prompts), split_size):
            prompt_batch_end = min(prompt_batch_start + split_size, len(prompts))
            prompt_batch = prompts[prompt_batch_start:prompt_batch_end]
            
            # Generate samples in units
            for sample_batch_start in range(0, num_samples, unit_num):
                cur_num = min(unit_num, num_samples - sample_batch_start)
                
                start_time = time.time()
                sampling_params = self._create_sampling_params(cur_num)
                
                outputs = self.llm.generate(prompt_batch, sampling_params)
                elapsed_time = time.time() - start_time
                
                self.logger.info(
                    f"Generated {cur_num} samples for {len(prompt_batch)} prompts "
                    f"in {elapsed_time:.2f} seconds"
                )
                
                self._save_generated_outputs(
                    outputs,
                    filenames,
                    prompt_batch_start,
                    sample_batch_start,
                    output_dir
                )
                
                # Save timing log
                timing_log = output_dir / f"generated-{prompt_batch_start}-{sample_batch_start}-time.log"
                timing_log.write_text(str(elapsed_time))


def main():
    """Main entry point for the generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate StarCoder programs using VLLM with TOML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file (required)"
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help="Override prompt directory from config"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    pprint(vars(args))
    
    # Initialize generator from config file
    try:
        generator = StarCoderGenerator.from_config_file(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate
    generator.generate(
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
