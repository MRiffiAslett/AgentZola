"""
Pydantic models for StarCoder generation configuration.

These are schema-only definitions with no default values.
All actual values should come from TOML configuration files.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for file paths."""
    prompt_dir: str = Field(description="Directory containing prompt files")
    output_dir: str = Field(description="Directory for generated outputs")
    hf_home: str = Field(description="HuggingFace home directory")
    hf_cache: Optional[str] = Field(default=None, description="HuggingFace cache directory")
    log_file: str = Field(description="Log file path")


class ModelConfig(BaseModel):
    """Configuration for the LLM model."""
    name: str = Field(description="Model identifier for HuggingFace")
    dtype: str = Field(description="Model data type")
    max_model_len: int = Field(description="Maximum model length")
    gpu_memory_utilization: float = Field(ge=0.0, le=1.0, description="GPU memory utilization (0.0 to 1.0)")
    swap_space: int = Field(description="Swap space in GB")


class GenerationConfig(BaseModel):
    """Configuration for text generation parameters."""
    num_samples: int = Field(gt=0, description="Number of samples to generate per prompt")
    max_tokens: int = Field(gt=0, description="Maximum tokens to generate")
    temperature: float = Field(ge=0.0, description="Temperature for sampling")
    top_p: float = Field(ge=0.0, le=1.0, description="Top-p for sampling")
    split_size: int = Field(gt=0, description="Batch size for processing prompts")
    unit_num: int = Field(gt=0, description="Unit batch size for generation")


class StoppingConfig(BaseModel):
    """Configuration for stopping criteria."""
    eof_strings: List[str] = Field(description="End-of-text strings that stop generation")


class GeneratorConfig(BaseModel):
    """Complete configuration for the generator."""
    paths: PathsConfig
    model: ModelConfig
    generation: GenerationConfig
    stopping: StoppingConfig

    @classmethod
    def from_toml(cls, toml_data: dict) -> "GeneratorConfig":
        """Create configuration from TOML dictionary."""
        return cls(
            paths=PathsConfig(**toml_data.get("paths", {})),
            model=ModelConfig(**toml_data.get("model", {})),
            generation=GenerationConfig(**toml_data.get("generation", {})),
            stopping=StoppingConfig(**toml_data.get("stopping", {})),
        )
