"""
Pydantic models for StarCoder generation configuration.

These are schema-only definitions with no default values.
All actual values should come from TOML configuration files.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    prompt_dir: str = Field(description="Directory containing prompt files")
    output_dir: str = Field(description="Directory for generated outputs")
    hf_home: str = Field(description="HuggingFace home directory")
    hf_cache: Optional[str] = Field(default=None, description="HuggingFace cache directory")
    log_file: str = Field(description="Log file path")
    test_output_root: Optional[str] = Field(default=None, description="Root directory for generated tests per optimization")
    logs_root: Optional[str] = Field(default=None, description="Root directory for execution logs")
    bandit_state_file: Optional[str] = Field(default=None, description="Path to JSON/YAML file for WhiteFoxState persistence")
    bug_reports_dir: Optional[str] = Field(default=None, description="Directory for bug reports")


class ModelConfig(BaseModel):
    name: str = Field(description="Model identifier for HuggingFace")
    dtype: str = Field(description="Model data type")
    max_model_len: int = Field(description="Maximum model length")
    gpu_memory_utilization: float = Field(ge=0.0, le=1.0, description="GPU memory utilization (0.0 to 1.0)")
    swap_space: int = Field(description="Swap space in GB")


class GenerationConfig(BaseModel):
    num_samples: int = Field(gt=0, description="Number of samples to generate per prompt")
    max_tokens: int = Field(gt=0, description="Maximum tokens to generate")
    temperature: float = Field(ge=0.0, description="Temperature for sampling")
    top_p: float = Field(ge=0.0, le=1.0, description="Top-p for sampling")
    split_size: int = Field(gt=0, description="Batch size for processing prompts")
    unit_num: int = Field(gt=0, description="Unit batch size for generation")
    optimizations_dir: Optional[str] = Field(default=None, description="Path to requirement prompt directory")
    optimizations: Optional[List[str]] = Field(default=None, description="Hardcoded list of optimization names to target")
    tests_per_optimization: int = Field(default=1000, description="Total tests to generate per optimization")
    tests_per_iteration: int = Field(default=10, description="Tests to generate per iteration")
    max_iterations: int = Field(default=100, description="Maximum iterations per optimization")
    examples_per_prompt: int = Field(default=3, description="Number of examples to include in feedback prompt (N in Thompson Sampling)")


class OraclesConfig(BaseModel):
    float_rtol: float = Field(default=1e-5, description="Relative tolerance for float comparison")
    float_atol: float = Field(default=1e-8, description="Absolute tolerance for float comparison")


class StoppingConfig(BaseModel):
    eof_strings: List[str] = Field(description="End-of-text strings that stop generation")


class GeneratorConfig(BaseModel):
    paths: PathsConfig
    model: ModelConfig
    generation: GenerationConfig
    stopping: StoppingConfig
    oracles: Optional[OraclesConfig] = Field(default=None, description="Oracle configuration")

    @classmethod
    def from_toml(cls, toml_data: dict) -> "GeneratorConfig":
        oracles_data = toml_data.get("oracles", {})
        if oracles_data:
            oracles = OraclesConfig(**oracles_data)
        else:
            oracles = OraclesConfig()
        
        return cls(
            paths=PathsConfig(**toml_data.get("paths", {})),
            model=ModelConfig(**toml_data.get("model", {})),
            generation=GenerationConfig(**toml_data.get("generation", {})),
            stopping=StoppingConfig(**toml_data.get("stopping", {})),
            oracles=oracles,
        )

