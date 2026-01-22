"""
Pydantic models for StarCoder generation configuration.

These are schema-only definitions with no default values.
All actual values should come from TOML configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class TestExecutionTask:
    """Task information for parallel test execution."""
    test_file: Path
    opt_name: str
    iteration: int
    sample_idx: int
    timeout: int = 7


@dataclass
class TestExecutionResult:
    """Result from parallel test execution including test metadata."""
    task: TestExecutionTask
    execution_result: Optional['ExecutionResult']
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether the test execution succeeded (no error)."""
        return self.error is None and self.execution_result is not None


class PathsConfig(BaseModel):
    prompt_dir: str = Field(description="Directory containing prompt files")
    output_dir: str = Field(description="Directory for generated outputs")
    hf_home: str = Field(description="HuggingFace home directory")
    hf_cache: Optional[str] = Field(default=None, description="HuggingFace cache directory")
    log_file: str = Field(description="Log file path")
    test_output_root: Optional[str] = Field(default=None, description="Root directory for generated tests per optimization")
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
    parallel_test_workers: Optional[int] = Field(default=None, description="Number of parallel workers for test execution (None = auto-detect CPU count)")
    parallel_optimizations: int = Field(default=1, description="Number of optimizations to process in parallel (1 = sequential, higher values enable parallelism)")


class OraclesConfig(BaseModel):
    float_rtol: float = Field(default=1e-5, description="Relative tolerance for float comparison")
    float_atol: float = Field(default=1e-8, description="Absolute tolerance for float comparison")


class StoppingConfig(BaseModel):
    eof_strings: List[str] = Field(description="End-of-text strings that stop generation")


class PassAliasesConfig(BaseModel):
    """Mapping from CamelCase optimization names to pass name aliases from instrumentation."""
    pass_name_aliases: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Mapping from optimization names to their pass log name aliases"
    )


class GeneratorConfig(BaseModel):
    paths: PathsConfig
    model: ModelConfig
    generation: GenerationConfig
    stopping: StoppingConfig
    oracles: Optional[OraclesConfig] = Field(default=None, description="Oracle configuration")
    pass_name_aliases: Optional[PassAliasesConfig] = Field(default=None, description="Pass name aliases configuration")

    @classmethod
    def from_toml(cls, toml_data: dict) -> "GeneratorConfig":
        oracles_data = toml_data.get("oracles", {})
        if oracles_data:
            oracles = OraclesConfig(**oracles_data)
        else:
            oracles = OraclesConfig()
        
        pass_aliases_data = toml_data.get("pass_name_aliases", {})
        if pass_aliases_data:
            # The TOML table is already a dict mapping optimization names to alias lists
            pass_aliases = PassAliasesConfig(pass_name_aliases=pass_aliases_data)
        else:
            pass_aliases = PassAliasesConfig()
        
        return cls(
            paths=PathsConfig(**toml_data.get("paths", {})),
            model=ModelConfig(**toml_data.get("model", {})),
            generation=GenerationConfig(**toml_data.get("generation", {})),
            stopping=StoppingConfig(**toml_data.get("stopping", {})),
            oracles=oracles,
            pass_name_aliases=pass_aliases,
        )

