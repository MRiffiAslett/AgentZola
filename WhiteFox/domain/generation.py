from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class TestExecutionTask:
    test_file: Path
    opt_name: str
    iteration: int
    sample_idx: int
    timeout: int = 7


@dataclass
class TestExecutionResult:
    task: TestExecutionTask
    execution_result: Optional["ExecutionResult"]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.execution_result is not None


class PathsConfig(BaseModel):
    prompt_dir: str
    output_dir: str
    hf_home: str
    hf_cache: Optional[str] = None
    log_file: str
    test_output_root: Optional[str] = None
    bandit_state_file: Optional[str] = None
    bug_reports_dir: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float = Field(ge=0.0, le=1.0)
    swap_space: int


class GenerationConfig(BaseModel):
    num_samples: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(ge=0.0, le=1.0)
    split_size: int = Field(gt=0)
    unit_num: int = Field(gt=0)
    optimizations_dir: Optional[str] = None
    optimizations: Optional[List[str]] = None
    tests_per_optimization: int = 1000
    tests_per_iteration: int = 10
    max_iterations: int = 100
    examples_per_prompt: int = 3
    parallel_test_workers: Optional[int] = None
    parallel_optimizations: int = 1
    test_timeout: int = 60


class OraclesConfig(BaseModel):
    float_rtol: float = 1e-5
    float_atol: float = 1e-8
    allowed_errors: List[str] = []


class StoppingConfig(BaseModel):
    eof_strings: List[str]


class PassAliasesConfig(BaseModel):
    pass_name_aliases: Optional[Dict[str, List[str]]] = None


class PromptsConfig(BaseModel):
    prompt_style: str = "paper"
    seed_file: str = ""
    feedback_instruction: str = ""


class SUTConfig(BaseModel):
    name: str  # e.g. "xla", "inductor", "tflite"
    framework: str  # e.g. "tensorflow", "pytorch"


class GeneratorConfig(BaseModel):
    paths: PathsConfig
    model: ModelConfig
    generation: GenerationConfig
    stopping: StoppingConfig
    sut: SUTConfig
    oracles: Optional[OraclesConfig] = None
    pass_name_aliases: Optional[PassAliasesConfig] = None
    prompts: Optional[PromptsConfig] = None

    @classmethod
    def from_toml(cls, toml_data: dict) -> "GeneratorConfig":
        oracles_data = toml_data.get("oracles", {})
        if oracles_data:
            oracles = OraclesConfig(**oracles_data)
        else:
            oracles = OraclesConfig()

        pass_aliases_data = toml_data.get("pass_name_aliases", {})
        if pass_aliases_data:
            pass_aliases = PassAliasesConfig(pass_name_aliases=pass_aliases_data)
        else:
            pass_aliases = PassAliasesConfig()

        prompts_data = toml_data.get("prompts", {})
        if prompts_data:
            prompts = PromptsConfig(**prompts_data)
        else:
            prompts = PromptsConfig()

        sut_data = toml_data.get("sut", {})
        if not sut_data:
            raise ValueError(
                "Config is missing a [sut] section with 'name' and 'framework'."
            )
        sut = SUTConfig(**sut_data)

        return cls(
            paths=PathsConfig(**toml_data.get("paths", {})),
            model=ModelConfig(**toml_data.get("model", {})),
            generation=GenerationConfig(**toml_data.get("generation", {})),
            stopping=StoppingConfig(**toml_data.get("stopping", {})),
            sut=sut,
            oracles=oracles,
            pass_name_aliases=pass_aliases,
            prompts=prompts,
        )
