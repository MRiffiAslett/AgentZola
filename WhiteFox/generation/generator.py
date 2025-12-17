
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, List

from vllm import LLM, SamplingParams

from models.generation import GeneratorConfig

from generation.spec import load_optimization_specs
from domain.bandit import (
    WhiteFoxState,
    OptimizationState,
)
from generation.bandit import (
    select_examples_thompson_sampling,
    update_bandit_after_generation,
)
from generation.prompts import build_base_prompt, build_feedback_prompt
from generation.harness import execute_test_in_subprocess
from generation.oracle import check_oracles

try:
    import tomllib
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
    
    def __init__(self, config: GeneratorConfig, config_file_path: Optional[Path] = None):
        self.config = config
        self.config_file_path = config_file_path
        self._setup_environment()
        self._setup_logging()
        self.llm = self._initialize_llm()
        
    @classmethod
    def from_config_file(cls, config_path: Path) -> "StarCoderGenerator":
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "rb") as f:
                toml_data = TOML_LOAD(f)
        except (TypeError, AttributeError):
            with open(config_path, "r") as f:
                toml_data = TOML_LOAD(f)
        
        try:
            config = GeneratorConfig.from_toml(toml_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration file {config_path}: {e}") from e
        
        return cls(config, config_file_path=config_path)
    
    def _setup_environment(self) -> None:
        if self.config.paths.hf_home:
            os.environ["HF_HOME"] = os.environ.get("HF_HOME", self.config.paths.hf_home)
        
        if self.config.paths.hf_cache:
            os.environ["HF_CACHE"] = self.config.paths.hf_cache
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            filename=self.config.paths.log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_llm(self) -> LLM:
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
    
    def _create_sampling_params(self, num_samples: int) -> SamplingParams:
        return SamplingParams(
            n=num_samples,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            max_tokens=self.config.generation.max_tokens,
            stop=self.config.stopping.eof_strings,
            seed=random.randint(0, 10000)
        )
    
    def _load_or_init_whitefox_state(self) -> WhiteFoxState:
        optimizations_dir = Path(self.config.generation.optimizations_dir)
        
        # Resolve relative paths relative to config file location or project root
        if not optimizations_dir.is_absolute():
            if self.config_file_path:
                config_dir = self.config_file_path.parent
                # If path starts with "WhiteFox/", resolve relative to project root
                # Otherwise, resolve relative to config file directory
                if str(optimizations_dir).startswith("WhiteFox/"):
                    # Find project root by going up from config file until we find
                    # a directory that contains "WhiteFox" as a subdirectory
                    current = config_dir
                    project_root = None
                    while current != current.parent:
                        if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                            project_root = current
                            break
                        current = current.parent
                    
                    if project_root:
                        optimizations_dir = (project_root / optimizations_dir).resolve()
                    else:
                        # Fallback: resolve relative to config file directory
                        optimizations_dir = (config_dir / optimizations_dir).resolve()
                else:
                    # Resolve relative to config file's directory
                    optimizations_dir = (config_dir / optimizations_dir).resolve()
            else:
                # Fallback: try to resolve relative to current working directory
                optimizations_dir = optimizations_dir.resolve()
        
        if not optimizations_dir.exists():
            raise FileNotFoundError(
                f"Optimizations directory not found: {optimizations_dir}. "
                "Please set generation.optimizations_dir in config. "
                f"Resolved from: {self.config.generation.optimizations_dir}"
            )
        
        optimizations_list = self.config.generation.optimizations
        specs = load_optimization_specs(optimizations_dir, optimizations_list)
        self.logger.info(f"Loaded {len(specs)} optimization specifications")
        
        state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
        state = WhiteFoxState.load(state_file, specs)
        
        for opt_name, spec in specs.items():
            if opt_name not in state.optimizations:
                state.optimizations[opt_name] = OptimizationState(spec=spec)
        
        return state
    
    def _save_generated_test(
        self,
        generated_text: str,
        optimization_name: str,
        iteration: int,
        sample_idx: int,
        output_root: Path
    ) -> Path:

        opt_dir = output_root / optimization_name
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = opt_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}.py"
        test_file.write_text(generated_text)
        
        return test_file
    
    def _run_single_optimization(
        self,
        opt_state: OptimizationState,
        output_root: Path,
        logs_root: Path,
        bug_reports_dir: Path,
        only_optimizations: Optional[List[str]] = None
    ) -> None:

        opt_name = opt_state.spec.internal_name
        
        if only_optimizations and opt_name not in only_optimizations:
            return
        
        self.logger.info(f"Processing optimization: {opt_name}")
        
        max_iterations = self.config.generation.max_iterations
        tests_per_iteration = self.config.generation.tests_per_iteration
        examples_per_prompt = self.config.generation.examples_per_prompt
        
        for iteration in range(max_iterations):
            self.logger.info(
                f"  Iteration {iteration + 1}/{max_iterations} for {opt_name}"
            )
            
            if opt_state.triggering_tests:
                example_tests = select_examples_thompson_sampling(
                    opt_state,
                    examples_per_prompt
                )
                prompt = build_feedback_prompt(opt_state.spec, example_tests)
            else:
                example_tests = []
                prompt = build_base_prompt(opt_state.spec)
            
            sampling_params = self._create_sampling_params(tests_per_iteration)
            outputs = self.llm.generate([prompt], sampling_params)
            
            generated_texts = []
            for output in outputs:
                for text_output in output.outputs:
                    generated_texts.append(text_output.text)
            
            new_triggering_tests = []
            num_triggered = 0
            num_not_triggered = 0
            
            for sample_idx, generated_text in enumerate(generated_texts):
                test_file = self._save_generated_test(
                    generated_text,
                    opt_name,
                    iteration,
                    sample_idx,
                    output_root
                )
                
                result = execute_test_in_subprocess(test_file)
                
                log_file = logs_root / opt_name / f"{test_file.stem}.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(result.log_text)
                
                pass_triggered = opt_state.spec.pass_log_name in result.triggered_passes
                
                if pass_triggered:
                    num_triggered += 1
                    is_new = True
                    for existing_test in opt_state.triggering_tests.values():
                        if existing_test.file_path == test_file:
                            is_new = False
                            break
                    
                    if is_new:
                        new_triggering_tests.append(test_file)
                else:
                    num_not_triggered += 1
                
                test_code = None
                try:
                    test_code = test_file.read_text()
                except Exception:
                    pass
                
                bug_reports = check_oracles(
                    result,
                    test_code=test_code
                )
                
                for bug_report in bug_reports:
                    bug_file = bug_reports_dir / f"{test_file.stem}-bug.json"
                    bug_report.save(bug_file)
                    self.logger.warning(
                        f"Bug detected: {bug_report.oracle_type} in {test_file}"
                    )
            
            if opt_state.triggering_tests or new_triggering_tests:
                update_bandit_after_generation(
                    opt_state,
                    example_tests,
                    num_triggered,
                    num_not_triggered,
                    new_triggering_tests
                )
            
            state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
            self.whitefox_state.save(state_file)
            
            self.logger.info(
                f"  Iteration {iteration + 1} complete: "
                f"{num_triggered} triggered, {num_not_triggered} not triggered, "
                f"{len(new_triggering_tests)} new triggering tests"
            )
    
    def generate_whitefox(
        self,
        only_optimizations: Optional[List[str]] = None
    ) -> None:
        self.whitefox_state = self._load_or_init_whitefox_state()
        
        output_dir = Path(self.config.paths.output_dir)
        output_root = Path(
            self.config.paths.test_output_root or 
            str(output_dir / "whitefox_tests")
        )
        logs_root = Path(
            self.config.paths.logs_root or
            str(output_dir / "whitefox_logs")
        )
        bug_reports_dir = Path(
            self.config.paths.bug_reports_dir or
            str(output_dir / "whitefox_bugs")
        )
        
        output_root.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)
        bug_reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting WhiteFox fuzzing with {len(self.whitefox_state.optimizations)} optimizations")
        
        for opt_state in self.whitefox_state.optimizations.values():
            try:
                self._run_single_optimization(
                    opt_state,
                    output_root,
                    logs_root,
                    bug_reports_dir,
                    only_optimizations
                )
            except Exception as e:
                self.logger.error(f"Error processing {opt_state.spec.internal_name}: {e}", exc_info=True)
        
        self.logger.info("WhiteFox fuzzing complete")

