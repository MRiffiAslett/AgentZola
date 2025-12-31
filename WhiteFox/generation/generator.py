
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
from generation.code_cleaner import clean_generated_code
from generation.logger import WhiteFoxLogger
from generation.api_validator import validate_tensorflow_apis
from generation.sanity_checker import run_sanity_check

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
    
    def _get_logging_dir(self) -> Optional[Path]:
        """Get the logging directory path."""
        cwd = Path.cwd()
        if cwd.name == "WhiteFox":
            project_root = cwd
        else:
            current = cwd
            project_root = None
            while current != current.parent:
                if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                    project_root = current / "WhiteFox"
                    break
                current = current.parent
            
            if not project_root:
                project_root = Path("WhiteFox")
        
        return project_root / "logging"
    
    def _setup_logging(self) -> None:
        # Determine logging directory - use WhiteFox/logging if relative, or absolute path
        log_file_path = Path(self.config.paths.log_file or "whitefox-llm-gen.log")
        
        # If log file is relative, put it in WhiteFox/logging
        if not log_file_path.is_absolute():
            # Find project root (directory containing WhiteFox)
            cwd = Path.cwd()
            project_root = None
            if cwd.name == "WhiteFox":
                project_root = cwd
            else:
                current = cwd
                while current != current.parent:
                    if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                        project_root = current / "WhiteFox"
                        break
                    current = current.parent
            
            if project_root:
                log_file_path = project_root / "logging" / log_file_path.name
            else:
                log_file_path = Path("logging") / log_file_path.name
        
        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_file_path),
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
        
        # Resolve relative paths
        if not optimizations_dir.is_absolute():
            # If path starts with "WhiteFox/", resolve relative to project root
            if str(optimizations_dir).startswith("WhiteFox/"):
                cwd = Path.cwd()
                project_root = None
                
                # If current directory is "WhiteFox", project root is its parent
                if cwd.name == "WhiteFox":
                    project_root = cwd.parent
                else:
                    # Go up until we find the directory containing WhiteFox
                    current = cwd
                    while current != current.parent:
                        if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                            project_root = current
                            break
                        current = current.parent
                
                if project_root:
                    optimizations_dir = (project_root / optimizations_dir).resolve()
                else:
                    # Fallback: assume we're at project root
                    optimizations_dir = (cwd / optimizations_dir).resolve()
            elif self.config_file_path:
                # Otherwise, resolve relative to config file directory
                config_dir = self.config_file_path.parent
                optimizations_dir = (config_dir / optimizations_dir).resolve()
            else:
                # Fallback: resolve relative to current working directory
                optimizations_dir = optimizations_dir.resolve()
        
        if not optimizations_dir.exists():
            raise FileNotFoundError(
                f"Optimizations directory not found: {optimizations_dir}. "
                "Please set generation.optimizations_dir in config. "
                f"Original path: {self.config.generation.optimizations_dir}, "
                f"Current working directory: {Path.cwd()}"
            )
        
        optimizations_list = self.config.generation.optimizations
        specs = load_optimization_specs(optimizations_dir, optimizations_list)
        self.logger.info(f"Loaded {len(specs)} optimization specifications")
        
        # Try to load state from logging directory first, then fallback to config or default
        logging_dir = self._get_logging_dir()
        state_file = None
        
        if logging_dir:
            state_file = logging_dir / "whitefox_state.json"
            if not state_file.exists():
                # Try old location as fallback
                old_state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
                # Also check in project root
                if not old_state_file.is_absolute():
                    project_root = logging_dir.parent
                    old_state_file = project_root / old_state_file
                
                if old_state_file.exists() and old_state_file != state_file:
                    # Move old state file to logging directory
                    import shutil
                    shutil.move(str(old_state_file), str(state_file))
                    self.logger.info(f"Moved state file from {old_state_file} to {state_file}")
        else:
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
        output_root: Path,
        whitefox_logger: Optional[WhiteFoxLogger] = None
    ) -> Path:

        opt_dir = output_root / optimization_name
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean the generated code
        cleaned_code = clean_generated_code(generated_text)
        
        # Validate TensorFlow APIs
        is_valid_api, api_errors = validate_tensorflow_apis(cleaned_code)
        
        # Track what changed during cleaning
        cleaning_changes = {
            "had_markdown": "```" in generated_text,
            "had_tf_import": "import tensorflow" in generated_text or "import tensorflow" in cleaned_code,
            "had_np_import": "import numpy" in generated_text or "import numpy" in cleaned_code,
            "raw_length": len(generated_text),
            "cleaned_length": len(cleaned_code),
            "api_valid": is_valid_api,
            "api_errors": api_errors,
        }
        
        # If invalid APIs detected, add comment and log warning
        if not is_valid_api:
            self.logger.warning(
                f"Invalid TensorFlow APIs detected in {optimization_name} it{iteration} sample{sample_idx}: {api_errors}"
            )
            # Add comment to code indicating API issues
            cleaned_code = f"# WARNING: Invalid APIs detected: {', '.join(api_errors)}\n# Code may fail at runtime\n{cleaned_code}"
        
        # Log the code generation
        if whitefox_logger:
            whitefox_logger.log_generated_code(
                optimization_name,
                iteration,
                sample_idx,
                generated_text,
                cleaned_code,
                cleaning_changes
            )
        
        test_file = opt_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}.py"
        test_file.write_text(cleaned_code)
        
        return test_file
    
    def _run_single_optimization(
        self,
        opt_state: OptimizationState,
        output_root: Path,
        logs_root: Path,
        bug_reports_dir: Path,
        whitefox_logger: WhiteFoxLogger,
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
            
            # Create a copy of state before iteration for logging
            import copy
            before_state = OptimizationState(
                spec=opt_state.spec,
                triggering_tests=copy.deepcopy(opt_state.triggering_tests)
            )
            
            if opt_state.triggering_tests:
                example_tests = select_examples_thompson_sampling(
                    opt_state,
                    examples_per_prompt
                )
                # Build prompt with token limit to prevent overflow
                prompt = build_feedback_prompt(
                    opt_state.spec, 
                    example_tests,
                    max_model_len=self.config.model.max_model_len
                )
                prompt_type = "feedback"
            else:
                example_tests = []
                prompt = build_base_prompt(opt_state.spec)
                prompt_type = "base"
            
            # Log the prompt
            whitefox_logger.log_prompt(
                opt_name,
                iteration,
                prompt_type,
                prompt,
                example_tests
            )
            
            try:
                sampling_params = self._create_sampling_params(tests_per_iteration)
                outputs = self.llm.generate([prompt], sampling_params)
            except Exception as e:
                whitefox_logger.log_error(
                    opt_name,
                    iteration,
                    "llm_generation_error",
                    str(e),
                    {"prompt_type": prompt_type, "num_examples": len(example_tests)},
                    e
                )
                self.logger.error(f"LLM generation error for {opt_name} it{iteration}: {e}", exc_info=True)
                continue
            
            generated_texts = []
            for output in outputs:
                for text_output in output.outputs:
                    generated_texts.append(text_output.text)
            
            new_triggering_tests = []
            num_triggered = 0
            num_not_triggered = 0
            
            for sample_idx, generated_text in enumerate(generated_texts):
                try:
                    test_file = self._save_generated_test(
                        generated_text,
                        opt_name,
                        iteration,
                        sample_idx,
                        output_root,
                        whitefox_logger
                    )
                    
                    result = execute_test_in_subprocess(test_file)
                    
                    # Save execution log to logs_root (for compatibility)
                    log_file = logs_root / opt_name / f"{test_file.stem}.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    log_file.write_text(result.log_text)
                    
                    pass_triggered = opt_state.spec.pass_log_name in result.triggered_passes
                    
                    # Log pass detection analysis
                    whitefox_logger.log_pass_detection_analysis(
                        opt_name,
                        iteration,
                        sample_idx,
                        opt_state.spec.pass_log_name,
                        result.log_text,
                        result.triggered_passes,
                        opt_state.spec.pass_log_name
                    )
                    
                    # Log execution result
                    whitefox_logger.log_execution_result(
                        opt_name,
                        iteration,
                        sample_idx,
                        test_file,
                        result,
                        pass_triggered,
                        opt_state.spec.pass_log_name
                    )
                    
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
                
                except Exception as e:
                    whitefox_logger.log_error(
                        opt_name,
                        iteration,
                        "sample_processing_error",
                        f"Error processing sample {sample_idx}: {str(e)}",
                        {"sample_idx": sample_idx},
                        e
                    )
                    self.logger.error(f"Error processing sample {sample_idx} for {opt_name} it{iteration}: {e}", exc_info=True)
            
            # Log state update
            if opt_state.triggering_tests or new_triggering_tests:
                update_bandit_after_generation(
                    opt_state,
                    example_tests,
                    num_triggered,
                    num_not_triggered,
                    new_triggering_tests
                )
            
            # Log state change
            whitefox_logger.log_state_update(
                opt_name,
                iteration,
                before_state,
                opt_state,
                num_triggered,
                num_not_triggered,
                new_triggering_tests,
                example_tests
            )
            
            # Save state to logging directory
            logging_dir = self._get_logging_dir()
            if logging_dir:
                state_file = logging_dir / "whitefox_state.json"
            else:
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
        
        # Determine logging directory - consolidate in WhiteFox/logging
        logging_dir = self._get_logging_dir()
        if not logging_dir:
            logging_dir = Path("logging")
        
        # Move generated-outputs into logging if it exists
        project_root = logging_dir.parent
        old_generated_outputs = project_root / "generated-outputs"
        if old_generated_outputs.exists() and old_generated_outputs.is_dir():
            generated_outputs_dst = logging_dir / "generated-outputs"
            if not generated_outputs_dst.exists():
                import shutil
                shutil.move(str(old_generated_outputs), str(generated_outputs_dst))
                self.logger.info(f"Moved generated-outputs from {old_generated_outputs} to {generated_outputs_dst}")
                # Update output_root to point to new location
                output_root = generated_outputs_dst / "whitefox_tests"
            else:
                # Already moved, use existing location
                output_root = generated_outputs_dst / "whitefox_tests"
        else:
            # Use new location in logging
            output_root = logging_dir / "generated-outputs" / "whitefox_tests"
        
        logs_root = logging_dir / "execution_logs"  # For compatibility with existing logs
        bug_reports_dir = logging_dir / "bug_reports"
        
        output_root.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)
        bug_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WhiteFox logger
        whitefox_logger = WhiteFoxLogger(logging_dir, self.logger)
        
        # Move whitefox_state.json to logging if it exists in old location
        old_state_file = project_root / "whitefox_state.json"
        new_state_file = logging_dir / "whitefox_state.json"
        if old_state_file.exists() and not new_state_file.exists():
            import shutil
            shutil.move(str(old_state_file), str(new_state_file))
            self.logger.info(f"Moved whitefox_state.json from {old_state_file} to {new_state_file}")
        
        self.logger.info(f"Starting WhiteFox fuzzing with {len(self.whitefox_state.optimizations)} optimizations")
        self.logger.info(f"Logging directory: {logging_dir}")
        
        # Run sanity check at start
        try:
            json_file, text_file = run_sanity_check(logging_dir, self.whitefox_state)
            self.logger.info(f"Sanity check completed: {text_file}")
        except Exception as e:
            self.logger.warning(f"Sanity check failed: {e}", exc_info=True)
        
        for opt_state in self.whitefox_state.optimizations.values():
            try:
                self._run_single_optimization(
                    opt_state,
                    output_root,
                    logs_root,
                    bug_reports_dir,
                    whitefox_logger,
                    only_optimizations
                )
            except Exception as e:
                whitefox_logger.log_error(
                    opt_state.spec.internal_name,
                    None,
                    "optimization_processing_error",
                    str(e),
                    {},
                    e
                )
                self.logger.error(f"Error processing {opt_state.spec.internal_name}: {e}", exc_info=True)
        
        # Run sanity check at end
        try:
            json_file, text_file = run_sanity_check(logging_dir, self.whitefox_state)
            self.logger.info(f"Final sanity check completed: {text_file}")
        except Exception as e:
            self.logger.warning(f"Final sanity check failed: {e}", exc_info=True)
        
        self.logger.info("WhiteFox fuzzing complete")

