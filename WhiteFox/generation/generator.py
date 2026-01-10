
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, List
import importlib.util

_temp_modules = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('generation.logging')}
sys.modules.pop('logging', None)

_temp_path = sys.path[:]
sys.path = [p for p in sys.path if 'generation' not in p.lower() or 'AgentZola' not in p]
_spec = importlib.util.find_spec('logging')
logging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logging)
sys.modules['logging'] = logging
sys.path[:] = _temp_path
for k, v in _temp_modules.items():
    sys.modules[k] = v

from vllm import LLM, SamplingParams

from models.generation import GeneratorConfig

from generation.prompts import load_optimization_specs
from domain.bandit import (
    WhiteFoxState,
    OptimizationState,
)
from generation.bandit import (
    select_examples_thompson_sampling,
    update_bandit_after_generation,
)
from generation.prompts import build_base_prompt, build_feedback_prompt, parse_generated_code
from generation.harness import execute_test_in_subprocess
from generation.oracle import check_oracles
from generation.code_cleaner import ensure_imports
from generation.logging import WhiteFoxLogger

import tomllib


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
        
        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
        
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
    
    def _get_logging_dir(self) -> Path:
        """Get the logging directory path."""
        generation_dir = Path(__file__).parent
        whitefox_dir = generation_dir.parent
        return whitefox_dir / "logging"
    
    def _setup_logging(self) -> None:
        log_file_path = Path(self.config.paths.log_file or "whitefox-llm-gen.log")
        logging_dir = self._get_logging_dir()
        
        if not log_file_path.is_absolute():
            log_file_path = logging_dir / log_file_path.name
        
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_file_path),
            filemode='w',  # Overwrite log file on each run
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
    
    def _resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to WhiteFox directory."""
        if path.is_absolute():
            return path
        
        whitefox_dir = self._get_logging_dir().parent
        return (whitefox_dir / path).resolve()
    
    def _load_or_init_whitefox_state(self) -> WhiteFoxState:
        optimizations_dir = self._resolve_path(Path(self.config.generation.optimizations_dir))
        optimizations_list = self.config.generation.optimizations
        specs = load_optimization_specs(optimizations_dir, optimizations_list)
        
        logging_dir = self._get_logging_dir()
        source_dir = logging_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        state_file = source_dir / "whitefox_state.json"
        
        # Always create a fresh state with all optimizations from specs
        state = WhiteFoxState(optimizations={})
        
        # Initialize state with all optimizations from specs
        for opt_name, spec in specs.items():
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
        
        # Parse generated code to extract just the Python code from the response
        parsed_code = parse_generated_code(generated_text)
        
        # Ensure required imports are present
        processed_code = ensure_imports(parsed_code)
        
        if whitefox_logger:
            whitefox_logger.log_generated_code(
                optimization_name,
                iteration,
                sample_idx,
                generated_text,
                processed_code,
                parsed_code
            )
        
        test_file = opt_dir / f"{optimization_name}-it{iteration}-sample{sample_idx}.py"
        test_file.write_text(processed_code)
        
        return test_file
    
    def _run_single_optimization(
        self,
        opt_state: OptimizationState,
        output_root: Path,
        logs_root: Path,
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
                prompt = build_feedback_prompt(
                    opt_state.spec, 
                    example_tests
                )
                prompt_type = "feedback"
            else:
                example_tests = []
                prompt = build_base_prompt(opt_state.spec)
                prompt_type = "base"
            
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
                    
                    test_code_content = test_file.read_text() if test_file.exists() else None
                    whitefox_logger.log_diagnostic(
                        opt_name,
                        iteration,
                        sample_idx,
                        "test_file_created",
                        "success",
                        {"test_file": str(test_file), "code_length": len(test_code_content) if test_code_content else 0},
                        test_code=test_code_content
                    )
                    
                    result = execute_test_in_subprocess(test_file, whitefox_logger, opt_name, iteration, sample_idx)
                    
                    log_file = logs_root / opt_name / f"{test_file.stem}.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    log_file.write_text(result.log_text)
                    
                    pass_triggered = opt_state.spec.pass_log_name in result.triggered_passes
                    
                    whitefox_logger.log_pass_detection_analysis(
                        opt_name,
                        iteration,
                        sample_idx,
                        opt_state.spec.pass_log_name,
                        result.log_text,
                        result.triggered_passes,
                        opt_state.spec.pass_log_name
                    )
                    
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
                        whitefox_logger.log_bug_report(bug_report)
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
            
            if opt_state.triggering_tests or new_triggering_tests:
                update_bandit_after_generation(
                    opt_state,
                    example_tests,
                    num_triggered,
                    num_not_triggered,
                    new_triggering_tests
                )
            
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
            
            logging_dir = self._get_logging_dir()
            source_dir = logging_dir / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            state_file = source_dir / "whitefox_state.json"
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
        logging_dir = self._get_logging_dir()
        project_root = logging_dir.parent
        source_dir = logging_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear state file to ensure fresh start with all optimizations
        state_file = source_dir / "whitefox_state.json"
        old_state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
        if not old_state_file.is_absolute():
            old_state_file = project_root / old_state_file
        
        # Remove any existing state files to start fresh
        import shutil
        if state_file.exists():
            self.logger.info(f"Removing existing state file for fresh start: {state_file}")
            state_file.unlink()
        if old_state_file.exists() and old_state_file != state_file:
            self.logger.info(f"Removing old state file for fresh start: {old_state_file}")
            old_state_file.unlink()
        
        # Now load or initialize fresh state with all optimizations
        self.whitefox_state = self._load_or_init_whitefox_state()
        
        output_dir = Path(self.config.paths.output_dir)
        output_root = Path(
            self.config.paths.test_output_root or 
            str(output_dir / "whitefox_tests")
        )
        
        old_generated_outputs = project_root / "generated-outputs"
        if old_generated_outputs.exists() and old_generated_outputs.is_dir():
            generated_outputs_dst = logging_dir / "generated-outputs"
            if not generated_outputs_dst.exists():
                shutil.move(str(old_generated_outputs), str(generated_outputs_dst))
                self.logger.info(f"Moved generated-outputs from {old_generated_outputs} to {generated_outputs_dst}")
                output_root = generated_outputs_dst / "whitefox_tests"
            else:
                output_root = generated_outputs_dst / "whitefox_tests"
        else:
            output_root = logging_dir / "generated-outputs" / "whitefox_tests"
        
        logs_root = logging_dir / "execution_logs"
        
        # Clear old logs and outputs at the start of each run
        # Clear execution logs directory
        if logs_root.exists():
            shutil.rmtree(logs_root)
        logs_root.mkdir(parents=True, exist_ok=True)
        
        # Clear generated outputs directory
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        whitefox_logger = WhiteFoxLogger(logging_dir, self.logger)
        whitefox_logger.clear_old_logs()
        
        self.logger.info(f"Starting WhiteFox fuzzing with {len(self.whitefox_state.optimizations)} optimizations")
        self.logger.info(f"Logging directory: {logging_dir}")
        
        for opt_state in self.whitefox_state.optimizations.values():
            try:
                self._run_single_optimization(
                    opt_state,
                    output_root,
                    logs_root,
                    whitefox_logger,
                    only_optimizations
                )
                # Update run summary after each optimization completes
                whitefox_logger.generate_run_summary(self.whitefox_state)
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
                # Still update summary even if optimization failed
                whitefox_logger.generate_run_summary(self.whitefox_state)
        
        whitefox_logger.flush()
        
        # Generate run summary log
        whitefox_logger.generate_run_summary(self.whitefox_state)
        
        self.logger.info("WhiteFox fuzzing complete")

