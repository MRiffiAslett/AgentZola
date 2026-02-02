
import os
import random
import sys
import threading
from pathlib import Path
from typing import Optional, List, Tuple
import importlib.util
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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

from domain.generation import GeneratorConfig, TestExecutionTask, TestExecutionResult

from generation.prompts import load_optimization_specs
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
from generation.code_refiner import parse_generated_code, refine_generated_code
from generation.logging import WhiteFoxLogger

import tomllib


def _execute_test_worker(task: TestExecutionTask) -> TestExecutionResult:

    try:
        from generation.harness import execute_test_in_subprocess
        from domain.harness import ExecutionResult
        
        result = execute_test_in_subprocess(
            task.test_file,
            whitefox_logger=None,  # Can't pass logger across processes
            optimization_name=task.opt_name,
            iteration=task.iteration,
            sample_idx=task.sample_idx,
            timeout=task.timeout
        )
        
        return TestExecutionResult(
            task=task,
            execution_result=result,
            error=None
        )
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        return TestExecutionResult(
            task=task,
            execution_result=None,
            error=error_msg
        )


class StarCoderGenerator:
    
    def __init__(self, config: GeneratorConfig, config_file_path: Optional[Path] = None):
        self.config = config
        self.config_file_path = config_file_path
        self._setup_environment()
        self._setup_logging()
        self.llm = self._initialize_llm()
        
        # Thread safety for parallel optimization execution
        self._state_lock = threading.RLock()  # Protects whitefox_state saves
        
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
        if path.is_absolute():
            return path
        
        whitefox_dir = self._get_logging_dir().parent
        return (whitefox_dir / path).resolve()
    
    def _load_or_init_whitefox_state(self) -> WhiteFoxState:
        optimizations_dir = self._resolve_path(Path(self.config.generation.optimizations_dir))
        optimizations_list = self.config.generation.optimizations
        pass_name_aliases = None
        if self.config.pass_name_aliases and self.config.pass_name_aliases.pass_name_aliases:
            pass_name_aliases = self.config.pass_name_aliases.pass_name_aliases
        specs = load_optimization_specs(optimizations_dir, optimizations_list, pass_name_aliases)
        
        logging_dir = self._get_logging_dir()
        source_dir = logging_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        state_file = source_dir / "whitefox_state.json"
        
        state = WhiteFoxState(optimizations={})
        
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
        
        processed_code = refine_generated_code(generated_text)
        
        parsed_code = parse_generated_code(generated_text)
        
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
    
    def _execute_tests_parallel(
        self,
        generated_texts: List[str],
        opt_name: str,
        iteration: int,
        output_root: Path,
        whitefox_logger: WhiteFoxLogger,
        timeout: Optional[int] = None
    ) -> List[TestExecutionResult]:
        
        max_workers = self.config.generation.parallel_test_workers
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        # Use timeout from config if not explicitly provided
        if timeout is None:
            timeout = self.config.generation.test_timeout
        
        # Phase 1: Create all test files (sequential - fast operation)
        tasks = []
        for sample_idx, generated_text in enumerate(generated_texts):
            test_file = self._save_generated_test(
                generated_text,
                opt_name,
                iteration,
                sample_idx,
                output_root,
                whitefox_logger
            )
            
            # Create task for parallel execution
            task = TestExecutionTask(
                test_file=test_file,
                opt_name=opt_name,
                iteration=iteration,
                sample_idx=sample_idx,
                timeout=timeout
            )
            tasks.append(task)
        
        results = []
        completed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_execute_test_worker, task): task
                    for task in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            whitefox_logger.trace(
                                f"      - Test {completed_count}/{len(tasks)} completed: "
                                f"{task.opt_name}-it{task.iteration}-sample{task.sample_idx}"
                            )
                        else:
                            whitefox_logger.trace(
                                f"      - Test {completed_count}/{len(tasks)} FAILED: "
                                f"{task.opt_name}-it{task.iteration}-sample{task.sample_idx}",
                                {"error": result.error[:200] if result.error else "Unknown"}
                            )
                    
                    except Exception as e:
                        whitefox_logger.trace(
                            f"      - Test {completed_count}/{len(tasks)} EXCEPTION: "
                            f"{task.opt_name}-it{task.iteration}-sample{task.sample_idx}",
                            {"error": str(e)[:200]}
                        )
                        
                        results.append(TestExecutionResult(
                            task=task,
                            execution_result=None,
                            error=f"Future exception: {type(e).__name__}: {str(e)}"
                        ))
        
        except Exception as e:
            whitefox_logger.trace(
                f"      - ProcessPoolExecutor error during parallel test execution",
                {"error": str(e)[:500], "completed": completed_count, "total": len(tasks)}
            )
            self.logger.error(f"Parallel execution error: {e}", exc_info=True)
            return results
        
        results.sort(key=lambda r: r.task.sample_idx)
        
        return results
    
    def _run_single_optimization(
        self,
        opt_state: OptimizationState,
        output_root: Path,
        whitefox_logger: WhiteFoxLogger,
        only_optimizations: Optional[List[str]] = None
    ) -> None:

        opt_name = opt_state.spec.internal_name
        
        if only_optimizations and opt_name not in only_optimizations:
            return
                
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
            
            
            execution_results = self._execute_tests_parallel(
                generated_texts,
                opt_name,
                iteration,
                output_root,
                whitefox_logger,
                timeout=None  # Use timeout from config
            )
            
            # Process results sequentially to maintain state consistency
            new_triggering_tests = []
            num_triggered = 0
            num_not_triggered = 0
            
            for exec_result in execution_results:
                task = exec_result.task
                sample_idx = task.sample_idx
                test_file = task.test_file
                
                
                try:
                    # Check if execution failed at the worker level
                    if not exec_result.success:
                        continue
                    
                    result = exec_result.execution_result
                    
                    whitefox_logger.trace(f"        * Execution completed for {test_file.name}", {
                        "naive_success": result.runtime_success_naive,
                        "xla_success": result.runtime_success_xla,
                        "autocluster_success": result.runtime_success_autocluster,
                    })
                    
                    # Check if pass was triggered
                    pass_triggered = opt_state.spec.matches_any_pass(result.triggered_passes)
                    
                    
                    whitefox_logger.log_pass_detection_analysis(
                        opt_name,
                        iteration,
                        sample_idx,
                        opt_state.spec.pass_log_name, 
                        result.log_text,
                        result.triggered_passes,
                        opt_state.spec.pass_log_name,
                        expected_passes=opt_state.spec.pass_log_names  
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
                        "result_processing_error",
                        f"Error processing result for sample {sample_idx}: {str(e)}",
                        {"sample_idx": sample_idx},
                        e
                    )
            
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
            
            # Thread-safe state saving for parallel execution
            with self._state_lock:
                logging_dir = self._get_logging_dir()
                source_dir = logging_dir / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                state_file = source_dir / "whitefox_state.json"
                self.whitefox_state.save(state_file)
        
        # Generate/update run summary after this optimization completes
        # This ensures we have partial results even if the job doesn't finish
        with self._state_lock:
            whitefox_logger.generate_run_summary(self.whitefox_state)
    
    def _run_optimizations_parallel(
        self,
        opt_states: List[OptimizationState],
        output_root: Path,
        whitefox_logger: WhiteFoxLogger,
        only_optimizations: Optional[List[str]],
        max_workers: int
    ) -> None:
      
        def _worker(opt_state: OptimizationState) -> Tuple[str, Optional[Exception]]:
            opt_name = opt_state.spec.internal_name
            try:
                self._run_single_optimization(
                    opt_state,
                    output_root,
                    whitefox_logger,
                    only_optimizations
                )
                return (opt_name, None)
            except Exception as e:
                whitefox_logger.log_error(
                    opt_name,
                    None,
                    "optimization_processing_error",
                    str(e),
                    {},
                    e
                )
                self.logger.error(f"Error processing {opt_name}: {e}", exc_info=True)
                return (opt_name, e)
        
        completed_count = 0
        total_count = len(opt_states)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all optimization tasks
            future_to_opt = {
                executor.submit(_worker, opt_state): opt_state
                for opt_state in opt_states
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_opt):
                opt_state = future_to_opt[future]
                opt_name = opt_state.spec.internal_name
                completed_count += 1
                
                try:
                    result_name, error = future.result()
                    if error is None:
                        self.logger.info(
                            f"✓ Completed optimization {completed_count}/{total_count}: {result_name}"
                        )
                    else:
                        self.logger.warning(
                            f"✗ Failed optimization {completed_count}/{total_count}: {result_name}"
                        )
                except Exception as e:
                    self.logger.error(
                        f"✗ Exception in optimization {completed_count}/{total_count}: {opt_name} - {e}",
                        exc_info=True
                    )
                
                # Update run summary after each optimization completes (even if it failed)
                # This ensures we have partial results if the job is interrupted
                with self._state_lock:
                    whitefox_logger.generate_run_summary(self.whitefox_state)
        
        self.logger.info(f"Parallel optimization execution completed: {completed_count}/{total_count}")
            
    
    def generate_whitefox(
        self,
        only_optimizations: Optional[List[str]] = None
    ) -> None:
        logging_dir = self._get_logging_dir()
        project_root = logging_dir.parent
        source_dir = logging_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = source_dir / "whitefox_state.json"
        old_state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
        if not old_state_file.is_absolute():
            old_state_file = project_root / old_state_file
        
        import shutil
        if state_file.exists():
            state_file.unlink()
        if old_state_file.exists() and old_state_file != state_file:
            old_state_file.unlink()
        
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
                output_root = generated_outputs_dst / "whitefox_tests"
            else:
                output_root = generated_outputs_dst / "whitefox_tests"
        else:
            output_root = logging_dir / "generated-outputs" / "whitefox_tests"
        
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        whitefox_logger = WhiteFoxLogger(logging_dir, self.logger)
        whitefox_logger.clear_old_logs()
        
        opt_states_to_process = [
            opt_state for opt_state in reversed(list(self.whitefox_state.optimizations.values()))
            if not only_optimizations or opt_state.spec.internal_name in only_optimizations
        ]
        
        parallel_optimizations = self.config.generation.parallel_optimizations
        
        if parallel_optimizations <= 1:
            # Sequential execution (backward compatible)
            self.logger.info("Running optimizations sequentially")
            for opt_state in opt_states_to_process:
                try:
                    self._run_single_optimization(
                        opt_state,
                        output_root,
                        whitefox_logger,
                        only_optimizations
                    )
                    # Note: run summary is generated inside _run_single_optimization
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
                    # Generate summary even after error so we capture partial results
                    with self._state_lock:
                        whitefox_logger.generate_run_summary(self.whitefox_state)
        else:
            # Parallel execution with thread pool
            self.logger.info(f"Running optimizations in parallel with {parallel_optimizations} workers")
            self._run_optimizations_parallel(
                opt_states_to_process,
                output_root,
                whitefox_logger,
                only_optimizations,
                parallel_optimizations
            )
        
        whitefox_logger.flush()
        
        # Generate run summary log once at the end
        whitefox_logger.generate_run_summary(self.whitefox_state)
        

