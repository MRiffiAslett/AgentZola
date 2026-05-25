import logging
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import tomllib
from domain.bandit import (
    OptimizationState,
    WhiteFoxState,
)
from domain.generation import GeneratorConfig, TestExecutionResult, TestExecutionTask
from generation.bandit import (
    select_examples_thompson_sampling,
    update_bandit_after_generation,
)
from generation.code_refiner import parse_generated_code, refine_generated_code
from generation.oracle import check_oracles
from generation.prompts import (
    build_base_prompt,
    build_feedback_prompt,
    load_optimization_specs,
)
from generation.wf_logging import CoverageCollector, WhiteFoxLogger, WhiteFoxProfiler
from vllm import LLM, SamplingParams


def _execute_test_worker(task: TestExecutionTask) -> TestExecutionResult:

    try:
        if task.extra_env:
            os.environ.update(task.extra_env)

        from generation.harness import execute_test_in_subprocess

        result = execute_test_in_subprocess(
            task.test_file,
            whitefox_logger=None,
            optimization_name=task.opt_name,
            iteration=task.iteration,
            sample_idx=task.sample_idx,
            timeout=task.timeout,
            harness=task.harness if hasattr(task, "harness") else None,
        )

        return TestExecutionResult(task=task, execution_result=result, error=None)

    except Exception as e:
        import traceback

        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        return TestExecutionResult(task=task, execution_result=None, error=error_msg)


def _pool_worker_init() -> None:
    """Set RLIMIT_AS on ProcessPoolExecutor workers so TF/XLA allocations
    in the worker process (not just Popen grandchildren) are bounded."""
    import resource
    import sys
    mem_gb = int(os.environ.get("WHITEFOX_TEST_MEM_LIMIT_GB", "8"))
    rlimit_bytes = mem_gb * 3 * (1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (rlimit_bytes, rlimit_bytes))
    except Exception as exc:
        print(
            f"WHITEFOX_POOL_INIT: RLIMIT_AS={rlimit_bytes} FAILED: {exc}",
            file=sys.stderr, flush=True,
        )
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft != rlimit_bytes:
            print(
                f"WHITEFOX_POOL_INIT: RLIMIT_AS mismatch: "
                f"wanted={rlimit_bytes}, got soft={soft} hard={hard}",
                file=sys.stderr, flush=True,
            )
    except Exception:
        pass


_HARNESS = {
    "xla": "generation.harness.xla.TensorFlowXLAHarness",
    "inductor": "generation.harness.inductor.PyTorchInductorHarness",
    "tflite": "generation.harness.tflite.TensorFlowLiteHarness",
}
_PARSER = {
    "xla": "generation.code_processing.tensorflow.TensorFlowCodeParser",
    "inductor": "generation.code_processing.pytorch.PyTorchCodeParser",
    "tflite": "generation.code_processing.tflite.TensorFlowLiteCodeParser",
}


def _import_class(dotted: str):
    from importlib import import_module

    mod_path, cls_name = dotted.rsplit(".", 1)
    return getattr(import_module(mod_path), cls_name)


class StarCoderGenerator:
    def __init__(
        self,
        config: GeneratorConfig,
        config_file_path: Optional[Path] = None,
    ):
        self.config = config
        self.config_file_path = config_file_path
        self.sut_name = config.sut.name

        self.harness = _import_class(_HARNESS[self.sut_name])()
        self.parser = _import_class(_PARSER[self.sut_name])()

        self.logging_dir = self._get_logging_dir()

        self._setup_environment()
        self._setup_logging()
        self.llm = self._initialize_llm()

        self._prompt_style = config.prompts.prompt_style if config.prompts else "paper"
        self._seed_qa = self._load_seed_qa()

        self._state_lock = threading.RLock()
        self._llm_lock = threading.Lock()

    @classmethod
    def from_config_file(cls, config_path: Path) -> "StarCoderGenerator":

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)

        config = GeneratorConfig.from_toml(toml_data)

        return cls(config, config_file_path=config_path)

    def _setup_environment(self) -> None:
        if self.config.paths.hf_home:
            os.environ["HF_HOME"] = os.environ.get("HF_HOME", self.config.paths.hf_home)

        if self.config.paths.hf_cache:
            os.environ["HF_CACHE"] = self.config.paths.hf_cache

    def _get_logging_dir(self) -> Path:
        override = os.environ.get("WHITEFOX_LOGGING_DIR")
        if override:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
        generation_dir = Path(__file__).parent
        whitefox_dir = generation_dir.parent
        return whitefox_dir / "logging"

    def _load_seed_qa(self) -> str:
        if self._prompt_style != "stacked":
            return ""
        seed_file = self.config.prompts.seed_file if self.config.prompts else ""
        if not seed_file:
            return ""
        path = self._resolve_path(Path(seed_file))
        if not path.exists():
            self.logger.warning(f"Seed file not found: {path}")
            return ""
        return path.read_text().strip()

    def _setup_logging(self) -> None:
        log_file_path = Path(self.config.paths.log_file or "whitefox-llm-gen.log")

        if not log_file_path.is_absolute():
            log_file_path = self.logging_dir / log_file_path.name

        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_file_path),
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
            seed=random.randint(0, 10000),
        )

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parent.parent

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path

        return (self._project_root() / path).resolve()

    def _load_or_init_whitefox_state(self) -> WhiteFoxState:
        optimizations_dir = self._resolve_path(
            Path(self.config.generation.optimizations_dir)
        )
        optimizations_list = self.config.generation.optimizations
        pass_name_aliases = None
        if (
            self.config.pass_name_aliases
            and self.config.pass_name_aliases.pass_name_aliases
        ):
            pass_name_aliases = self.config.pass_name_aliases.pass_name_aliases
        specs = load_optimization_specs(
            optimizations_dir, optimizations_list, pass_name_aliases
        )

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
        whitefox_logger: Optional[WhiteFoxLogger] = None,
    ) -> Path:

        opt_dir = output_root / optimization_name
        opt_dir.mkdir(parents=True, exist_ok=True)

        processed_code = refine_generated_code(
            generated_text,
            framework=self.config.sut.framework,
            parser=self.parser,
        )

        parsed_code = parse_generated_code(generated_text)

        if whitefox_logger:
            whitefox_logger.log_generated_code(
                optimization_name,
                iteration,
                sample_idx,
                generated_text,
                processed_code,
                parsed_code,
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
        timeout: Optional[int] = None,
    ) -> List[TestExecutionResult]:

        max_workers = self.config.generation.parallel_test_workers
        env_workers = os.environ.get("WHITEFOX_PARALLEL_TEST_WORKERS")
        if env_workers:
            try:
                max_workers = int(env_workers)
            except ValueError:
                pass
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        max_workers = max(1, int(max_workers))

        if timeout is None:
            timeout = self.config.generation.test_timeout

        tasks = []
        for sample_idx, generated_text in enumerate(generated_texts):
            test_file = self._save_generated_test(
                generated_text,
                opt_name,
                iteration,
                sample_idx,
                output_root,
                whitefox_logger,
            )

            task = TestExecutionTask(
                test_file=test_file,
                opt_name=opt_name,
                iteration=iteration,
                sample_idx=sample_idx,
                timeout=timeout,
                extra_env=self.coverage.env_vars(),
            )
            tasks.append(task)

        results = []

        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_pool_worker_init,
            ) as executor:
                future_to_task = {
                    executor.submit(_execute_test_worker, task): task for task in tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        result = future.result()
                        results.append(result)

                    except Exception as e:
                        results.append(
                            TestExecutionResult(
                                task=task,
                                execution_result=None,
                                error=f"Future exception: {type(e).__name__}: {str(e)}",
                            )
                        )

        except Exception as e:
            self.logger.error(f"Parallel execution error: {e}", exc_info=True)
            return results

        results.sort(key=lambda r: r.task.sample_idx)

        return results

    def _run_single_optimization(
        self,
        opt_state: OptimizationState,
        output_root: Path,
        whitefox_logger: WhiteFoxLogger,
        only_optimizations: Optional[List[str]] = None,
    ) -> None:

        opt_name = opt_state.spec.internal_name

        if only_optimizations and opt_name not in only_optimizations:
            return

        max_iterations = self.config.generation.max_iterations
        tests_per_iteration = self.config.generation.tests_per_iteration
        examples_per_prompt = self.config.generation.examples_per_prompt

        opt_llm_time_s = 0.0
        opt_exec_time_s = 0.0
        opt_coverage_merge_time_s = 0.0
        opt_coverage_merged_profraw = 0
        opt_total_samples_executed = 0
        opt_iterations_done = 0
        coverage_merge_every_iters = int(
            os.environ.get("WHITEFOX_COVERAGE_MERGE_EVERY_ITERS", "1")
        )
        coverage_merge_every_iters = max(1, coverage_merge_every_iters)

        early_stop_iters = int(
            os.environ.get("WHITEFOX_EARLY_STOP_ITERS", "20")
        )

        for iteration in range(max_iterations):
            self.logger.info(
                f"  Iteration {iteration + 1}/{max_iterations} for {opt_name}"
            )
            opt_iterations_done += 1

            import copy

            before_state = OptimizationState(
                spec=opt_state.spec,
                triggering_tests=copy.deepcopy(opt_state.triggering_tests),
            )

            if opt_state.triggering_tests:
                example_tests = select_examples_thompson_sampling(
                    opt_state, examples_per_prompt
                )
                feedback_instruction = (
                    self.config.prompts.feedback_instruction
                    if self.config.prompts
                    else ""
                )
                prompt = build_feedback_prompt(
                    opt_state.spec,
                    example_tests,
                    feedback_instruction,
                    self._prompt_style,
                    seed_qa=self._seed_qa,
                )
                prompt_type = "feedback"
            else:
                example_tests = []
                instruction_header = (
                    self.config.prompts.instruction_header
                    if self.config.prompts
                    else ""
                )
                prompt = build_base_prompt(
                    opt_state.spec,
                    self._prompt_style,
                    self._seed_qa,
                    instruction_header=instruction_header,
                )
                prompt_type = "base"

            whitefox_logger.log_prompt(
                opt_name, iteration, prompt_type, prompt, example_tests
            )

            try:
                sampling_params = self._create_sampling_params(tests_per_iteration)
                t_llm0 = time.monotonic()
                with self._llm_lock:
                    outputs = self.llm.generate([prompt], sampling_params)
                t_llm1 = time.monotonic()
                opt_llm_time_s += t_llm1 - t_llm0
            except Exception as e:
                whitefox_logger.log_error(
                    opt_name,
                    iteration,
                    "llm_generation_error",
                    str(e),
                    {"prompt_type": prompt_type, "num_examples": len(example_tests)},
                    e,
                )
                self.logger.error(
                    f"LLM generation error for {opt_name} it{iteration}: {e}",
                    exc_info=True,
                )
                continue

            generated_texts = []
            for output in outputs:
                for text_output in output.outputs:
                    generated_texts.append(text_output.text)

            self.logger.info(
                "  [%s] it%d: LLM produced %d samples (prompt_type=%s, "
                "tokens: %s)",
                opt_name, iteration, len(generated_texts), prompt_type,
                ", ".join(
                    str(len(t)) for t in generated_texts
                ),
            )

            t_exec0 = time.monotonic()
            execution_results = self._execute_tests_parallel(
                generated_texts,
                opt_name,
                iteration,
                output_root,
                whitefox_logger,
                timeout=None,
            )
            t_exec1 = time.monotonic()
            opt_exec_time_s += t_exec1 - t_exec0
            opt_total_samples_executed += len(generated_texts)

            should_merge = (
                coverage_merge_every_iters <= 1
                or ((iteration + 1) % coverage_merge_every_iters == 0)
                or (iteration + 1 == max_iterations)
            )
            merged_count = 0
            if should_merge:
                t_merge0 = time.monotonic()
                merged_count = self.coverage.merge_pending()
                t_merge1 = time.monotonic()
                opt_coverage_merge_time_s += t_merge1 - t_merge0
                opt_coverage_merged_profraw += int(merged_count or 0)

            new_triggering_tests = []
            num_triggered = 0
            num_not_triggered = 0

            for exec_result in execution_results:
                task = exec_result.task
                sample_idx = task.sample_idx
                test_file = task.test_file

                try:
                    if not exec_result.success:
                        self.logger.warning(
                            "  [%s] it%d sample%d: worker failed — %s",
                            opt_name, iteration, sample_idx,
                            (exec_result.error or "unknown error")[:200],
                        )
                        whitefox_logger.log_execution_failure(
                            opt_name,
                            iteration,
                            sample_idx,
                            test_file,
                            exec_result.error or "unknown error",
                        )
                        continue

                    result = exec_result.execution_result

                    pass_triggered = opt_state.spec.matches_any_pass(
                        result.triggered_passes
                    )

                    suffix = "_Triggered" if pass_triggered else "_NotTriggered"
                    new_name = test_file.stem + suffix + test_file.suffix
                    new_path = test_file.with_name(new_name)
                    test_file.rename(new_path)
                    log_file = test_file.with_suffix(".log")
                    if log_file.exists():
                        log_file.rename(new_path.with_suffix(".log"))
                    test_file = new_path
                    result.test_file = new_path

                    self.logger.info(
                        "  [%s] it%d sample%d: pass_triggered=%s  "
                        "triggered_passes=%s  pass_name_aliases=%s",
                        opt_name, iteration, sample_idx,
                        pass_triggered,
                        sorted(result.triggered_passes) if result.triggered_passes else "{}",
                        opt_state.spec.pass_log_names,
                    )

                    whitefox_logger.log_pass_detection_analysis(
                        opt_name,
                        iteration,
                        sample_idx,
                        opt_state.spec.pass_log_name,
                        result.log_text,
                        result.triggered_passes,
                        opt_state.spec.pass_log_name,
                        expected_passes=opt_state.spec.pass_log_names,
                    )

                    whitefox_logger.log_execution_result(
                        opt_name,
                        iteration,
                        sample_idx,
                        test_file,
                        result,
                        pass_triggered,
                        opt_state.spec.pass_log_name,
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
                        test_code=test_code,
                        allowed_errors=self.config.oracles.allowed_errors,
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
                        e,
                    )

            self.logger.info(
                "  [%s] it%d summary: %d triggered, %d not triggered, "
                "%d new triggering tests, %d total existing",
                opt_name, iteration,
                num_triggered, num_not_triggered,
                len(new_triggering_tests), len(opt_state.triggering_tests),
            )

            if opt_state.triggering_tests or new_triggering_tests:
                update_bandit_after_generation(
                    opt_state,
                    example_tests,
                    num_triggered,
                    num_not_triggered,
                    new_triggering_tests,
                )

            whitefox_logger.log_state_update(
                opt_name,
                iteration,
                before_state,
                opt_state,
                num_triggered,
                num_not_triggered,
                new_triggering_tests,
                example_tests,
            )

            if (
                early_stop_iters > 0
                and iteration + 1 >= early_stop_iters
                and not opt_state.triggering_tests
            ):
                self.logger.info(
                    "  [%s] Early stopping: 0 triggering tests after %d "
                    "iterations (threshold=%d). Skipping remaining %d "
                    "iterations.",
                    opt_name, iteration + 1, early_stop_iters,
                    max_iterations - iteration - 1,
                )
                t_merge0 = time.monotonic()
                n = self.coverage.merge_pending()
                t_merge1 = time.monotonic()
                opt_coverage_merge_time_s += t_merge1 - t_merge0
                opt_coverage_merged_profraw += int(n or 0)
                break

        if hasattr(self, "profiler") and self.profiler is not None:
            self.profiler.set_optimization_metadata(
                opt_name,
                {
                    "llm_time_s": opt_llm_time_s,
                    "exec_time_s": opt_exec_time_s,
                    "coverage_merge_time_s": opt_coverage_merge_time_s,
                    "coverage_merged_profraw": opt_coverage_merged_profraw,
                    "iterations": opt_iterations_done,
                    "samples_executed": opt_total_samples_executed,
                },
            )

        with self._state_lock:
            whitefox_logger.generate_run_summary(
                sorted(self.whitefox_state.optimizations.keys()),
                opt_states=self.whitefox_state.optimizations,
            )

    def _run_optimizations_parallel(
        self,
        opt_states: List[OptimizationState],
        output_root: Path,
        whitefox_logger: WhiteFoxLogger,
        only_optimizations: Optional[List[str]],
        max_workers: int,
    ) -> None:

        def _worker(opt_state: OptimizationState) -> Tuple[str, Optional[Exception]]:
            opt_name = opt_state.spec.internal_name
            try:
                self._run_single_optimization(
                    opt_state, output_root, whitefox_logger, only_optimizations
                )
                return (opt_name, None)
            except Exception as e:
                whitefox_logger.log_error(
                    opt_name, None, "optimization_processing_error", str(e), {}, e
                )
                self.logger.error(f"Error processing {opt_name}: {e}", exc_info=True)
                return (opt_name, e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_opt = {
                executor.submit(_worker, opt_state): opt_state
                for opt_state in opt_states
            }

            for future in as_completed(future_to_opt):
                opt_name = ""
                try:
                    opt_name, _exc = future.result()
                except Exception as e:
                    self.logger.error(f"Exception in optimization: {e}", exc_info=True)
                    opt_state = future_to_opt.get(future)
                    if opt_state is not None:
                        opt_name = opt_state.spec.internal_name

                if opt_name:
                    self.profiler.append_optimization_segment(opt_name)
                    whitefox_logger.flush_and_clear()

                with self._state_lock:
                    whitefox_logger.generate_run_summary(
                        sorted(self.whitefox_state.optimizations.keys()),
                        opt_states=self.whitefox_state.optimizations,
                    )

    def generate_whitefox(self, only_optimizations: Optional[List[str]] = None) -> None:
        project_root = self._project_root()

        self.coverage = CoverageCollector(self.logging_dir)
        os.environ.update(self.coverage.env_vars())
        self.coverage.verify()

        config_dict = {
            "generation": {
                "parallel_test_workers": self.config.generation.parallel_test_workers,
                "parallel_optimizations": self.config.generation.parallel_optimizations,
            }
        }
        self.profiler = WhiteFoxProfiler(self.logging_dir, config=config_dict)
        self.profiler.begin_run()
        self.profiler.start_monitoring(interval=5.0)

        import shutil

        self.whitefox_state = self._load_or_init_whitefox_state()

        output_dir = Path(self.config.paths.output_dir)
        output_root = Path(
            self.config.paths.test_output_root or str(output_dir / "whitefox_tests")
        )

        generated_outputs_name = "generated-outputs"
        whitefox_tests_name = "whitefox_tests"

        old_generated_outputs = project_root / generated_outputs_name
        if old_generated_outputs.exists() and old_generated_outputs.is_dir():
            generated_outputs_dst = self.logging_dir / generated_outputs_name
            if not generated_outputs_dst.exists():
                shutil.move(str(old_generated_outputs), str(generated_outputs_dst))
            output_root = generated_outputs_dst / whitefox_tests_name
        else:
            output_root = (
                self.logging_dir / generated_outputs_name / whitefox_tests_name
            )

        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        whitefox_logger = WhiteFoxLogger(self.logging_dir, self.logger)
        whitefox_logger.clear_old_logs()

        opt_states_to_process = [
            opt_state
            for opt_state in self.whitefox_state.optimizations.values()
            if not only_optimizations
            or opt_state.spec.internal_name in only_optimizations
        ]

        parallel_optimizations = self.config.generation.parallel_optimizations

        self.logger.info(
            "WhiteFox run config: %d optimizations to process, "
            "max_iterations=%d, tests_per_iteration=%d, "
            "test_timeout=%ds, parallel_workers=%d, parallel_opts=%d",
            len(opt_states_to_process),
            self.config.generation.max_iterations,
            self.config.generation.tests_per_iteration,
            self.config.generation.test_timeout,
            self.config.generation.parallel_test_workers,
            parallel_optimizations,
        )
        if only_optimizations:
            self.logger.info("  Filtering to: %s", only_optimizations)

        if parallel_optimizations <= 1:
            self.logger.info("Running optimizations sequentially")
            for opt_state in opt_states_to_process:
                try:
                    self._run_single_optimization(
                        opt_state, output_root, whitefox_logger, only_optimizations
                    )
                except Exception as e:
                    whitefox_logger.log_error(
                        opt_state.spec.internal_name,
                        None,
                        "optimization_processing_error",
                        str(e),
                        {},
                        e,
                    )
                    self.logger.error(
                        f"Error processing {opt_state.spec.internal_name}: {e}",
                        exc_info=True,
                    )
                    with self._state_lock:
                        whitefox_logger.generate_run_summary(
                            sorted(self.whitefox_state.optimizations.keys()),
                            opt_states=self.whitefox_state.optimizations,
                        )
                finally:
                    self.profiler.append_optimization_segment(
                        opt_state.spec.internal_name
                    )
                    whitefox_logger.flush_and_clear()
                    import gc
                    gc.collect()
        else:
            self.logger.info(
                f"Running optimizations in parallel with {parallel_optimizations} workers"
            )
            self._run_optimizations_parallel(
                opt_states_to_process,
                output_root,
                whitefox_logger,
                only_optimizations,
                parallel_optimizations,
            )

        whitefox_logger.flush()

        self._shutdown_llm()

        profraw_files = list(self.coverage.profraw_dir.glob("*.profraw"))
        profraw_count = len(profraw_files)
        profraw_bytes = sum(f.stat().st_size for f in profraw_files)
        self.logger.info(
            "Coverage finalize: %d profraw files (%.1f MB) in %s",
            profraw_count, profraw_bytes / 1024 / 1024,
            self.coverage.profraw_dir,
        )
        coverage_result = self.coverage.finalize()

        whitefox_logger.generate_run_summary(
            sorted(self.whitefox_state.optimizations.keys()),
            opt_states=self.whitefox_state.optimizations,
            coverage_data=coverage_result,
        )
        self.logger.info("Run summary written to %s", self.logging_dir / "run_summary_detailed.log")
        self.logger.info(
            "Generation quality log written to %s",
            self.logging_dir / "generation_quality.log",
        )

        self.profiler.generate_report()
        self.logger.info("Resource profile saved to %s", self.profiler.report_file)

    def _shutdown_llm(self) -> None:
        try:
            if hasattr(self, "llm") and self.llm is not None:
                self.logger.info("Shutting down vLLM engine …")
                engine = getattr(self.llm, "llm_engine", None)
                if engine is not None and hasattr(engine, "shutdown"):
                    engine.shutdown()
                del self.llm
                self.llm = None
                self.logger.info("vLLM engine shut down.")
        except Exception as exc:
            self.logger.warning("vLLM shutdown error (non-fatal): %s", exc)
