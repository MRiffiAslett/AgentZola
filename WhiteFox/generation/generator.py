"""
WhiteFox Generator - Core fuzzing logic.

This module contains the StarCoderGenerator class that implements the WhiteFox
white-box compiler fuzzing methodology using LLMs.
"""

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, List

from vllm import LLM, SamplingParams

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from AgentZola.WhiteFox.models.generation import GeneratorConfig

# WhiteFox imports
from .spec import load_optimization_specs
from .bandit import (
    WhiteFoxState,
    OptimizationState,
    select_examples_thompson_sampling,
    update_bandit_after_generation,
)
from .prompts import build_base_prompt, build_feedback_prompt
from .harness import execute_test_in_subprocess, check_oracles

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
    """WhiteFox compiler fuzzer using LLMs for test generation."""
    
    def __init__(self, config: GeneratorConfig):
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
    
    def _load_or_init_whitefox_state(self) -> WhiteFoxState:
        """Load existing WhiteFox state or initialize new state."""
        # Load optimization specs
        optimizations_dir = Path(self.config.generation.optimizations_dir)
        if not optimizations_dir.exists():
            raise FileNotFoundError(
                f"Optimizations directory not found: {optimizations_dir}. "
                "Please set generation.optimizations_dir in config."
            )
        
        # Use hardcoded optimizations list from config if available, otherwise scan directory
        optimizations_list = self.config.generation.optimizations
        specs = load_optimization_specs(optimizations_dir, optimizations_list)
        self.logger.info(f"Loaded {len(specs)} optimization specifications")
        
        # Load or create state
        state_file = Path(self.config.paths.bandit_state_file or "whitefox_state.json")
        state = WhiteFoxState.load(state_file, specs)
        
        # Initialize any new optimizations that weren't in the saved state
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
        """
        Save a generated test to a file.
        
        Returns:
            Path to the saved test file.
        """
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
        """
        Run fuzzing for a single optimization.
        
        Args:
            opt_state: OptimizationState for this optimization.
            output_root: Root directory for test outputs.
            logs_root: Root directory for execution logs.
            bug_reports_dir: Directory for bug reports.
            only_optimizations: If provided, only process optimizations in this list.
        """
        opt_name = opt_state.spec.internal_name
        
        # Skip if filtering and this optimization is not in the list
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
            
            # Select examples using Thompson Sampling (if any exist)
            if opt_state.triggering_tests:
                example_tests = select_examples_thompson_sampling(
                    opt_state,
                    examples_per_prompt
                )
                prompt = build_feedback_prompt(opt_state.spec, example_tests)
            else:
                example_tests = []
                prompt = build_base_prompt(opt_state.spec)
            
            # Generate batch of tests using existing vLLM setup
            sampling_params = self._create_sampling_params(tests_per_iteration)
            outputs = self.llm.generate([prompt], sampling_params)
            
            # Extract generated texts
            generated_texts = []
            for output in outputs:
                for text_output in output.outputs:
                    generated_texts.append(text_output.text)
            
            # Save and execute each generated test
            new_triggering_tests = []
            num_triggered = 0
            num_not_triggered = 0
            
            for sample_idx, generated_text in enumerate(generated_texts):
                # Save test file
                test_file = self._save_generated_test(
                    generated_text,
                    opt_name,
                    iteration,
                    sample_idx,
                    output_root
                )
                
                # Execute test
                result = execute_test_in_subprocess(test_file)
                
                # Save execution log
                log_file = logs_root / opt_name / f"{test_file.stem}.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(result.log_text)
                
                # Check if this optimization was triggered
                pass_triggered = opt_state.spec.pass_log_name in result.triggered_passes
                
                if pass_triggered:
                    num_triggered += 1
                    # Check if this is a new triggering test
                    is_new = True
                    for existing_test in opt_state.triggering_tests.values():
                        if existing_test.file_path == test_file:
                            is_new = False
                            break
                    
                    if is_new:
                        new_triggering_tests.append(test_file)
                else:
                    num_not_triggered += 1
                
                # Check oracles and save bug reports
                bug_reports = check_oracles(
                    result,
                    rtol=self.config.oracles.float_rtol,
                    atol=self.config.oracles.float_atol
                )
                
                for bug_report in bug_reports:
                    bug_file = bug_reports_dir / f"{test_file.stem}-bug.json"
                    bug_report.save(bug_file)
                    self.logger.warning(
                        f"Bug detected: {bug_report.oracle_type} in {test_file}"
                    )
            
            # Update bandit state
            if opt_state.triggering_tests or new_triggering_tests:
                update_bandit_after_generation(
                    opt_state,
                    example_tests,
                    num_triggered,
                    num_not_triggered,
                    new_triggering_tests
                )
            
            # Persist state after each iteration
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
        """
        Run WhiteFox fuzzing loop.
        
        Args:
            only_optimizations: If provided, only process these optimizations
                (list of internal names).
        """
        # Initialize WhiteFox state
        self.whitefox_state = self._load_or_init_whitefox_state()
        
        # Setup directories
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
        
        # Process each optimization
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

