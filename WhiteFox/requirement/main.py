
import argparse
from pathlib import Path
from prompt_gen import generate_requirement_prompts, Src2NLTFXLA
from prompt_exec import GPTModel


def main():

    base_dir = Path(__file__).parent.parent  # WhiteFox directory
    
    config_file = base_dir / "xilo_xla" / "config" / "requirements.toml"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        import tomllib
        with open(config_file, 'rb') as f:
            config = tomllib.load(f)
    except ImportError:
        import toml
        with open(config_file, 'r') as f:
            config = toml.load(f)
    
    gpt_config = config.get('gpt', {})
    use_mini = gpt_config.get('use_mini', False)
    
    opt_spec_path = base_dir / "xilo_xla" / "artifacts" / "optimization_specification_xla.json"
    template_path = base_dir / "xilo_xla" / "prompt_template.txt"
    prompts_dir = base_dir / "xilo_xla" / "artifacts" / "requirement-prompts"
    gpt_output_dir = base_dir / "xilo_xla" / "artifacts" / "generation-prompts"
    
    if not opt_spec_path.exists():
        raise FileNotFoundError(
            f"Optimization specification file not found: {opt_spec_path}\n"
            "Please ensure the file exists before running."
        )
    
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_path}\n"
            "Please ensure the file exists before running."
        )
    
    prompts, fallback_optimizations = generate_requirement_prompts(
        optpath=str(opt_spec_path),
        template_path=str(template_path),
        outdir=str(prompts_dir),
        use_mini=use_mini
    )

    
    try:
        api_key_file = gpt_config.get('api_key_file', 'openai.key')
        model = gpt_config.get('model', 'gpt-4-turbo')
        temperature = gpt_config.get('temperature', 0.7)
        max_tokens = gpt_config.get('max_tokens', 2048)
        timeout = gpt_config.get('timeout', 300)
        n_samples = gpt_config.get('n_samples', 1)
        max_retries = gpt_config.get('max_retries', 3)
        retry_delay = gpt_config.get('retry_delay', 10)
        skip_existing = gpt_config.get('skip_existing', False)
        system_message = gpt_config.get('system_message', None)
        
        prefix_text = config.get('prefix', {}).get('text', '')
        
        api_key_path = Path(api_key_file)
        if not api_key_path.exists():
            raise FileNotFoundError(
                f"API key file not found: {api_key_file}\n"
                f"Expected location: {api_key_path.absolute()}\n"
                "Please create this file with your OpenAI API key or update "
                "'api_key_file' path in config/requirements.toml"
            )
        
        if gpt_output_dir.exists():
            import shutil
            shutil.rmtree(gpt_output_dir)
            print(f"   ✓ Removed existing files in {gpt_output_dir}")
        gpt_output_dir.mkdir(exist_ok=True, parents=True)
        print(f"   ✓ Created fresh directory: {gpt_output_dir}")
        
        gpt = GPTModel(
            api_key_file=api_key_file,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            system_message=system_message
        )
        
        fallback_dir = base_dir / "xilo_xla" / "artifacts" / "Prompts" / "req"
        
        results = gpt.batch_generate_requirements(
            prompts=prompts,
            output_dir=gpt_output_dir,
            n_samples=n_samples,
            skip_existing=skip_existing,
            fallback_optimizations=fallback_optimizations,
            fallback_dir=str(fallback_dir),
            prefix_text=prefix_text
        )
        
    except Exception as e:
        print(f"\n✗ Error processing prompts through GPT: {e}")
        raise
    
    return prompts


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import sys
        sys.exit(1)

