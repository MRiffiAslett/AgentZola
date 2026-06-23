import sys
from pathlib import Path

from prompt_exec import GPTModel
from prompt_gen import generate_requirement_prompts

_SUT_DIR_MAP = {
    "xla": "xilo_xla",
    "inductor": "xilo_inductor",
    "tflite": "xilo_tflite",
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sut",
        type=str,
        choices=["xla", "inductor", "tflite"],
        default="xla",
    )
    parser.add_argument(
        "--source-version",
        type=str,
        default=None,
        help="Source code version tag (e.g. 20230507 or 20250806). "
             "Selects optimization_specification_{sut}-{version}.json and "
             "writes to requirement-prompts-{version}/ and generation-prompts-{version}/. "
             "Omit to use the default (unversioned) spec and output dirs.",
    )
    args = parser.parse_args()

    sut = args.sut
    source_version = args.source_version
    sut_dir_name = _SUT_DIR_MAP[sut]

    base_dir = Path(__file__).parent.parent

    config_file = base_dir / sut_dir_name / "config" / "requirements.toml"

    import tomllib

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    gpt_config = config.get("gpt", {})
    use_mini = gpt_config.get("use_mini", False)

    spec_suffix = f"-{source_version}" if source_version else ""
    dir_suffix = f"-{source_version}" if source_version else ""

    opt_spec_path = (
        base_dir / sut_dir_name / "artifacts" / f"optimization_specification_{sut}{spec_suffix}.json"
    )
    template_path = base_dir / sut_dir_name / "artifacts" / "prompt_template.txt"
    exemplar_desc_path = base_dir / sut_dir_name / "artifacts" / "exemplar_description.txt"
    prompts_dir = base_dir / sut_dir_name / "artifacts" / f"requirement-prompts{dir_suffix}"
    gpt_output_dir = base_dir / sut_dir_name / "artifacts" / f"generation-prompts{dir_suffix}"

    prompts, fallback_optimizations = generate_requirement_prompts(
        optpath=str(opt_spec_path),
        template_path=str(template_path),
        outdir=str(prompts_dir),
        use_mini=use_mini,
        exemplar_description_path=str(exemplar_desc_path),
    )

    try:
        api_key_file = gpt_config.get("api_key_file")
        model = gpt_config.get("model")
        temperature = gpt_config.get("temperature")
        max_tokens = gpt_config.get("max_tokens")
        timeout = gpt_config.get("timeout")
        n_samples = gpt_config.get("n_samples")
        skip_existing = gpt_config.get("skip_existing")
        system_message = gpt_config.get("system_message")

        if gpt_output_dir.exists():
            import shutil

            shutil.rmtree(gpt_output_dir)
        gpt_output_dir.mkdir(exist_ok=True, parents=True)

        gpt = GPTModel(
            api_key_file=api_key_file,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            system_message=system_message,
        )

        fallback_dir = (
            base_dir / sut_dir_name / "artifacts" / "whitefox_original" / "req"
        )

        results = gpt.batch_generate_requirements(
            prompts=prompts,
            output_dir=gpt_output_dir,
            n_samples=n_samples,
            skip_existing=skip_existing,
            fallback_optimizations=fallback_optimizations,
            fallback_dir=str(fallback_dir),
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
        sys.exit(1)
