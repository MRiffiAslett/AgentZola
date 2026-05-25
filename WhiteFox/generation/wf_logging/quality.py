"""Classify generated tests by execution quality (pre-oracle)."""

import ast
from typing import Any, Dict, List, Optional

_TIMEOUT_MARKERS = ("Execution timeout",)

_XLA_UNSUPPORTED_MARKERS = (
    "UnimplementedError",
    "XlaCompilationError",
    "is not supported in XLA",
    "not supported by XLA",
    "cannot be compiled by XLA",
    "Couldn't compile XLA",
    "XLA compilation failed",
    "NotImplementedError: XLA",
)

_INVALID_TF_API_MARKERS = (
    "AttributeError",
    "has no attribute",
    "module 'tensorflow' has no attribute",
    "InvalidArgumentError",
    "TypeError:",
    "ValueError:",
    "got an unexpected keyword argument",
    "No module named",
    "ImportError",
    "NameError:",
)

FUNNEL_KEYS = (
    "syntax_valid",
    "imports_successfully",
    "eager_executable",
    "xla_compilable",
)

FAILURE_KEYS = (
    "invalid_tf_api",
    "unsupported_by_xla",
    "timeout",
)

QUALITY_TABLE_ROWS = (
    ("syntax_valid", "syntactically valid Python"),
    ("imports_successfully", "imports successfully"),
    ("eager_executable", "eager executable"),
    ("xla_compilable", "XLA compilable"),
    ("invalid_tf_api", "invalid TensorFlow API usage"),
    ("unsupported_by_xla", "unsupported by XLA"),
    ("timeout", "timeout"),
)


def check_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _mode_error(result: Any, mode: str) -> str:
    mr = result.get_mode(mode)
    return mr.runtime_error or mr.compile_error or ""


def _all_errors(result: Any) -> str:
    return "\n".join(
        err for mode in result.modes for err in (_mode_error(result, mode),) if err
    )


def _is_timeout(error_text: str) -> bool:
    return any(marker in error_text for marker in _TIMEOUT_MARKERS)


def _is_unsupported_by_xla(error_text: str) -> bool:
    return any(marker in error_text for marker in _XLA_UNSUPPORTED_MARKERS)


def _is_invalid_tf_api(error_text: str) -> bool:
    return any(marker in error_text for marker in _INVALID_TF_API_MARKERS)


def _primary_eager_mode(modes: List[str]) -> Optional[str]:
    for name in ("naive", "eager", "cpu"):
        if name in modes:
            return name
    return modes[0] if modes else None


def _primary_xla_mode(modes: List[str]) -> Optional[str]:
    for name in ("xla", "jit", "compiled"):
        if name in modes:
            return name
    return None


def classify_generation_quality(
    test_code: Optional[str],
    result: Optional[Any] = None,
    worker_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Return funnel flags and a primary failure label for one generated test."""
    quality: Dict[str, Any] = {
        "syntax_valid": False,
        "imports_successfully": False,
        "eager_executable": False,
        "xla_compilable": False,
        "invalid_tf_api": False,
        "unsupported_by_xla": False,
        "timeout": False,
        "worker_failed": False,
        "primary_failure": None,
    }

    if worker_error:
        quality["worker_failed"] = True
        quality["primary_failure"] = "worker_failed"
        if _is_timeout(worker_error):
            quality["timeout"] = True
            quality["primary_failure"] = "timeout"
        return quality

    if test_code is not None:
        quality["syntax_valid"] = check_syntax_valid(test_code)

    if result is None:
        if test_code is not None and not quality["syntax_valid"]:
            quality["primary_failure"] = "syntax_invalid"
        return quality

    errors = _all_errors(result)
    if _is_timeout(errors):
        quality["timeout"] = True
        quality["primary_failure"] = "timeout"
        return quality

    modes = result.modes
    eager_mode = _primary_eager_mode(modes)
    xla_mode = _primary_xla_mode(modes)

    if eager_mode:
        eager_mr = result.get_mode(eager_mode)
        quality["imports_successfully"] = eager_mr.compile_success
        quality["eager_executable"] = eager_mr.runtime_success

    if xla_mode:
        quality["xla_compilable"] = result.get_mode(xla_mode).compile_success

    if test_code is not None and not quality["syntax_valid"]:
        quality["primary_failure"] = "syntax_invalid"
        return quality

    if eager_mode and not quality["imports_successfully"]:
        eager_err = _mode_error(result, eager_mode)
        if _is_invalid_tf_api(eager_err):
            quality["invalid_tf_api"] = True
            quality["primary_failure"] = "invalid_tf_api"
        else:
            quality["primary_failure"] = "import_failed"
        return quality

    if eager_mode and not quality["eager_executable"]:
        quality["primary_failure"] = "eager_failed"
        return quality

    if xla_mode and not quality["xla_compilable"]:
        xla_err = _mode_error(result, xla_mode)
        if _is_unsupported_by_xla(xla_err):
            quality["unsupported_by_xla"] = True
            quality["primary_failure"] = "unsupported_by_xla"
        else:
            quality["primary_failure"] = "xla_compile_failed"
        return quality

    return quality


def default_quality_stats() -> Dict[str, int]:
    stats = {"executed": 0, "worker_failed": 0}
    for key in FUNNEL_KEYS:
        stats[key] = 0
    for key in FAILURE_KEYS:
        stats[key] = 0
    return stats


def accumulate_quality_stats(stats: Dict[str, int], quality: Dict[str, Any]) -> None:
    if quality.get("worker_failed"):
        stats["worker_failed"] += 1
    else:
        stats["executed"] += 1

    for key in FUNNEL_KEYS:
        if quality.get(key):
            stats[key] += 1
    for key in FAILURE_KEYS:
        if quality.get(key):
            stats[key] += 1
