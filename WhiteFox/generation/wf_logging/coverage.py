"""LLVM source-based coverage: collection, merging, reporting."""

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PATH_EQUIV = (
    "/proc/self/cwd,"
    "/vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_223995/"
    "a4a32a9063034e2db7bdf417555977d5/execroot/org_tensorflow"
)


def _find_tf_so_files() -> List[str]:
    """Find instrumented TensorFlow .so files in the current environment."""
    r = subprocess.run(
        [sys.executable, "-c", "import tensorflow as tf; print(tf.__file__)"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning("Could not locate TensorFlow: %s", r.stderr.strip())
        return []
    tf_dir = Path(r.stdout.strip()).parent
    return sorted(str(p) for p in tf_dir.rglob("*.so"))


class CoverageCollector:
    """One instance per WhiteFox run.  profraw → profdata → report."""

    def __init__(self, logging_dir: Path):
        self.cov_dir = logging_dir / "coverage"
        self.profraw_dir = self.cov_dir / "profraw"
        self.profraw_dir.mkdir(parents=True, exist_ok=True)
        self.profdata_file = self.cov_dir / "merged.profdata"
        self.report_file = logging_dir / "coverage_report.log"
        self._so_files: Optional[List[str]] = None
        self._lock = threading.Lock()

    # ---- collection --------------------------------------------------------

    def env_vars(self) -> dict:
        """Env vars that make each TF subprocess write a unique .profraw."""
        pattern = str(self.profraw_dir / "wf_%p_%m.profraw")
        return {"LLVM_PROFILE_FILE": pattern}

    # ---- merging ------------------------------------------------------------

    def merge(self) -> bool:
        profraw_files = sorted(self.profraw_dir.glob("*.profraw"))
        if not profraw_files:
            logger.warning("No .profraw files found in %s", self.profraw_dir)
            return False
        cmd = ["llvm-profdata", "merge", "-sparse", "-o", str(self.profdata_file)]
        cmd += [str(f) for f in profraw_files]
        logger.info(
            "Merging %d profraw files → %s", len(profraw_files), self.profdata_file
        )
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error("llvm-profdata merge failed: %s", r.stderr.strip())
            return False
        return True

    # ---- reporting ----------------------------------------------------------

    def _get_so_files(self) -> List[str]:
        if self._so_files is None:
            self._so_files = _find_tf_so_files()
        return self._so_files

    def report(self) -> bool:
        if not self.profdata_file.exists():
            logger.warning("No merged profdata; skipping report")
            return False
        so_files = self._get_so_files()
        if not so_files:
            logger.warning("No TensorFlow .so files found; skipping report")
            return False

        cmd = [
            "llvm-cov",
            "report",
            so_files[0],
            f"-instr-profile={self.profdata_file}",
            f"-path-equivalence={_PATH_EQUIV}",
        ]
        for so in so_files[1:]:
            cmd += ["-object", so]

        logger.info("Generating coverage report → %s", self.report_file)
        r = subprocess.run(cmd, capture_output=True, text=True)

        with open(self.report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LLVM SOURCE-BASED COVERAGE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Profdata: {self.profdata_file}\n")
            f.write(f"Objects:  {len(so_files)} TensorFlow .so files\n\n")
            if r.returncode == 0:
                f.write(r.stdout)
            else:
                f.write(f"llvm-cov report failed (exit {r.returncode}):\n")
                f.write(r.stderr)
            f.write("\n")

        if r.returncode != 0:
            logger.error("llvm-cov report failed: %s", r.stderr.strip())
            return False
        return True

    # ---- convenience --------------------------------------------------------

    def finalize(self) -> None:
        """Merge all profraw files so far and regenerate the report.

        Thread-safe; safe to call after every optimization.
        """
        with self._lock:
            if self.merge():
                self.report()
