"""LLVM source-based coverage: collection, merging, reporting."""

import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PATH_EQUIV = (
    "/proc/self/cwd,"
    "/vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_223995/"
    "a4a32a9063034e2db7bdf417555977d5/execroot/org_tensorflow"
)

_VERIFY_SCRIPT = """\
import os, sys
print("LLVM_PROFILE_FILE=" + os.environ.get("LLVM_PROFILE_FILE", "<unset>"))
import tensorflow as tf
tf.constant(1)
"""


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
        self.cov_dir.mkdir(parents=True, exist_ok=True)
        # Write profraw to /tmp (local disk, fast) — not the shared filesystem.
        self.profraw_dir = Path(tempfile.mkdtemp(prefix="wf_profraw_"))
        self.profdata_file = self.cov_dir / "merged.profdata"
        self.report_file = logging_dir / "coverage_report.log"
        self.diag_file = logging_dir / "coverage_diagnostics.log"
        self._so_files: Optional[List[str]] = None
        self._lock = threading.Lock()

    # ---- diagnostics -------------------------------------------------------

    def verify(self) -> None:
        """Run once at startup.  Logs whether the TF wheel actually produces
        profraw files and whether LLVM_PROFILE_FILE reaches subprocesses.
        Results go to coverage_diagnostics.log.
        """
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("COVERAGE DIAGNOSTICS")
        lines.append("=" * 70)

        # 1. Check env var propagation + profraw output in a real subprocess
        with tempfile.TemporaryDirectory() as tmp:
            probe = str(Path(tmp) / "probe_%p.profraw")
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = probe
            r = subprocess.run(
                [sys.executable, "-c", _VERIFY_SCRIPT],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            lines.append("")
            lines.append("--- subprocess stdout ---")
            lines.append(r.stdout.strip())
            if r.stderr:
                lines.append("--- subprocess stderr (last 5 lines) ---")
                for line in r.stderr.strip().splitlines()[-5:]:
                    lines.append(line)
            profraw_files = list(Path(tmp).glob("*.profraw"))
            lines.append("")
            lines.append(f"LLVM_PROFILE_FILE sent: {probe}")
            lines.append(f"profraw files written:  {len(profraw_files)}")
            for pf in profraw_files:
                lines.append(f"  {pf}  ({pf.stat().st_size} bytes)")
            if not profraw_files:
                lines.append(
                    "⚠  No profraw produced — the TF wheel may not be "
                    "instrumented with -fprofile-instr-generate / "
                    "-fcoverage-mapping."
                )
            else:
                lines.append("✓  TF wheel produces profraw files.")

        # 2. Check for __llvm_profile symbols in main TF .so
        so_files = self._get_so_files()
        lines.append("")
        lines.append(f"TF .so files found: {len(so_files)}")
        if so_files:
            first_so = so_files[0]
            nm = subprocess.run(
                ["nm", "-D", first_so],
                capture_output=True,
                text=True,
            )
            profile_syms = [
                sym for sym in nm.stdout.splitlines() if "__llvm_profile" in sym
            ]
            lines.append(
                f"__llvm_profile symbols in {Path(first_so).name}: {len(profile_syms)}"
            )
            for s in profile_syms[:10]:
                lines.append(f"  {s}")

        lines.append("")
        lines.append("=" * 70)

        text = "\n".join(lines) + "\n"
        self.diag_file.write_text(text)
        logger.info("Coverage diagnostics written to %s", self.diag_file)

        # Also log the key result to the main logger
        if not profraw_files:
            logger.warning(
                "COVERAGE: probe subprocess produced 0 profraw files. "
                "The TF wheel may not be instrumented."
            )
        else:
            logger.info(
                "COVERAGE: probe subprocess wrote %d profraw file(s) — "
                "instrumentation confirmed.",
                len(profraw_files),
            )

    # ---- collection --------------------------------------------------------

    def env_vars(self) -> dict:
        """Env vars that make each TF subprocess write a unique .profraw."""
        pattern = str(self.profraw_dir / "wf_%p.profraw")
        return {"LLVM_PROFILE_FILE": pattern}

    # ---- merging ------------------------------------------------------------

    _MERGE_BATCH = 3  # profraw files per llvm-profdata invocation (~3GB peak)

    def merge(self) -> bool:
        profraw_files = sorted(self.profraw_dir.glob("*.profraw"))
        if not profraw_files:
            logger.warning("No .profraw files found in %s", self.profraw_dir)
            return False

        logger.info(
            "Merging %d profraw files (batch size %d) → %s",
            len(profraw_files),
            self._MERGE_BATCH,
            self.profdata_file,
        )

        # Merge in small batches to avoid OOM.  Each batch folds new profraw
        # files into the single accumulated profdata on disk.
        # Write to a temp file then rename to avoid truncating the input
        # profdata that is also our output target.
        tmp_profdata = self.profdata_file.with_suffix(".profdata.tmp")

        for i in range(0, len(profraw_files), self._MERGE_BATCH):
            batch = profraw_files[i : i + self._MERGE_BATCH]
            inputs = [str(f) for f in batch]
            if self.profdata_file.exists():
                inputs.insert(0, str(self.profdata_file))

            cmd = [
                "llvm-profdata",
                "merge",
                "-sparse",
                "-o",
                str(tmp_profdata),
            ] + inputs

            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("llvm-profdata merge failed: %s", r.stderr.strip())
                tmp_profdata.unlink(missing_ok=True)
                return False

            tmp_profdata.rename(self.profdata_file)

            # Delete this batch's profraw files immediately.
            for pf in batch:
                pf.unlink(missing_ok=True)

        logger.info("Merged and cleaned %d profraw files", len(profraw_files))
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
