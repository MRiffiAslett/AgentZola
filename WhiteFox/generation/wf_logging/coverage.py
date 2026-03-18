"""LLVM source-based coverage: collection, merging, reporting."""

import glob
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


_TFBUILD_LLVM_GLOB = (
    "/vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_*/*/external/"
    "llvm_linux_x86_64/bin"
)


def _llvm_tool_version(tool_path: str) -> str:
    """Return the version string reported by an LLVM tool, or '?' on error."""
    try:
        r = subprocess.run(
            [tool_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in r.stdout.splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "?"


def _find_all_llvm_tools(name: str) -> List[str]:
    """Return *all* candidate paths for an LLVM tool, de-duplicated.

    Search order (highest priority first):
    1. WHITEFOX_LLVM_DIR environment variable
    2. Anything already on PATH
    3. Versioned variants in /usr/bin
    4. LLVM bundled inside the TF build tree
    """
    candidates: List[str] = []
    seen: set = set()

    def _add(p: str) -> None:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            candidates.append(p)

    # 0. Explicit override via env var (highest priority)
    llvm_dir = os.environ.get("WHITEFOX_LLVM_DIR")
    if llvm_dir:
        p = os.path.join(llvm_dir, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _add(p)

    # 1. Anything already on PATH
    path = shutil.which(name)
    if path:
        _add(path)

    # 2. Versioned variants in /usr/bin  (e.g. llvm-profdata-17)
    for c in sorted(glob.glob(f"/usr/bin/{name}-*"), reverse=True):
        if shutil.which(c):
            _add(c)

    # 3. LLVM bundled inside the TF build tree (one per bazel root)
    for llvm_bin in sorted(glob.glob(_TFBUILD_LLVM_GLOB)):
        p = os.path.join(llvm_bin, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _add(p)

    return candidates


def _find_llvm_tool(name: str) -> str:
    """Best-effort single-tool lookup (used for quick calls like nm)."""
    candidates = _find_all_llvm_tools(name)
    return candidates[0] if candidates else name

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
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = "/dev/null"  # suppress ~1GB profraw write
    r = subprocess.run(
        [sys.executable, "-c", "import tensorflow as tf; print(tf.__file__)"],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        logger.warning("Could not locate TensorFlow: %s", r.stderr.strip())
        return []
    tf_dir = Path(r.stdout.strip()).parent
    so_files = sorted(str(p) for p in tf_dir.rglob("*.so"))
    logger.info("Found %d TF .so files under %s", len(so_files), tf_dir)
    return so_files


class CoverageCollector:
    """One instance per WhiteFox run.  profraw → profdata → report."""

    def __init__(self, logging_dir: Path):
        self.cov_dir = logging_dir / "coverage"
        self.cov_dir.mkdir(parents=True, exist_ok=True)
        # Write profraw to local /tmp (fast) — not the shared filesystem.
        # tempfile.mkdtemp() honours TMPDIR which on SLURM clusters often
        # points to a shared mount, so we force /tmp explicitly.
        self.profraw_dir = Path(tempfile.mkdtemp(prefix="wf_profraw_", dir="/tmp"))
        self.profdata_file = self.cov_dir / "merged.profdata"
        self.report_file = logging_dir / "coverage_report.log"
        self.diag_file = logging_dir / "coverage_diagnostics.log"
        self._so_files: Optional[List[str]] = None
        self._lock = threading.Lock()
        self._llvm_dir: Optional[str] = None  # cached compatible LLVM bin dir

        logger.info(
            "CoverageCollector: profraw_dir=%s, profdata=%s",
            self.profraw_dir, self.profdata_file,
        )

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
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
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

            # 2. Probe llvm-profdata compatibility while profraw still exists
            if profraw_files:
                llvm_dir_env = os.environ.get("WHITEFOX_LLVM_DIR", "")
                lines.append(
                    f"WHITEFOX_LLVM_DIR: {llvm_dir_env or '<not set>'}"
                )
                candidates = _find_all_llvm_tools("llvm-profdata")
                lines.append(f"llvm-profdata candidates found: {len(candidates)}")
                for c in candidates:
                    ver = _llvm_tool_version(c)
                    lines.append(f"  {c}  ({ver})")
                compat = self._select_profdata(profraw_files[0])
                if compat:
                    lines.append(f"✓  Compatible llvm-profdata: {compat}")
                else:
                    lines.append(
                        "⚠  No compatible llvm-profdata found — coverage "
                        "merge/report will be skipped."
                    )
                    lines.append(
                        "   Profraw v8 needs LLVM 15-17. Set WHITEFOX_LLVM_DIR "
                        "to a directory containing a compatible llvm-profdata."
                    )
                lines.append("")

        # 3. Check for __llvm_profile symbols in main TF .so
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

    # ---- LLVM tool selection ------------------------------------------------

    @staticmethod
    def _profraw_version(profraw_path: Path) -> Optional[int]:
        """Read the raw profile format version from the profraw header.

        The header layout (little-endian):
          bytes  0-7 : magic
          bytes  8-15: version (u64, low byte = format version)
        """
        try:
            with open(profraw_path, "rb") as f:
                header = f.read(16)
            if len(header) < 16:
                return None
            ver_u64 = struct.unpack("<Q", header[8:16])[0]
            return ver_u64 & 0xFF
        except Exception as exc:
            logger.warning("Could not read profraw header: %s", exc)
            return None

    def _select_profdata(self, sample_profraw: Path) -> Optional[str]:
        """Find a compatible llvm-profdata for the given profraw file.

        Strategy:
        1. Read the profraw format version from the binary header.
        2. For each candidate tool, check its LLVM major version (instant).
           - Profraw v8 ↔ LLVM 15-17, profraw v9 ↔ LLVM 18+
        3. Fall back to running ``llvm-profdata show`` if the heuristic
           is inconclusive.
        Caches the result so subsequent calls are instant.
        """
        if self._llvm_dir is not None:
            tool = os.path.join(self._llvm_dir, "llvm-profdata")
            if os.path.isfile(tool):
                return tool

        candidates = _find_all_llvm_tools("llvm-profdata")
        if not candidates:
            logger.error("No llvm-profdata binary found anywhere")
            return None

        profraw_ver = self._profraw_version(sample_profraw)
        logger.info(
            "Profraw format version: %s (from %s)",
            profraw_ver, sample_profraw.name,
        )

        # Map profraw versions to compatible LLVM major-version ranges.
        _COMPAT = {
            8: range(15, 18),   # LLVM 15, 16, 17
            9: range(18, 30),   # LLVM 18+
        }
        wanted = _COMPAT.get(profraw_ver)

        for tool in candidates:
            ver_str = _llvm_tool_version(tool)
            # Extract major version from strings like "LLVM version 17.0.6"
            major = None
            m = re.search(r"(\d+)\.\d+", ver_str)
            if m:
                major = int(m.group(1))

            # Fast path: version-number heuristic
            if wanted is not None and major is not None:
                if major in wanted:
                    self._llvm_dir = os.path.dirname(os.path.realpath(tool))
                    logger.info(
                        "Selected %s (LLVM %d matches profraw v%d)",
                        tool, major, profraw_ver,
                    )
                    return tool
                logger.info(
                    "Skipping %s (LLVM %d incompatible with profraw v%d)",
                    tool, major, profraw_ver,
                )
                continue

            # Slow path: actually run the tool against the profraw.
            logger.info(
                "Probing %s against profraw (version heuristic unavailable)",
                tool,
            )
            try:
                r = subprocess.run(
                    [tool, "show", str(sample_profraw)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception as exc:
                logger.warning("Skipping %s (probe error: %s)", tool, exc)
                continue
            if "version mismatch" in r.stderr:
                logger.info("Skipping %s (profraw version mismatch)", tool)
                continue
            if r.returncode != 0:
                logger.warning(
                    "Skipping %s (probe exit %d: %s)",
                    tool, r.returncode, r.stderr.strip()[:200],
                )
                continue
            self._llvm_dir = os.path.dirname(os.path.realpath(tool))
            logger.info("Selected compatible llvm-profdata: %s", tool)
            return tool

        logger.error(
            "None of the %d llvm-profdata candidates are compatible "
            "with profraw v%s produced by this TF wheel.",
            len(candidates), profraw_ver,
        )
        return None

    def _get_llvm_tool(self, name: str) -> str:
        """Return *name* from the cached compatible LLVM directory."""
        if self._llvm_dir:
            p = os.path.join(self._llvm_dir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return _find_llvm_tool(name)

    # ---- merging ------------------------------------------------------------

    _MERGE_BATCH = 3  # profraw files per llvm-profdata invocation (~3GB peak)

    def merge(self) -> bool:
        profraw_files = sorted(self.profraw_dir.glob("*.profraw"))
        if not profraw_files:
            logger.warning("No .profraw files found in %s", self.profraw_dir)
            return False

        profdata_tool = self._select_profdata(profraw_files[0])
        if profdata_tool is None:
            return False

        logger.info(
            "Merging %d profraw files (batch size %d) → %s",
            len(profraw_files),
            self._MERGE_BATCH,
            self.profdata_file,
        )

        tmp_profdata = self.profdata_file.with_suffix(".profdata.tmp")
        total_batches = (len(profraw_files) + self._MERGE_BATCH - 1) // self._MERGE_BATCH

        for i in range(0, len(profraw_files), self._MERGE_BATCH):
            batch = profraw_files[i : i + self._MERGE_BATCH]
            batch_num = i // self._MERGE_BATCH + 1
            batch_sizes = [f.stat().st_size / 1024 / 1024 for f in batch]
            logger.info(
                "  merge batch %d/%d: %d files (%.1f MB total)",
                batch_num, total_batches, len(batch), sum(batch_sizes),
            )

            inputs = [str(f) for f in batch]
            if self.profdata_file.exists():
                inputs.insert(0, str(self.profdata_file))

            cmd = [
                profdata_tool,
                "merge",
                "-sparse",
                "-o",
                str(tmp_profdata),
            ] + inputs

            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                logger.error("llvm-profdata merge timed out (batch %d)", batch_num)
                tmp_profdata.unlink(missing_ok=True)
                return False
            except FileNotFoundError:
                logger.error("llvm-profdata not found: %s", profdata_tool)
                return False
            if r.returncode != 0:
                logger.error("llvm-profdata merge failed: %s", r.stderr.strip())
                tmp_profdata.unlink(missing_ok=True)
                return False

            tmp_profdata.rename(self.profdata_file)

            for pf in batch:
                pf.unlink(missing_ok=True)

        profdata_size = self.profdata_file.stat().st_size / 1024 / 1024
        logger.info(
            "Merged %d profraw files → %s (%.1f MB)",
            len(profraw_files), self.profdata_file, profdata_size,
        )
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
            self._get_llvm_tool("llvm-cov"),
            "report",
            so_files[0],
            f"-instr-profile={self.profdata_file}",
            f"-path-equivalence={_PATH_EQUIV}",
        ]
        for so in so_files[1:]:
            cmd += ["-object", so]

        logger.info("Generating coverage report → %s", self.report_file)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("llvm-cov report timed out")
            return False
        except FileNotFoundError:
            logger.error("llvm-cov not found on PATH")
            return False

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

        report_size = self.report_file.stat().st_size
        logger.info(
            "Coverage report written: %s (%d bytes)", self.report_file, report_size,
        )
        return True

    # ---- convenience --------------------------------------------------------

    def finalize(self) -> None:
        """Merge all profraw files so far and regenerate the report.

        Thread-safe; safe to call after every optimization.
        Non-fatal: logs errors but never crashes the fuzzing run.
        """
        logger.info("Coverage finalize starting …")
        try:
            with self._lock:
                merged = self.merge()
                if merged:
                    self.report()
                else:
                    logger.warning("Coverage merge returned False — skipping report")
        except Exception as exc:
            logger.error("Coverage finalize failed: %s", exc, exc_info=True)
        logger.info("Coverage finalize done.")
