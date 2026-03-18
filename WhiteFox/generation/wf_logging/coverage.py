"""LLVM source-based coverage: collect profraw, merge, report XLA lines hit."""

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
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLVM tool discovery
# ---------------------------------------------------------------------------

_TFBUILD_LLVM_GLOB = (
    "/vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_*/*/external/"
    "llvm_linux_x86_64/bin"
)

# XLA source paths contain this substring inside the bazel execroot.
# Matches both old (tensorflow/compiler/xla/) and new (third_party/xla/xla/).
_XLA_PATH_MARKERS = ("/xla/service/", "/xla/hlo/")


def _llvm_tool_version(tool_path: str) -> str:
    """Return the version string reported by an LLVM tool, or '?' on error."""
    try:
        r = subprocess.run(
            [tool_path, "--version"],
            capture_output=True, text=True, timeout=5,
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

    llvm_dir = os.environ.get("WHITEFOX_LLVM_DIR")
    if llvm_dir:
        p = os.path.join(llvm_dir, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _add(p)

    path = shutil.which(name)
    if path:
        _add(path)

    for c in sorted(glob.glob(f"/usr/bin/{name}-*"), reverse=True):
        if shutil.which(c):
            _add(c)

    for llvm_bin in sorted(glob.glob(_TFBUILD_LLVM_GLOB)):
        p = os.path.join(llvm_bin, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _add(p)

    return candidates


def _find_llvm_tool(name: str) -> str:
    candidates = _find_all_llvm_tools(name)
    return candidates[0] if candidates else name


def _find_tf_so_files() -> List[str]:
    """Find TensorFlow .so files in the current environment."""
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = "/dev/null"
    r = subprocess.run(
        [sys.executable, "-c", "import tensorflow as tf; print(tf.__file__)"],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        logger.warning("Could not locate TensorFlow: %s", r.stderr.strip())
        return []
    tf_dir = Path(r.stdout.strip()).parent
    so_files = sorted(str(p) for p in tf_dir.rglob("*.so"))
    logger.info("Found %d TF .so files under %s", len(so_files), tf_dir)
    return so_files


# ---------------------------------------------------------------------------
# Verify script – checks that the TF wheel produces profraw files.
# ---------------------------------------------------------------------------

_VERIFY_SCRIPT = """\
import os, sys
print("LLVM_PROFILE_FILE=" + os.environ.get("LLVM_PROFILE_FILE", "<unset>"))
import tensorflow as tf
tf.constant(1)
"""


# ---------------------------------------------------------------------------
# CoverageCollector
# ---------------------------------------------------------------------------

class CoverageCollector:
    """Collect profraw files, merge them, and report XLA lines hit."""

    def __init__(self, logging_dir: Path):
        self.cov_dir = logging_dir / "coverage"
        self.cov_dir.mkdir(parents=True, exist_ok=True)
        self.profraw_dir = Path(tempfile.mkdtemp(prefix="wf_profraw_", dir="/tmp"))
        self.profdata_file = self.cov_dir / "merged.profdata"
        self.report_file = logging_dir / "coverage_report.log"
        self.diag_file = logging_dir / "coverage_diagnostics.log"
        self._so_files: Optional[List[str]] = None
        self._lock = threading.Lock()
        self._llvm_dir: Optional[str] = None

        logger.info(
            "CoverageCollector: profraw_dir=%s, profdata=%s",
            self.profraw_dir, self.profdata_file,
        )

    # ---- env ---------------------------------------------------------------

    def env_vars(self) -> dict:
        """Env vars that make each TF subprocess write a unique .profraw."""
        return {"LLVM_PROFILE_FILE": str(self.profraw_dir / "wf_%p.profraw")}

    # ---- diagnostics -------------------------------------------------------

    def verify(self) -> None:
        """Run once at startup. Writes coverage_diagnostics.log."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("COVERAGE DIAGNOSTICS")
        lines.append("=" * 70)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            probe = str(Path(tmp) / "probe_%p.profraw")
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = probe
            r = subprocess.run(
                [sys.executable, "-c", _VERIFY_SCRIPT],
                capture_output=True, text=True, env=env, timeout=120,
            )
            lines.append("")
            lines.append("--- subprocess stdout ---")
            lines.append(r.stdout.strip())
            if r.stderr:
                lines.append("--- subprocess stderr (last 3 lines) ---")
                for line in r.stderr.strip().splitlines()[-3:]:
                    lines.append(line)

            profraw_files = list(Path(tmp).glob("*.profraw"))
            lines.append("")
            lines.append(f"profraw files written: {len(profraw_files)}")
            for pf in profraw_files:
                lines.append(f"  {pf}  ({pf.stat().st_size} bytes)")

            if not profraw_files:
                lines.append("⚠  No profraw produced — TF wheel may not be instrumented.")
            else:
                lines.append("✓  TF wheel produces profraw files.")

            # Probe llvm-profdata compatibility while profraw still exists
            if profraw_files:
                llvm_dir_env = os.environ.get("WHITEFOX_LLVM_DIR", "")
                lines.append(f"WHITEFOX_LLVM_DIR: {llvm_dir_env or '<not set>'}")
                candidates = _find_all_llvm_tools("llvm-profdata")
                lines.append(f"llvm-profdata candidates: {len(candidates)}")
                for c in candidates:
                    lines.append(f"  {c}  ({_llvm_tool_version(c)})")
                compat = self._select_profdata(profraw_files[0])
                if compat:
                    lines.append(f"✓  Compatible llvm-profdata: {compat}")
                else:
                    lines.append("⚠  No compatible llvm-profdata found.")
                    lines.append("   Set WHITEFOX_LLVM_DIR to a directory with LLVM 15-17 tools.")

        lines.append("")
        lines.append("=" * 70)

        self.diag_file.write_text("\n".join(lines) + "\n")
        logger.info("Coverage diagnostics written to %s", self.diag_file)

        if not profraw_files:
            logger.warning("COVERAGE: probe produced 0 profraw files.")
        else:
            logger.info(
                "COVERAGE: probe wrote %d profraw file(s) — instrumentation confirmed.",
                len(profraw_files),
            )

    # ---- LLVM tool selection ------------------------------------------------

    @staticmethod
    def _profraw_version(profraw_path: Path) -> Optional[int]:
        """Read the raw profile format version from the profraw binary header."""
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
        """Find a compatible llvm-profdata by matching profraw version to LLVM version."""
        if self._llvm_dir is not None:
            tool = os.path.join(self._llvm_dir, "llvm-profdata")
            if os.path.isfile(tool):
                return tool

        candidates = _find_all_llvm_tools("llvm-profdata")
        if not candidates:
            logger.error("No llvm-profdata binary found anywhere")
            return None

        profraw_ver = self._profraw_version(sample_profraw)
        logger.info("Profraw format version: %s (from %s)", profraw_ver, sample_profraw.name)

        _COMPAT = {8: range(15, 18), 9: range(18, 30)}
        wanted = _COMPAT.get(profraw_ver)

        for tool in candidates:
            ver_str = _llvm_tool_version(tool)
            major = None
            m = re.search(r"(\d+)\.\d+", ver_str)
            if m:
                major = int(m.group(1))

            if wanted is not None and major is not None:
                if major in wanted:
                    self._llvm_dir = os.path.dirname(os.path.realpath(tool))
                    logger.info("Selected %s (LLVM %d matches profraw v%d)", tool, major, profraw_ver)
                    return tool
                logger.info("Skipping %s (LLVM %d incompatible with profraw v%d)", tool, major, profraw_ver)
                continue

            # Slow fallback: probe the tool against the profraw
            logger.info("Probing %s (version heuristic unavailable)", tool)
            try:
                r = subprocess.run(
                    [tool, "show", str(sample_profraw)],
                    capture_output=True, text=True, timeout=120,
                )
            except Exception as exc:
                logger.warning("Skipping %s (probe error: %s)", tool, exc)
                continue
            if "version mismatch" in r.stderr:
                logger.info("Skipping %s (profraw version mismatch)", tool)
                continue
            if r.returncode != 0:
                logger.warning("Skipping %s (exit %d: %s)", tool, r.returncode, r.stderr.strip()[:200])
                continue
            self._llvm_dir = os.path.dirname(os.path.realpath(tool))
            logger.info("Selected compatible llvm-profdata: %s", tool)
            return tool

        logger.error(
            "None of the %d llvm-profdata candidates are compatible with profraw v%s.",
            len(candidates), profraw_ver,
        )
        return None

    def _get_llvm_tool(self, name: str) -> str:
        if self._llvm_dir:
            p = os.path.join(self._llvm_dir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return _find_llvm_tool(name)

    # ---- merging ------------------------------------------------------------

    _MERGE_BATCH = 3

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
            len(profraw_files), self._MERGE_BATCH, self.profdata_file,
        )

        tmp_profdata = self.profdata_file.with_suffix(".profdata.tmp")
        total_batches = (len(profraw_files) + self._MERGE_BATCH - 1) // self._MERGE_BATCH

        for i in range(0, len(profraw_files), self._MERGE_BATCH):
            batch = profraw_files[i : i + self._MERGE_BATCH]
            batch_num = i // self._MERGE_BATCH + 1
            batch_mb = sum(f.stat().st_size for f in batch) / 1024 / 1024
            logger.info("  merge batch %d/%d: %d files (%.1f MB)", batch_num, total_batches, len(batch), batch_mb)

            inputs = [str(f) for f in batch]
            if self.profdata_file.exists():
                inputs.insert(0, str(self.profdata_file))

            cmd = [profdata_tool, "merge", "-sparse", "-o", str(tmp_profdata)] + inputs
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

        profdata_mb = self.profdata_file.stat().st_size / 1024 / 1024
        logger.info("Merged %d profraw files → %s (%.1f MB)", len(profraw_files), self.profdata_file, profdata_mb)
        return True

    # ---- reporting ----------------------------------------------------------

    def _get_so_files(self) -> List[str]:
        if self._so_files is None:
            self._so_files = _find_tf_so_files()
        return self._so_files

    @staticmethod
    def _parse_report_line(line: str) -> Optional[Tuple[str, int, int]]:
        """Parse one llvm-cov report line → (filename, total_lines, missed_lines) or None."""
        parts = line.split()
        if len(parts) < 10:
            return None
        filename = parts[0]
        try:
            total_lines = int(parts[-4])
            missed_lines = int(parts[-3])
            return (filename, total_lines, missed_lines)
        except (ValueError, IndexError):
            return None

    def report(self) -> Optional[Dict[str, int]]:
        """Run llvm-cov report, parse output, return XLA lines hit.

        Writes a concise coverage_report.log with the XLA line count.
        Returns dict with keys: xla_lines_hit, xla_lines_total, all_lines_hit, all_lines_total
        or None on failure.
        """
        if not self.profdata_file.exists():
            logger.warning("No merged profdata; skipping report")
            return None
        so_files = self._get_so_files()
        if not so_files:
            logger.warning("No TensorFlow .so files found; skipping report")
            return None

        cmd = [
            self._get_llvm_tool("llvm-cov"), "report",
            so_files[0],
            f"-instr-profile={self.profdata_file}",
        ]
        for so in so_files[1:]:
            cmd += ["-object", so]

        logger.info("Running llvm-cov report …")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("llvm-cov report timed out")
            return None
        except FileNotFoundError:
            logger.error("llvm-cov not found")
            return None

        if r.returncode != 0:
            logger.error("llvm-cov report failed (exit %d): %s", r.returncode, r.stderr.strip()[:500])
            return None

        # Parse the report: extract per-file line counts
        xla_lines_total = 0
        xla_lines_missed = 0
        all_lines_total = 0
        all_lines_missed = 0
        xla_files = 0
        total_files = 0

        for line in r.stdout.splitlines():
            parsed = self._parse_report_line(line)
            if parsed is None:
                continue
            filename, total, missed = parsed
            if filename == "TOTAL":
                all_lines_total = total
                all_lines_missed = missed
                continue
            total_files += 1
            if any(marker in filename for marker in _XLA_PATH_MARKERS):
                xla_files += 1
                xla_lines_total += total
                xla_lines_missed += missed

        xla_lines_hit = xla_lines_total - xla_lines_missed
        all_lines_hit = all_lines_total - all_lines_missed

        result = {
            "xla_lines_hit": xla_lines_hit,
            "xla_lines_total": xla_lines_total,
            "all_lines_hit": all_lines_hit,
            "all_lines_total": all_lines_total,
        }

        xla_pct = (xla_lines_hit / xla_lines_total * 100) if xla_lines_total else 0

        with open(self.report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("XLA COVERAGE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"XLA lines hit:    {xla_lines_hit:>8,}\n")
            f.write(f"XLA lines total:  {xla_lines_total:>8,}\n")
            f.write(f"XLA coverage:     {xla_pct:>7.2f}%\n")
            f.write(f"XLA source files: {xla_files:>8,}\n")
            f.write(f"\n")
            f.write(f"--- for reference ---\n")
            f.write(f"All TF lines hit:   {all_lines_hit:>8,}\n")
            f.write(f"All TF lines total: {all_lines_total:>8,}\n")
            f.write(f"All TF files:       {total_files:>8,}\n")
            f.write(f"\n")
            f.write(f"XLA path filters: {_XLA_PATH_MARKERS}\n")
            f.write("=" * 60 + "\n")

        logger.info(
            "XLA coverage: %d / %d lines hit (%.2f%%) across %d XLA files",
            xla_lines_hit, xla_lines_total, xla_pct, xla_files,
        )
        return result

    # ---- convenience --------------------------------------------------------

    def finalize(self) -> Optional[Dict[str, int]]:
        """Merge profraw files and generate the XLA coverage report.

        Returns the coverage dict or None on failure.
        """
        logger.info("Coverage finalize starting …")
        result = None
        try:
            with self._lock:
                if self.merge():
                    result = self.report()
                else:
                    logger.warning("Coverage merge returned False — skipping report")
        except Exception as exc:
            logger.error("Coverage finalize failed: %s", exc, exc_info=True)
        logger.info("Coverage finalize done.")
        return result
