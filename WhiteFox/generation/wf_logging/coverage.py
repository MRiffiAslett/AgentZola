

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
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TFBUILD_LLVM_GLOB = (
    "/vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_*/*/external/"
    "llvm_linux_x86_64/bin"
)

_XLA_PATH_MARKERS = (
    "/xla/",
    "xla/",
    "/tensorflow/compiler/xla/",
    "tensorflow/compiler/xla/",
    "/third_party/xla/",
    "third_party/xla/",
)


def _llvm_tool_version(tool_path: str) -> str:
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


_VERIFY_SCRIPT = """\
import os, sys
import tensorflow as tf
tf.constant(1)
"""


class CoverageCollector:
    """Incremental profraw → merged.profdata; final llvm-cov XLA line summary."""

    def __init__(self, logging_dir: Path):
        self.cov_dir = logging_dir / "coverage"
        self.cov_dir.mkdir(parents=True, exist_ok=True)
        self.profraw_dir = Path(tempfile.mkdtemp(prefix="wf_profraw_", dir="/tmp"))
        self.profdata_file = self.cov_dir / "merged.profdata"
        self.report_file = logging_dir / "coverage_report.log"
        self.diag_file = logging_dir / "coverage_diagnostics.log"
        self._so_files: Optional[List[str]] = None
        self._lock = threading.RLock()
        self._llvm_dir: Optional[str] = None
        self._profdata_tool: Optional[str] = None

        # Fresh merged profile for this WhiteFox run (avoid stacking onto old runs).
        self.profdata_file.unlink(missing_ok=True)
        tmp = self.profdata_file.with_suffix(".profdata.tmp")
        tmp.unlink(missing_ok=True)

        logger.info(
            "CoverageCollector: profraw_dir=%s, profdata=%s",
            self.profraw_dir, self.profdata_file,
        )

    def env_vars(self) -> dict:
        return {"LLVM_PROFILE_FILE": str(self.profraw_dir / "wf_%p.profraw")}

    def verify(self) -> None:
        """Quick check that TF writes profraw and llvm-profdata can read it."""
        lines: List[str] = ["COVERAGE DIAGNOSTICS", "=" * 50]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            probe = str(Path(tmp) / "probe_%p.profraw")
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = probe
            subprocess.run(
                [sys.executable, "-c", _VERIFY_SCRIPT],
                capture_output=True, text=True, env=env, timeout=120,
            )
            profraw_files = list(Path(tmp).glob("*.profraw"))
            lines.append(f"profraw files from probe: {len(profraw_files)}")
            if profraw_files:
                t = self._select_profdata(profraw_files[0])
                lines.append(f"llvm-profdata: {t or 'NOT FOUND'}")
            else:
                lines.append("TF wheel may not be instrumented (no profraw).")

        self.diag_file.write_text("\n".join(lines) + "\n")
        logger.info("Coverage diagnostics written to %s", self.diag_file)

    @staticmethod
    def _profraw_version(profraw_path: Path) -> Optional[int]:
        try:
            with open(profraw_path, "rb") as f:
                header = f.read(16)
            if len(header) < 16:
                return None
            ver_u64 = struct.unpack("<Q", header[8:16])[0]
            return ver_u64 & 0xFF
        except Exception:
            return None

    def _select_profdata(self, sample_profraw: Path) -> Optional[str]:
        if self._llvm_dir is not None:
            tool = os.path.join(self._llvm_dir, "llvm-profdata")
            if os.path.isfile(tool):
                return tool

        candidates = _find_all_llvm_tools("llvm-profdata")
        if not candidates:
            logger.error("No llvm-profdata binary found")
            return None

        profraw_ver = self._profraw_version(sample_profraw)
        _COMPAT = {8: range(15, 18), 9: range(18, 30)}
        wanted = _COMPAT.get(profraw_ver)

        for tool in candidates:
            ver_str = _llvm_tool_version(tool)
            major = None
            m = re.search(r"(\d+)\.\d+", ver_str)
            if m:
                major = int(m.group(1))

            if wanted is not None and major is not None and major in wanted:
                self._llvm_dir = os.path.dirname(os.path.realpath(tool))
                logger.info("Selected llvm-profdata: %s", tool)
                return tool

            try:
                r = subprocess.run(
                    [tool, "show", str(sample_profraw)],
                    capture_output=True, text=True, timeout=120,
                )
            except Exception as exc:
                logger.warning("Probe %s: %s", tool, exc)
                continue
            if "version mismatch" in r.stderr:
                continue
            if r.returncode == 0:
                self._llvm_dir = os.path.dirname(os.path.realpath(tool))
                logger.info("Selected llvm-profdata (probe): %s", tool)
                return tool

        logger.error("No compatible llvm-profdata for profraw v%s", profraw_ver)
        return None

    def _get_llvm_tool(self, name: str) -> str:
        if self._llvm_dir:
            p = os.path.join(self._llvm_dir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return _find_llvm_tool(name)

    @staticmethod
    def _is_valid_profraw_header(profraw_path: Path) -> bool:
        try:
            with open(profraw_path, "rb") as f:
                header = f.read(16)
            if len(header) < 16:
                return False
            ver_u64 = struct.unpack("<Q", header[8:16])[0]
            return (ver_u64 & 0xFF) > 0
        except Exception:
            return False

    def merge_pending(self) -> int:
        """Merge pending .profraw into merged.profdata, delete raw on success.

        Returns:
            Number of profraw files successfully merged.
        """
        with self._lock:
            pending = sorted(self.profraw_dir.glob("*.profraw"))
            if not pending:
                return 0

            if self._profdata_tool is None:
                self._profdata_tool = self._select_profdata(pending[0])
            profdata_tool = self._profdata_tool
            if not profdata_tool:
                return 0

            tmp_out = self.profdata_file.with_suffix(".profdata.tmp")

            valid: List[Path] = []
            for pf in pending:
                try:
                    if pf.stat().st_size == 0:
                        pf.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if not self._is_valid_profraw_header(pf):
                    logger.warning("Skipping unreadable profraw header: %s", pf.name)
                    continue
                valid.append(pf)

            if not valid:
                return 0

            tmp_out.unlink(missing_ok=True)

            # llvm-profdata merge can parallelize work internally; let it use CPUs.
            # Default: min(8, cpu_count). Override via WHITEFOX_LLVM_PROFDATA_JOBS.
            jobs_env = os.environ.get("WHITEFOX_LLVM_PROFDATA_JOBS")
            if jobs_env:
                try:
                    jobs = int(jobs_env)
                except ValueError:
                    jobs = 1
            else:
                jobs = min(8, multiprocessing.cpu_count() or 1)
            jobs = max(1, jobs)

            cmd: List[str] = [
                profdata_tool,
                "merge",
                "-sparse",
                "--failure-mode=all",
                "-j",
                str(jobs),
                "-o",
                str(tmp_out),
            ]
            if self.profdata_file.exists():
                cmd.append(str(self.profdata_file))
            cmd += [str(p) for p in valid]

            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
            except FileNotFoundError:
                logger.error("llvm-profdata not found: %s", profdata_tool)
                return 0
            except Exception as exc:
                logger.error("llvm-profdata merge error: %s", exc)
                tmp_out.unlink(missing_ok=True)
                return 0

            if r.returncode != 0:
                err = (r.stderr or "").strip()[:400]
                logger.warning(
                    "merge failed (rc=%d): %s",
                    r.returncode, err or "(no stderr)",
                )
                tmp_out.unlink(missing_ok=True)
                return 0

            if not tmp_out.exists() or tmp_out.stat().st_size == 0:
                logger.warning("merge produced no output")
                tmp_out.unlink(missing_ok=True)
                return 0

            tmp_out.replace(self.profdata_file)
            for pf in valid:
                pf.unlink(missing_ok=True)

            return len(valid)

    def report(self) -> Optional[Dict[str, int]]:
        if not self.profdata_file.exists():
            logger.warning("No merged profdata; skipping report")
            return None
        so_files = self._so_files
        if so_files is None:
            so_files = _find_tf_so_files()
            self._so_files = so_files
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
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        except FileNotFoundError:
            logger.error("llvm-cov not found")
            return None

        if r.returncode != 0:
            logger.error("llvm-cov report failed (exit %d): %s", r.returncode, r.stderr.strip()[:500])
            return None

        report_lines = r.stdout.splitlines()
        lines_col = 7
        for ln in report_lines:
            if "Filename" in ln and "Lines" in ln:
                tokens = ln.split()
                for i, tok in enumerate(tokens):
                    if tok.strip().lower().rstrip(":").startswith("lines"):
                        lines_col = i
                        break
                break

        def parse_line(line: str) -> Optional[Tuple[str, int, int]]:
            parts = line.split()
            if not parts:
                return None
            fn = parts[0]

            def _to_int(tok: str) -> Optional[int]:
                t = tok.replace(",", "").strip()
                return int(t) if t.isdigit() else None

            try:
                t, m = _to_int(parts[lines_col]), _to_int(parts[lines_col + 1])
                if t is not None and m is not None:
                    return (fn, t, m)
            except IndexError:
                pass
            ints: List[int] = []
            for tok in parts[1:]:
                v = _to_int(tok)
                if v is not None:
                    ints.append(v)
            if len(ints) >= 2:
                return (fn, ints[-2], ints[-1])
            return None

        xla_lines_total = 0
        xla_lines_missed = 0
        all_lines_total = 0
        all_lines_missed = 0
        xla_files = 0
        total_files = 0

        for line in report_lines:
            parsed = parse_line(line)
            if parsed is None:
                continue
            filename, total, missed = parsed
            if filename.startswith("TOTAL"):
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
        xla_pct = (xla_lines_hit / xla_lines_total * 100) if xla_lines_total else 0

        result = {
            "xla_lines_hit": xla_lines_hit,
            "xla_lines_total": xla_lines_total,
            "all_lines_hit": all_lines_hit,
            "all_lines_total": all_lines_total,
        }

        with open(self.report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("XLA COVERAGE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"XLA lines hit:    {xla_lines_hit:>8,}\n")
            f.write(f"XLA lines total:  {xla_lines_total:>8,}\n")
            f.write(f"XLA coverage:     {xla_pct:>7.2f}%\n")
            f.write(f"XLA source files: {xla_files:>8,}\n")
            f.write("\n--- for reference ---\n")
            f.write(f"All TF lines hit:   {all_lines_hit:>8,}\n")
            f.write(f"All TF lines total: {all_lines_total:>8,}\n")
            f.write(f"All TF files:       {total_files:>8,}\n")
            f.write(f"\nXLA path filters: {_XLA_PATH_MARKERS}\n")
            f.write("=" * 60 + "\n")

        logger.info(
            "XLA coverage: %d / %d lines hit (%.2f%%) across %d XLA files",
            xla_lines_hit, xla_lines_total, xla_pct, xla_files,
        )
        return result

    def _write_coverage_unavailable(self, detail: str) -> None:
        try:
            self.report_file.write_text(
                "=" * 60 + "\n"
                "COVERAGE UNAVAILABLE\n"
                "=" * 60 + "\n\n"
                f"{detail}\n"
            )
        except OSError as exc:
            logger.warning("Could not write coverage stub: %s", exc)

    def finalize(self) -> Optional[Dict[str, int]]:
        logger.info("Coverage finalize starting …")
        result = None
        try:
            self.merge_pending()
            if self.profdata_file.exists():
                result = self.report()
                if result is None:
                    self._write_coverage_unavailable("llvm-cov report failed. See logs.")
            else:
                logger.warning("No merged.profdata after final merge_pending — skipping report")
                self._write_coverage_unavailable("No merged profile (merge failed or no valid profraw).")
        except Exception as exc:
            logger.error("Coverage finalize failed: %s", exc, exc_info=True)
            self._write_coverage_unavailable(f"Coverage finalize error: {exc!r}")
        logger.info("Coverage finalize done.")
        return result
