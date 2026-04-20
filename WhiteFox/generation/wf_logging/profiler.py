import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import psutil


class WhiteFoxProfiler:
    def __init__(self, log_dir: Path, config: Optional[Dict] = None):
        self.log_dir = Path(log_dir)
        self.report_file = self.log_dir / "resource_profile.log"

        self.start_time = time.time()
        self.process = psutil.Process()

        self.config = config or {}
        self._calculate_estimates()

        self.peak_memory_mb = 0.0
        self.peak_child_count = 0
        self.samples: List[dict] = []
        self.segment_samples: List[dict] = []

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._segment_lock = threading.Lock()
        self._optimization_metadata: Dict[str, Dict[str, Any]] = {}

    def set_optimization_metadata(
        self, optimization_name: str, metadata: Dict[str, Any]
    ) -> None:
        """Attach extra metrics to be printed for the given optimization segment."""
        with self._segment_lock:
            self._optimization_metadata[optimization_name] = metadata

    def _calculate_estimates(self):
        gen_config = self.config.get("generation", {})

        self.parallel_test_workers = gen_config.get("parallel_test_workers", 4)
        self.parallel_optimizations = gen_config.get("parallel_optimizations", 2)

        llm_cpu_memory = 3.0 * 1024
        system_overhead = 2.0 * 1024

        per_process_total = (0.1 + 1.2 + 0.3 + 0.4) * 1024

        self.total_processes = self.parallel_test_workers * self.parallel_optimizations
        base_memory = llm_cpu_memory + system_overhead
        process_memory = self.total_processes * per_process_total
        peak_buffer = 4.0 * 1024

        self.estimated_peak_mb = base_memory + process_memory + peak_buffer

    def begin_run(self) -> None:
        """Truncate the report file and write the run header. Call before start_monitoring()."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("WHITEFOX RESOURCE PROFILE\n")
            f.write(
                "Per-optimization segments are appended as each optimization completes.\n"
            )
            f.write("A final run summary is appended at job end.\n")
            f.write("=" * 70 + "\n\n")

    def start_monitoring(self, interval: float = 5.0):
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self, interval: float):
        while self._monitoring:
            try:
                self._capture_snapshot()
                time.sleep(interval)
            except Exception:
                pass

    def _capture_snapshot(self):
        with self._lock:
            try:
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024

                children = self.process.children(recursive=True)
                child_count = len(children)

                if mem_mb > self.peak_memory_mb:
                    self.peak_memory_mb = mem_mb
                if child_count > self.peak_child_count:
                    self.peak_child_count = child_count

                sample = {
                    "time": time.time() - self.start_time,
                    "memory_mb": mem_mb,
                    "child_count": child_count,
                }
                self.samples.append(sample)
                self.segment_samples.append(dict(sample))
            except Exception:
                pass

    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._capture_snapshot()

    def append_optimization_segment(self, optimization_name: str) -> None:
        """Append a snapshot for one optimization, then resume monitoring."""
        with self._segment_lock:
            self.stop_monitoring()
            try:
                self._write_optimization_segment(optimization_name)
            finally:
                self.segment_samples.clear()
                self.start_monitoring(5.0)

    def _write_optimization_segment(self, optimization_name: str) -> None:
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write("-" * 70 + "\n")
            f.write(f"  OPTIMIZATION: {optimization_name}\n")
            f.write("-" * 70 + "\n")

            if not self.segment_samples:
                f.write("  (no samples in this segment)\n\n")
                return

            peak_mem = max(s["memory_mb"] for s in self.segment_samples)
            peak_child = max(s["child_count"] for s in self.segment_samples)
            avg_mem = sum(s["memory_mb"] for s in self.segment_samples) / len(
                self.segment_samples
            )
            avg_child = sum(s["child_count"] for s in self.segment_samples) / len(
                self.segment_samples
            )
            t0 = self.segment_samples[0]["time"]
            t1 = self.segment_samples[-1]["time"]
            duration = max(0.0, t1 - t0)

            f.write(f"  Segment wall time (sample span): {duration:.1f} s\n")
            f.write("  Memory (this segment):\n")
            f.write(f"    Peak RSS:     {peak_mem:8.1f} MB ({peak_mem / 1024:5.1f} GB)\n")
            f.write(f"    Average RSS:  {avg_mem:8.1f} MB ({avg_mem / 1024:5.1f} GB)\n")
            f.write("  Child processes (this segment):\n")
            f.write(f"    Peak:         {peak_child:3d}\n")
            f.write(f"    Average:      {avg_child:5.1f}\n")
            f.write(f"  Samples:       {len(self.segment_samples):4d}\n\n")

            meta = self._optimization_metadata.get(optimization_name)
            if meta:
                f.write("  Timing breakdown (wall-clock, summed across iterations):\n")
                f.write(
                    f"    LLM generation:   {meta.get('llm_time_s', 0.0):8.1f} s\n"
                )
                f.write(
                    f"    Test execution:   {meta.get('exec_time_s', 0.0):8.1f} s\n"
                )
                f.write(
                    f"    Coverage merge:   {meta.get('coverage_merge_time_s', 0.0):8.1f} s\n"
                )
                f.write(
                    f"    Coverage merged:  {meta.get('coverage_merged_profraw', 0):8d} profraw\n"
                )
                f.write(f"    Iterations:       {meta.get('iterations', 0):8d}\n")
                f.write(
                    f"    Samples executed: {meta.get('samples_executed', 0):8d}\n\n"
                )

    def generate_report(self):
        """Append the full-run summary (after all optimizations)."""
        self.stop_monitoring()

        elapsed = time.time() - self.start_time

        if self.samples:
            avg_memory = sum(s["memory_mb"] for s in self.samples) / len(self.samples)
            avg_children = sum(s["child_count"] for s in self.samples) / len(
                self.samples
            )
        else:
            avg_memory = 0
            avg_children = 0

        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("WHITEFOX RESOURCE PROFILE — FULL RUN TOTAL\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Total runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)\n\n"
            )

            f.write("-" * 70 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Parallel test workers:      {self.parallel_test_workers:3d}\n")
            f.write(f"  Parallel optimizations:     {self.parallel_optimizations:3d}\n")
            f.write(f"  Total concurrent processes: {self.total_processes:3d}\n\n")

            f.write("-" * 70 + "\n")
            f.write("MEMORY USAGE (Estimated vs Actual)\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"  Estimated peak:  {self.estimated_peak_mb:8.1f} MB ({self.estimated_peak_mb / 1024:5.1f} GB)\n"
            )
            f.write(
                f"  Actual peak:     {self.peak_memory_mb:8.1f} MB ({self.peak_memory_mb / 1024:5.1f} GB)\n"
            )

            diff_mb = self.peak_memory_mb - self.estimated_peak_mb
            diff_pct = (
                (diff_mb / self.estimated_peak_mb * 100)
                if self.estimated_peak_mb > 0
                else 0
            )

            if abs(diff_pct) < 10:
                status = "✅ Close to estimate"
            elif diff_pct > 0:
                status = f"⚠️  {diff_pct:+.1f}% over estimate"
            else:
                status = f"✅ {diff_pct:+.1f}% under estimate"

            f.write(f"  Difference:      {diff_mb:+8.1f} MB ({status})\n")
            f.write(
                f"  Average memory:  {avg_memory:8.1f} MB ({avg_memory / 1024:5.1f} GB)\n\n"
            )

            f.write("-" * 70 + "\n")
            f.write("PROCESS USAGE\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Expected concurrent:  {self.total_processes:3d}\n")
            f.write(f"  Peak observed:        {self.peak_child_count:3d}\n")
            f.write(f"  Average observed:     {avg_children:5.1f}\n\n")

            slurm_mem_env = os.environ.get("SLURM_MEM_PER_NODE", "")
            if slurm_mem_env.isdigit() and int(slurm_mem_env) > 0:
                slurm_mb = int(slurm_mem_env)
            else:
                slurm_mb = int(psutil.virtual_memory().total / 1024 / 1024)
            slurm_gb = slurm_mb / 1024

            f.write("-" * 70 + "\n")
            f.write(f"ANALYSIS FOR {slurm_gb:.0f}GB ALLOCATION\n")
            f.write("-" * 70 + "\n")
            free_mb = slurm_mb - self.peak_memory_mb
            margin_pct = (free_mb / slurm_mb) * 100

            f.write(f"  SLURM allocation:  {slurm_gb:5.0f} GB ({slurm_mb:8.0f} MB)\n")
            f.write(
                f"  Peak usage:        {self.peak_memory_mb / 1024:5.1f} GB ({self.peak_memory_mb:8.0f} MB)\n"
            )
            f.write(
                f"  Free memory:       {free_mb / 1024:5.1f} GB ({free_mb:8.0f} MB)\n"
            )
            f.write(f"  Safety margin:     {margin_pct:5.1f}%\n\n")

            if margin_pct >= 40:
                f.write("  Status: ✅ SAFE - Good safety margin\n")
            elif margin_pct >= 20:
                f.write("  Status: ⚠️  TIGHT - Limited safety margin\n")
                f.write(
                    "  Recommendation: Monitor for OOM, consider reducing workers\n"
                )
            elif margin_pct >= 0:
                f.write("  Status: 🔴 RISKY - Very close to limit\n")
                f.write(
                    "  Recommendation: Reduce parallel_test_workers or parallel_optimizations\n"
                )
            else:
                f.write("  Status: ❌ OOM RISK - Exceeded allocation!\n")
                f.write("  Recommendation: Reduce workers or request more memory\n")

            f.write("\n")

            if margin_pct < 40:
                f.write("-" * 70 + "\n")
                f.write(f"SUGGESTED CONFIGURATIONS ({slurm_gb:.0f}GB)\n")
                f.write("-" * 70 + "\n")

                configs = [
                    (1, 4, "Sequential optimizations"),
                    (2, 2, "Balanced parallel"),
                    (2, 3, "Medium parallel"),
                ]

                per_process_mb = 2.0 * 1024
                base_mb = 5.0 * 1024
                buffer_mb = 4.0 * 1024

                for opt, workers, desc in configs:
                    procs = opt * workers
                    est_mb = base_mb + (procs * per_process_mb) + buffer_mb
                    free = slurm_mb - est_mb
                    margin = (free / slurm_mb) * 100

                    if margin >= 40:
                        status = "✅"
                    elif margin >= 20:
                        status = "⚠️"
                    else:
                        status = "🔴"

                    f.write(
                        f"  {status} opt={opt}, workers={workers} → {procs:2d} procs, "
                        f"~{est_mb / 1024:5.1f}GB, {margin:5.1f}% free - {desc}\n"
                    )

                f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("SAMPLING INFO\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total samples:     {len(self.samples):4d}\n")
            f.write("  Sampling interval: ~5 seconds\n\n")

            f.write("=" * 70 + "\n")
