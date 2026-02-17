import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
        self.samples = []

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

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

                self.samples.append(
                    {
                        "time": time.time() - self.start_time,
                        "memory_mb": mem_mb,
                        "child_count": child_count,
                    }
                )
            except Exception:
                pass

    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._capture_snapshot()

    def generate_report(self):
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

        with open(self.report_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("WHITEFOX RESOURCE PROFILE\n")
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

            f.write("-" * 70 + "\n")
            f.write("ANALYSIS FOR 32GB ALLOCATION\n")
            f.write("-" * 70 + "\n")

            slurm_gb = 32
            slurm_mb = slurm_gb * 1024
            free_mb = slurm_mb - self.peak_memory_mb
            margin_pct = (free_mb / slurm_mb) * 100

            f.write(f"  SLURM allocation:  {slurm_gb:5d} GB ({slurm_mb:8.0f} MB)\n")
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
                f.write("SUGGESTED CONFIGURATIONS (32GB)\n")
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
