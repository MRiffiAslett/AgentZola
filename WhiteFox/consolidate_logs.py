#!/usr/bin/env python3
"""
Script to consolidate existing logs into the WhiteFox/logging directory.

Moves logs from:
- generated-outputs/whitefox_logs/ -> logging/execution_logs/
- generated-outputs/whitefox_bugs/ -> logging/bug_reports/

And creates a summary of existing logs.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List


def consolidate_logs(project_root: Path = None):
    """Consolidate existing logs into logging directory."""
    if project_root is None:
        # Find project root
        cwd = Path.cwd()
        if cwd.name == "WhiteFox":
            project_root = cwd
        else:
            current = cwd
            project_root = None
            while current != current.parent:
                if (current / "WhiteFox").exists() and (current / "WhiteFox").is_dir():
                    project_root = current / "WhiteFox"
                    break
                current = current.parent
            
            if not project_root:
                project_root = Path("WhiteFox")
    
    logging_dir = project_root / "logging"
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    moved_items = []
    
    # Move whitefox_state.json
    state_file_src = project_root / "whitefox_state.json"
    state_file_dst = logging_dir / "whitefox_state.json"
    if state_file_src.exists() and not state_file_dst.exists():
        shutil.move(str(state_file_src), str(state_file_dst))
        moved_items.append(("state_file", str(state_file_dst)))
        print(f"Moved whitefox_state.json to {state_file_dst}")
    elif state_file_src.exists():
        # Both exist, keep the newer one or merge?
        print(f"Warning: Both {state_file_src} and {state_file_dst} exist. Keeping {state_file_dst}")
    
    # Move entire generated-outputs directory
    generated_outputs_src = project_root / "generated-outputs"
    generated_outputs_dst = logging_dir / "generated-outputs"
    
    if generated_outputs_src.exists() and generated_outputs_src.is_dir():
        if not generated_outputs_dst.exists():
            shutil.move(str(generated_outputs_src), str(generated_outputs_dst))
            moved_items.append(("generated-outputs", str(generated_outputs_dst)))
            print(f"Moved generated-outputs to {generated_outputs_dst}")
        else:
            # Merge contents
            print(f"Warning: Both {generated_outputs_src} and {generated_outputs_dst} exist. Merging...")
            for item in generated_outputs_src.iterdir():
                dst_item = generated_outputs_dst / item.name
                if item.is_dir():
                    if dst_item.exists():
                        # Merge directories
                        for subitem in item.rglob("*"):
                            rel_path = subitem.relative_to(item)
                            dst_subitem = dst_item / rel_path
                            dst_subitem.parent.mkdir(parents=True, exist_ok=True)
                            if subitem.is_file() and not dst_subitem.exists():
                                shutil.move(str(subitem), str(dst_subitem))
                    else:
                        shutil.move(str(item), str(dst_item))
                else:
                    if not dst_item.exists():
                        shutil.move(str(item), str(dst_item))
            # Remove empty source directory
            try:
                generated_outputs_src.rmdir()
            except OSError:
                pass
            moved_items.append(("generated-outputs_merged", str(generated_outputs_dst)))
    
    # Also move individual subdirectories if they exist separately
    execution_logs_src = project_root / "generated-outputs" / "whitefox_logs"
    execution_logs_dst = logging_dir / "execution_logs"
    
    bug_reports_src = project_root / "generated-outputs" / "whitefox_bugs"
    bug_reports_dst = logging_dir / "bug_reports"
    
    # If generated-outputs was moved, these are now under logging/generated-outputs
    # But we also want them in the top-level logging directories for easy access
    if generated_outputs_dst.exists():
        whitefox_logs_src = generated_outputs_dst / "whitefox_logs"
        whitefox_bugs_src = generated_outputs_dst / "whitefox_bugs"
        
        if whitefox_logs_src.exists() and not execution_logs_dst.exists():
            shutil.move(str(whitefox_logs_src), str(execution_logs_dst))
            moved_items.append(("execution_logs", str(execution_logs_dst)))
            print(f"Moved whitefox_logs to {execution_logs_dst}")
        
        if whitefox_bugs_src.exists() and not bug_reports_dst.exists():
            shutil.move(str(whitefox_bugs_src), str(bug_reports_dst))
            moved_items.append(("bug_reports", str(bug_reports_dst)))
            print(f"Moved whitefox_bugs to {bug_reports_dst}")
    
    # Create summary
    summary = {
        "total_items_moved": len(moved_items),
        "items": [{"type": t, "path": p} for t, p in moved_items],
        "logging_directory": str(logging_dir),
    }
    
    summary_file = logging_dir / "consolidation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nConsolidation complete!")
    print(f"  Total items moved: {summary['total_items_moved']}")
    for item_type, item_path in moved_items:
        print(f"  - {item_type}: {item_path}")
    print(f"  Summary saved to: {summary_file}")


if __name__ == "__main__":
    consolidate_logs()

