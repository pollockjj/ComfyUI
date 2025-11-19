"""Markdown report generation for test results."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def write_markdown_report(
    test_results: Dict[str, Dict[str, Any]],
    config: dict,
    rank: int
) -> str:
    """Generate structured markdown report from test results.
    
    Args:
        test_results: Dict mapping test names to result dicts
        config: Test configuration
        rank: Distributed rank
    
    Returns:
        Path to generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Store logs in parallel-attention repo, not ComfyUI
    import os
    parallel_attention_root = Path(os.environ.get("PARALLEL_ATTENTION_ROOT", "/home/johnj/parallel-attention"))
    output_dir = parallel_attention_root / "test_logs" / "unit_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_test_results.md"
    
    # Count results by status
    passed = sum(1 for r in test_results.values() if r['status'] == 'PASS')
    failed = sum(1 for r in test_results.values() if r['status'] == 'FAIL')
    skipped = sum(1 for r in test_results.values() if r['status'] == 'SKIP')
    errors = sum(1 for r in test_results.values() if r['status'] == 'ERROR')
    
    with open(output_path, 'w') as f:
        # Header
        f.write(f"# Unit Test Results: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Configuration
        f.write("**Configuration:**\n")
        f.write(f"- Model: {config.get('model', 'flux')}\n")
        f.write(f"- Backend: {config.get('bps_backend_name', 'unknown')}\n")
        f.write(f"- Rank: {rank}\n")
        f.write(f"- Ulysses Degree: {config.get('ulysses_degree', 1)}\n")
        f.write(f"- Ring Degree: {config.get('ring_degree', 1)}\n")
        f.write(f"- Attention Backend: {config.get('attention_backend', 'unknown')}\n\n")
        
        # Summary
        f.write("## Test Results Summary\n")
        f.write(f"✅ Passed: {passed} | ❌ Failed: {failed} | ")
        f.write(f"⏭️ Skipped: {skipped} | 💥 Errors: {errors}\n\n")
        
        # Summary table
        f.write("| Test | Status | Duration |\n")
        f.write("|------|--------|----------|\n")
        for name, result in test_results.items():
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌",
                "SKIP": "⏭️",
                "ERROR": "💥"
            }[result['status']]
            duration = result.get('duration', 0.0)
            f.write(f"| {name} | {status_icon} {result['status']} | {duration:.3f}s |\n")
        
        f.write("\n---\n\n")
        
        # Detailed results
        for name, result in test_results.items():
            f.write(f"## {name}\n")
            f.write(f"**Status:** {result['status']}\n")
            f.write(f"**Duration:** {result.get('duration', 0.0):.3f}s\n\n")
            f.write(f"{result['message']}\n\n")
            
            if 'traceback' in result:
                f.write("**Stack Trace:**\n```python\n")
                f.write(result['traceback'])
                f.write("\n```\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"Test report written to: {output_path}")
    return str(output_path)
