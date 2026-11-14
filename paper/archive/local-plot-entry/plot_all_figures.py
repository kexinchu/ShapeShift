#!/usr/bin/env python3
"""Regenerate all combined paper figures."""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    scripts = [
        "plot_ablation_llada8b.py",
        "plot_dinfer_wrapper.py",
        "plot_fullstep_llada8b.py",
        "plot_main_online_combined.py",
    ]
    for script in scripts:
        print(f"running {script}...")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
