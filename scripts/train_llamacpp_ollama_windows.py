#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import train_llamacpp_ollama_macos as base

base.DEFAULT_OUTPUT_DIR = Path("training_runs/llamacpp-llama32-1b-windows")


if __name__ == "__main__":
    try:
        base.main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)