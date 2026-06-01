#!/usr/bin/env python3
import runpy
from pathlib import Path


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "scripts" / "gmod_ai_studio.py"
    runpy.run_path(str(script_path), run_name="__main__")