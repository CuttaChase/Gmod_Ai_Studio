# macOS Setup

The macOS path now uses MLX for LoRA fine-tuning and Ollama for local chat and model import.

Run this from the repo root:

```bash
./setup_macos.sh
```

That script:

- installs Python, Ollama, Git, CMake, and llama.cpp with Homebrew
- creates `.venv`
- normalizes `gmod_ai_studio_settings.json`
- pulls the Ollama models from your current settings
- makes sure `llama-finetune` is available
- builds the starter datasets in `datasets/`

Then start the Studio:

```bash
source .venv/bin/activate
python3 run_gmod_ai_studio.py
```

Useful manual commands:

```bash
python3 scripts/prepare_training_data.py
python3 scripts/train_mlx_gmod_macos.py
python3 scripts/chat_train_memory_ollama.py
```

The older `scripts/train_llamacpp_ollama_macos.py` path is still available for advanced manual use with a non-quantized GGUF base, but the Studio now defaults to the MLX trainer on Mac because the usual Ollama runtime blobs are quantized and cannot be fine-tuned reliably with the local llama.cpp path.

If you scrape new documentation into `datasets/scraped/`, rebuilding the dataset will include that scraped data automatically.