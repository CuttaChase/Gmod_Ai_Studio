# GMod AI Studio

This repo is now set up around one local training flow on both operating systems:

- Ollama for model management and chat
- llama.cpp for local fine-tuning
- JSONL datasets in `datasets/`
- scraped source data in `datasets/scraped/`

## Fast Start

macOS:

```bash
./setup_macos.sh
source .venv/bin/activate
python3 run_gmod_ai_studio.py
```

Windows PowerShell:

```powershell
.\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python .\run_gmod_ai_studio.py
```

## What The Setup Scripts Do

Both setup scripts are meant to leave you ready to train with Ollama and llama.cpp right away.

They do this:

- install the main prerequisites for that OS
- create `.venv`
- normalize `gmod_ai_studio_settings.json`
- make sure Ollama is running
- pull the chat model and base model from your settings
- make sure `llama-finetune` is available
- build the starter datasets in `datasets/`

## Folder Layout

Root:

- `run_gmod_ai_studio.py`
- `gmod_ai_studio_settings.json`
- `setup_macos.sh`
- `setup_windows.ps1`
- `datasets/`
- `output/`
- `training_runs/`
- `scripts/`
- `readme/`
- `scrape website data/`

Python scripts:

- `scripts/gmod_ai_studio.py`
- `scripts/chat_train_memory_ollama.py`
- `scripts/prepare_training_data.py`
- `scripts/train_mlx_gmod_macos.py`
- `scripts/train_llamacpp_ollama_windows.py`

## Datasets

Training and eval files live in `datasets/`.

In the Studio, Training Settings now supports selecting multiple `*_train.jsonl` files at once. When you start training, the Studio combines the selected train datasets into one prepared JSONL automatically.

Default generated files:

- `datasets/gmod_lua_train.jsonl`
- `datasets/gmod_lua_eval.jsonl`
- `datasets/gmod_lua_focused_train.jsonl`
- `datasets/gmod_lua_focused_eval.jsonl`
- `datasets/gmod_lua_quickstart_train.jsonl`
- `datasets/gmod_lua_quickstart_eval.jsonl`

If you add your own `*_train.jsonl` and `*_eval.jsonl` files to `datasets/`, they show up in the Studio dropdown automatically.

## Scraped Website Data

The scraper files now live in `scrape website data/`.

Windows batch entrypoint:

```bat
scrape website data\scrape_facepunch_gmod.bat
```

PowerShell entrypoint:

```powershell
.\scrape website data\scrape_facepunch_gmod.ps1
```

By default, the scraper writes JSON into:

```text
datasets/scraped/
```

Any scraped JSON files in that folder are picked up by `scripts/prepare_training_data.py` the next time you rebuild datasets.

## Optional Manual Commands

Rebuild datasets:

```bash
python3 scripts/prepare_training_data.py
```

Mac llama.cpp training:

```bash
python3 scripts/train_mlx_gmod_macos.py
```

Windows llama.cpp training:

```powershell
python .\scripts\train_llamacpp_ollama_windows.py
```

On Windows, the Studio also has a `Windows Acceleration` setting so you can keep the path on `Auto`, force `CPU Only`, or choose `CUDA GPU`.

Chat and save memory rows for future retraining:

```bash
python3 scripts/chat_train_memory_ollama.py
```

## Notes

- `output/` stores chat-memory logs and exported message rows.
- `training_runs/` stores training corpora, GGUF output, and Ollama import files.
- The Studio is now wired for llama.cpp on both Mac and Windows.