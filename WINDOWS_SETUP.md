# Windows Setup

This repo now uses the same local training stack on Windows as the rest of the project:

- Ollama for model management
- llama.cpp for local fine-tuning
- datasets from `datasets/`
- optional scraped source data from `datasets/scraped/`

## One-Time Setup

Open PowerShell in the repo root and run:

```powershell
.\setup_windows.ps1
```

That script tries to do the whole Windows setup for you:

- installs Python 3.12
- installs Ollama
- installs Git and CMake
- installs Visual Studio Build Tools for C++
- creates `.venv`
- normalizes `gmod_ai_studio_settings.json`
- pulls the Ollama models from your settings
- builds `llama.cpp` if `llama-finetune.exe` is missing
- rebuilds the local datasets

The Studio includes a `Windows Acceleration` setting with these modes:

- `Auto`: use CUDA when the CUDA toolkit is available, otherwise fall back to CPU
- `CPU Only`: always keep training on CPU
- `CUDA GPU`: require a CUDA-capable build and use GPU offload

If PowerShell blocks script execution in that window, run this first and then run setup again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Start The Studio

After setup finishes:

```powershell
.\.venv\Scripts\Activate.ps1
python .\run_gmod_ai_studio.py
```

Then open the local Studio URL shown in the terminal if the browser does not open automatically.

## Manual Commands

Rebuild datasets:

```powershell
python .\scripts\prepare_training_data.py
```

Run the Windows llama.cpp trainer directly:

```powershell
python .\scripts\train_llamacpp_ollama_windows.py
```

Save chat memory for future retraining:

```powershell
python .\scripts\chat_train_memory_ollama.py
```

## Scraping Website Data

The scraper files are in `scrape website data\`.

Batch entrypoint:

```bat
scrape website data\scrape_facepunch_gmod.bat
```

PowerShell entrypoint:

```powershell
.\scrape website data\scrape_facepunch_gmod.ps1
```

By default, scraped output is saved into `datasets\scraped\`. The next dataset rebuild will automatically use any scraped JSON files found there.