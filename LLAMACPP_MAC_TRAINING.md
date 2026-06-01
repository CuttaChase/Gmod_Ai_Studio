# Mac Training

The recommended macOS training path now uses `MLX` for LoRA fine-tuning and Ollama for import/runtime.

Main command:

```bash
python3 scripts/train_mlx_gmod_macos.py
```

Default assumptions:

- dataset: `datasets/gmod_lua_quickstart_train.jsonl`
- base model: local Ollama model `llama3.2:1b`
- selected rows: `256`
- epochs: `1`
- output dir: `training_runs/llamacpp-llama32-1b-mac`
- created Ollama model: `gmod-llama32-1b-llamacpp`

## What the script does

1. Reads the selected JSONL train dataset plus optional chat-memory rows.
2. Builds MLX `train.jsonl`, `valid.jsonl`, and `test.jsonl` message datasets.
3. Runs local `mlx_lm lora` fine-tuning.
4. Fuses the adapters into a local model directory.
5. Writes a Modelfile for Ollama.
6. Optionally runs `ollama create --experimental` on the fused model directory.

## Chat Memory

To save more future training rows from conversation:

```bash
python3 scripts/chat_train_memory_ollama.py
```

Later training runs can append `output/chat_memory_train.jsonl` automatically.

## Install llama.cpp

If `mlx_lm` is not already available, reinstall the repo environment or run the repo setup again. The old llama.cpp-only path remains available for advanced manual use with an explicit non-quantized GGUF base.

If you still want the legacy llama.cpp path:

```bash
brew install llama.cpp
```

If Homebrew does not provide `llama-finetune`, build from source:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release --target llama-finetune llama-perplexity llama-cli llama-server
```

## First dry run

```bash
python3 scripts/train_mlx_gmod_macos.py --prepare-only
```

## Example override

```bash
python3 scripts/train_mlx_gmod_macos.py \
  --mlx-model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --dataset-file datasets/gmod_lua_quickstart_train.jsonl \
  --eval-file datasets/gmod_lua_quickstart_eval.jsonl \
  --context-length 512 \
  --batch-size 2 \
  --iters 100
```

## Ollama import

After a successful run, the script writes files into your chosen output directory, including `fused_model/`, `Modelfile`, MLX adapter artifacts, and the prepared dataset folder, then it can register the model with Ollama automatically.