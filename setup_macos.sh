#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OLLAMA_MODEL="qwen2.5-coder:1.5b"
DEFAULT_MLX_MODEL="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup_macos.sh can only be run on macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon (arm64) is required for the MLX setup path in setup_macos.sh." >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are required. Starting the installer now..."
  xcode-select --install || true
  echo "Finish the Xcode Command Line Tools install, then rerun setup_macos.sh." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required for setup_macos.sh. Install Homebrew from https://brew.sh and run this script again." >&2
  exit 1
fi

brew install python@3.12 ollama git

PYTHON_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON=""
for candidate in .venv/bin/python .venv/bin/python3; do
  if [[ -x "$candidate" ]]; then
    VENV_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$VENV_PYTHON" ]]; then
  echo "The virtual environment Python executable was not created. Remove .venv and rerun setup_macos.sh." >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install -U pip setuptools wheel
"$VENV_PYTHON" -m pip install -U "mlx-lm>=0.18.1" "huggingface_hub>=0.26.0"

"$VENV_PYTHON" - <<'PY'
import json
from pathlib import Path

repo = Path.cwd()
settings_path = repo / "gmod_ai_studio_settings.json"
default_chat_model = "qwen2.5-coder:1.5b"
default_mlx_model = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
legacy_chat_models = {"", "ministral-3:3b", "llama3.2:1b", "qwen3.5:4b"}
legacy_mlx_models = {"", "mlx-community/Llama-3.2-1B-Instruct-4bit"}
legacy_ollama_models = {"", "llama3.2:1b"}
legacy_batch_sizes = {"", "64"}
legacy_output_dirs = {
  "",
  "training_runs/gmod-studio-mac",
  "training_runs/llamacpp-llama32-1b-mac",
  "training_runs/mlx-llama32-1b-mac",
}
legacy_ollama_model_names = {"", "gmod_lua", "gmod-llama32-1b-llamacpp", "gmod-llama32-1b-mlx"}

if settings_path.exists():
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
else:
    settings = {}


def replace_if_missing_or_legacy(mapping, key, value, legacy_values):
  current = str(mapping.get(key) or "").strip()
  if current in legacy_values:
    mapping[key] = value

app = settings.setdefault("app", {})
chat = settings.setdefault("chat", {})
dataset = settings.setdefault("dataset", {})
training = settings.setdefault("training", {})

app.setdefault("host", "127.0.0.1")
app.setdefault("port", 8765)
app.setdefault("auto_open_browser", True)
app.setdefault("poll_interval_ms", 2500)

replace_if_missing_or_legacy(chat, "model", default_chat_model, legacy_chat_models)
chat.setdefault("memory_file", "training_runs/chat_memory/session_memory.json")
chat.setdefault("training_log_file", "output/chat_memory_train.jsonl")
chat.setdefault("messages_log_file", "output/chat_memory_messages.jsonl")
chat.setdefault("max_history_messages", 48)
chat.setdefault("temperature", 0.1)
chat.setdefault("seed", 3407)
chat.setdefault("timeout", 600.0)
chat.setdefault("num_predict", 768)
chat.setdefault("device", "auto")
chat.setdefault("enable_thinking", False)

dataset["script"] = "scripts/prepare_training_data.py"
dataset.setdefault("extra_args", "")

training["platform"] = "mac"
training["script"] = "scripts/train_mlx_gmod_macos.py"
training["dataset_file"] = "datasets/gmod_lua_coding_train.jsonl"
training["dataset_files"] = ["datasets/gmod_lua_coding_train.jsonl"]
training["eval_file"] = "datasets/gmod_lua_coding_eval.jsonl"
replace_if_missing_or_legacy(training, "mlx_model", default_mlx_model, legacy_mlx_models)
training.setdefault("base_model_gguf", "")
replace_if_missing_or_legacy(training, "base_ollama_model", default_chat_model, legacy_ollama_models)
replace_if_missing_or_legacy(training, "output_dir", "training_runs/mlx-qwen25-coder-1_5b-mac", legacy_output_dirs)
training.setdefault("export_model_bundle", True)
training.setdefault("export_model_dir", "models")
training.setdefault("memory_dataset_file", "output/chat_memory_train.jsonl")
training.setdefault("skip_memory_dataset", False)
training.setdefault("limit", 128)
training.setdefault("offset", 0)
training.setdefault("context_length", 1024)
replace_if_missing_or_legacy(training, "batch_size", 2, legacy_batch_sizes)
training.setdefault("ubatch_size", 16)
training.setdefault("epochs", 1)
training.setdefault("iters", 300)
training.setdefault("grad_accumulation_steps", 4)
training.setdefault("num_layers", 8)
training.setdefault("steps_per_report", 10)
training.setdefault("steps_per_eval", 50)
training.setdefault("val_batches", 2)
training.setdefault("save_every", 50)
training.setdefault("gpu_layers", 0)
training.setdefault("threads", 4)
training.setdefault("seed", 3407)
training.setdefault("validation_split", 0.1)
training.setdefault("windows_acceleration", "auto")
training.setdefault("mac_acceleration", "auto")
training.setdefault("llama_bin_dir", "")
training.setdefault("prepare_only", False)
training.setdefault("create_ollama_model", True)
replace_if_missing_or_legacy(training, "ollama_model_name", "gmod-qwen25-coder-1_5b-mlx", legacy_ollama_model_names)
training.setdefault("grad_checkpoint", True)
training.setdefault("mask_prompt", True)
training.setdefault("dry_run", False)
training.setdefault("extra_args", "")

(repo / "datasets" / "scraped").mkdir(parents=True, exist_ok=True)
(repo / "output").mkdir(parents=True, exist_ok=True)
(repo / "training_runs" / "chat_memory" / "users").mkdir(parents=True, exist_ok=True)

settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

if ! ollama list >/dev/null 2>&1; then
  if ! pgrep -f "[o]llama serve" >/dev/null 2>&1; then
    nohup ollama serve >"${TMPDIR:-/tmp}/gmod-ai-studio-ollama.log" 2>&1 &
  fi

  ready=0
  for _ in {1..30}; do
    if ollama list >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    echo "Ollama did not become ready in time. Start it manually with 'ollama serve' and rerun this setup script." >&2
    exit 1
  fi
fi

OLLAMA_MODELS=()
while IFS= read -r model; do
  if [[ -n "$model" ]]; then
    OLLAMA_MODELS+=("$model")
  fi
done < <("$VENV_PYTHON" - <<'PY'
import json
from pathlib import Path

settings = json.loads(Path("gmod_ai_studio_settings.json").read_text(encoding="utf-8"))
models = []
for value in [
    settings["chat"].get("model"),
    settings["training"].get("base_ollama_model"),
    "qwen2.5-coder:1.5b",
]:
    cleaned = str(value or "").strip()
    if cleaned and cleaned not in models:
        models.append(cleaned)
for model in models:
    print(model)
PY
)

for model in "${OLLAMA_MODELS[@]}"; do
  echo "Pulling Ollama model: $model"
  ollama pull "$model"
done

"$VENV_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

model_name = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
print(f"Caching MLX model: {model_name}")
snapshot_download(repo_id=model_name, resume_download=True)
PY

"$VENV_PYTHON" scripts/prepare_training_data.py

echo
echo "Mac setup complete."
echo "Cached Ollama coding model: $DEFAULT_OLLAMA_MODEL"
echo "Cached MLX training model: $DEFAULT_MLX_MODEL"
echo "Next:"
echo "  cd \"$REPO_ROOT\""
echo "  source .venv/bin/activate"
echo "  python3 run_gmod_ai_studio.py"