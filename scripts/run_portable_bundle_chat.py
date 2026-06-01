#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run chat generation directly from a portable model bundle.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--messages-json", default="[]")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def load_bundle_metadata(bundle_dir):
    bundle_json_path = bundle_dir / "bundle.json"
    if not bundle_json_path.exists():
        return {}
    try:
        return json.loads(bundle_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_bundle_backend(bundle_dir, metadata):
    trained_model_file = str(metadata.get("trained_model_file") or "").strip()
    if trained_model_file:
        trained_model_path = bundle_dir / trained_model_file
        if trained_model_path.is_dir():
            return "mlx", trained_model_path
        if trained_model_path.exists() and trained_model_path.suffix.lower() == ".gguf":
            return "gguf", trained_model_path
    fused_model_dir = bundle_dir / "fused_model"
    if fused_model_dir.is_dir():
        return "mlx", fused_model_dir
    for child in sorted(bundle_dir.iterdir(), key=lambda path: path.name.lower()):
        if child.is_file() and child.suffix.lower() == ".gguf":
            return "gguf", child
    raise RuntimeError(f"Could not find a portable model inside {bundle_dir}")


def build_fallback_prompt(messages):
    parts = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            label = "System"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = "User"
        parts.append(f"{label}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def normalize_generated_text(text):
    cleaned_text = str(text or "")
    for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        cleaned_text = cleaned_text.replace(token, "")
    return cleaned_text.strip()


def run_mlx_bundle(model_path, messages, temperature, seed, num_predict, warmup):
    import mlx.core as mx
    from mlx_lm import generate, load

    mx.random.seed(int(seed))
    model, tokenizer = load(str(model_path))
    if warmup:
        return {
            "message": {"content": ""},
            "done_reason": "load",
            "backend": "mlx",
        }

    prompt = build_fallback_prompt(messages)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            prompt = apply_chat_template(messages, tokenize=False)

    started_at = time.time()
    try:
        assistant_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max(1, int(num_predict)),
            temp=float(temperature),
        )
    except TypeError:
        assistant_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max(1, int(num_predict)),
        )
    return {
        "message": {"content": normalize_generated_text(assistant_text)},
        "done_reason": "stop",
        "backend": "mlx",
        "eval_count": max(0, len(str(assistant_text).split())),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def resolve_llama_cli():
    candidates = ["llama-cli"]
    if sys.platform.startswith("win"):
        candidates.insert(0, "llama-cli.exe")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("llama-cli was not found in PATH. Install llama.cpp to use GGUF portable bundles directly.")


def run_gguf_bundle(model_path, messages, temperature, seed, num_predict, warmup):
    command = [
        resolve_llama_cli(),
        "-m",
        str(model_path),
        "-p",
        "Warmup" if warmup else build_fallback_prompt(messages),
        "-n",
        str(1 if warmup else max(1, int(num_predict))),
        "--temp",
        str(float(temperature)),
        "--seed",
        str(int(seed)),
        "--no-display-prompt",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not run llama-cli: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"llama-cli exited with code {completed.returncode}")
    return {
        "message": {"content": "" if warmup else normalize_generated_text(stdout)},
        "done_reason": "load" if warmup else "stop",
        "backend": "gguf",
        "eval_count": max(0, len(stdout.split())),
    }


def main():
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise RuntimeError(f"Bundle directory not found: {bundle_dir}")

    try:
        messages = json.loads(args.messages_json or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("--messages-json must be valid JSON.") from exc
    if not isinstance(messages, list):
        raise RuntimeError("--messages-json must decode to a list of chat messages.")

    metadata = load_bundle_metadata(bundle_dir)
    backend, model_path = resolve_bundle_backend(bundle_dir, metadata)
    if backend == "mlx":
        result = run_mlx_bundle(model_path, messages, args.temperature, args.seed, args.num_predict, args.warmup)
    else:
        result = run_gguf_bundle(model_path, messages, args.temperature, args.seed, args.num_predict, args.warmup)
    print(json.dumps(result), end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)