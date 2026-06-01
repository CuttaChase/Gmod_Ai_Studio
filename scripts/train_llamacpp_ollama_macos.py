#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SYSTEM_PROMPT = "You are an expert Garry's Mod Lua assistant. Answer using Lua 5.1 and Garry's Mod API behavior accurately, concisely, and with working examples when useful."
DEFAULT_OUTPUT_DIR = Path("training_runs/llamacpp-llama32-1b-mac")
DEFAULT_TRAINED_MODEL_NAME = "finetuned-model.gguf"
DEFAULT_BASE_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_MEMORY_DATASET = Path("output/chat_memory_train.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and run a local llama.cpp fine-tune, then optionally register the resulting GGUF with Ollama."
    )
    parser.add_argument("--dataset-file", default="datasets/gmod_lua_quickstart_train.jsonl")
    parser.add_argument("--base-model-gguf")
    parser.add_argument("--base-ollama-model", default=DEFAULT_BASE_OLLAMA_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--memory-dataset-file", default=str(DEFAULT_MEMORY_DATASET))
    parser.add_argument("--skip-memory-dataset", action="store_true")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ubatch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "metal", "cuda"], help="Preferred device for llama.cpp training: auto, cpu, metal, or cuda.")
    parser.add_argument("--llama-bin-dir")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--create-ollama-model", dest="create_ollama_model", action="store_true", default=True)
    parser.add_argument("--skip-ollama-create", dest="create_ollama_model", action="store_false")
    parser.add_argument("--ollama-model-name", default="gmod-llama32-1b-llamacpp")
    parser.add_argument("--export-model-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "answer" not in row:
                raise ValueError(f"Line {line_number} is missing question/answer fields")
            rows.append(row)
    return rows


def normalize_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_chatml_sample(question, answer):
    question = normalize_text(question)
    answer = normalize_text(answer)
    return textwrap.dedent(
        f"""\
        <|im_start|>system
        {SYSTEM_PROMPT}
        <|im_end|>
        <|im_start|>user
        {question}
        <|im_end|>
        <|im_start|>assistant
        {answer}
        <|im_end|>
        """
    )


def select_rows(rows, offset, limit):
    if limit > 0:
        rows = rows[offset : offset + limit]
    else:
        rows = rows[offset:]
    if not rows:
        raise ValueError("No rows selected from the dataset")
    return rows


def read_memory_rows(path):
    rows = []
    skipped_empty = 0
    skipped_truncated = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if row.get("truncated"):
                skipped_truncated += 1
                continue
            if not question or not answer:
                skipped_empty += 1
                continue
            rows.append({"question": question, "answer": answer})

    return rows, skipped_empty, skipped_truncated


def deduplicate_training_rows(rows):
    unique_rows = []
    seen = set()
    duplicate_count = 0

    for row in rows:
        question = normalize_text(str(row.get("question") or ""))
        answer = normalize_text(str(row.get("answer") or ""))
        if not question or not answer:
            continue
        key = (question, answer)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        normalized_row = dict(row)
        normalized_row["question"] = question
        normalized_row["answer"] = answer
        unique_rows.append(normalized_row)

    return unique_rows, duplicate_count


def build_corpus(rows):
    if not rows:
        raise ValueError("No rows selected from the dataset")
    return "\n".join(build_chatml_sample(row["question"], row["answer"]) for row in rows), rows


def resolve_executable(name, bin_dir=None):
    candidate_names = [name]
    if os.name == "nt" and not Path(name).suffix:
        candidate_names.extend([f"{name}.exe", f"{name}.cmd", f"{name}.bat"])

    for candidate_name in candidate_names:
        if bin_dir:
            candidate = Path(bin_dir) / candidate_name
            if candidate.exists():
                return str(candidate)
        resolved = shutil.which(candidate_name)
        if resolved:
            return resolved
    return None


def warn_if_probably_quantized(path):
    lowered = path.name.lower()
    quant_markers = ("q2", "q3", "q4", "q5", "q6", "q8", "iq", "ggml")
    if "f32" in lowered:
        return
    if any(marker in lowered for marker in quant_markers):
        print(
            "Warning: your base GGUF file name looks quantized. llama.cpp finetune is currently documented as experimental and most reliable with FP32 GGUF models.",
            file=sys.stderr,
        )


def build_finetune_command(executable, base_model_gguf, corpus_path, args):
    command = [
        executable,
        "--model",
        str(base_model_gguf),
        "--file",
        str(corpus_path),
        "-c",
        str(args.context_length),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--epochs",
        str(args.epochs),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--val-split",
        str(args.validation_split),
    ]
    if args.gpu_layers > 0:
        command.extend(["-ngl", str(args.gpu_layers)])
    return command


def build_modelfile(trained_model_path, num_ctx):
    return textwrap.dedent(
        '''\
        FROM {trained_model_path}

        TEMPLATE """{{ if .System }}<|im_start|>system
        {{{{ .System }}}}
        <|im_end|>
        {{ end }}<|im_start|>user
        {{{{ .Prompt }}}}
        <|im_end|>
        <|im_start|>assistant
        {{{{ .Response }}}}<|im_end|>
        """

        SYSTEM """{system_prompt}"""

        PARAMETER num_ctx {num_ctx}
        PARAMETER stop "<|im_end|>"
        '''
    ).format(
        trained_model_path=trained_model_path,
        system_prompt=SYSTEM_PROMPT,
        num_ctx=num_ctx,
    )


def sanitize_model_name_for_path(model_name):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(model_name or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "exported-model"


def export_model_bundle(export_root, model_name, trained_model_path, num_ctx, output_dir):
    bundle_dir = Path(export_root) / sanitize_model_name_for_path(model_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    exported_model_path = bundle_dir / trained_model_path.name
    shutil.copy2(trained_model_path, exported_model_path)

    portable_modelfile_path = bundle_dir / "Modelfile"
    portable_modelfile_path.write_text(
        build_modelfile(f"./{exported_model_path.name}", num_ctx),
        encoding="utf-8",
    )

    readme_path = bundle_dir / "README.txt"
    readme_path.write_text(
        textwrap.dedent(
            f"""\
            Portable Ollama bundle for {model_name}

            Files
            - {exported_model_path.name}: trained GGUF model file
            - Modelfile: portable Ollama Modelfile that references the GGUF in this folder

            Import on another system
            1. Copy this entire folder to the target machine.
            2. Open a terminal in this folder.
            3. Run:
               ollama create {model_name} -f Modelfile

            Source output directory
            {output_dir.resolve()}
            """
        ),
        encoding="utf-8",
    )

    metadata_path = bundle_dir / "bundle.json"
    metadata_path.write_text(
        json.dumps(
            {
                "ollama_model_name": model_name,
                "trained_model_file": exported_model_path.name,
                "source_output_dir": str(output_dir.resolve()),
                "import_command": f"ollama create {model_name} -f Modelfile",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return bundle_dir


def run_command(command, cwd=None, dry_run=False):
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def ensure_llama_finetune_available(args):
    executable = resolve_executable("llama-finetune", args.llama_bin_dir)
    if executable:
        return executable

    if platform.system() == "Windows":
        message = textwrap.dedent(
            r"""\
            Could not find `llama-finetune` on PATH.

            Install or build llama.cpp first. In this repo, the easiest path is:
                .\setup_windows.ps1

            Manual Windows build example:
                git clone https://github.com/ggml-org/llama.cpp.git %USERPROFILE%\llama.cpp
                cd %USERPROFILE%\llama.cpp
                cmake -S . -B build
                cmake --build build --config Release --target llama-finetune llama-perplexity llama-cli llama-server

            Then rerun this script with either:
                --llama-bin-dir C:\Users\you\llama.cpp\build\bin\Release

            or by putting `llama-finetune.exe` on your PATH.
            """
        ).strip()
    else:
        message = textwrap.dedent(
            """\
            Could not find `llama-finetune` on PATH.

            Install llama.cpp first. On macOS, a common path is:
                brew install llama.cpp

            If your package manager build does not include `llama-finetune`, build it from source:
                git clone https://github.com/ggml-org/llama.cpp.git
                cd llama.cpp
                cmake -B build -DLLAMA_METAL=ON
                cmake --build build --config Release --target llama-finetune llama-perplexity llama-cli llama-server

            Then rerun this script with either:
                --llama-bin-dir /path/to/llama.cpp/build/bin

            or by putting `llama-finetune` on your PATH.
            """
        ).strip()
    raise RuntimeError(message)


def ensure_ollama_available():
    executable = resolve_executable("ollama")
    if executable:
        return executable
    raise RuntimeError("Could not find `ollama` on PATH. Install Ollama before using --create-ollama-model.")


def inspect_ollama_model_metadata(ollama_executable, model_name):
    result = subprocess.run(
        [ollama_executable, "show", model_name],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        key = parts[0].rstrip(":").lower()
        if key in {"architecture", "parameters", "quantization"} and len(parts) > 1:
            metadata[key] = " ".join(parts[1:])
    return metadata


def reject_quantized_ollama_training_base(ollama_executable, model_name):
    metadata = inspect_ollama_model_metadata(ollama_executable, model_name)
    quantization = metadata.get("quantization", "").upper()
    if not quantization:
        return
    if quantization.startswith("Q") or quantization.startswith("IQ"):
        architecture = metadata.get("architecture", "unknown architecture")
        parameter_count = metadata.get("parameters", "unknown size")
        raise RuntimeError(
            f"The selected Ollama base model `{model_name}` is quantized ({quantization}, {architecture}, {parameter_count}). "
            "This local llama.cpp fine-tune path is currently crashing on quantized Ollama runtime models, which is why the Studio Start Training button fails. "
            "On macOS, use `scripts/train_mlx_gmod_macos.py` for the default Studio training path instead. "
            "Use `--prepare-only` if you only want to build the corpus, or pass `--base-model-gguf` with a non-quantized F16/BF16/F32 GGUF training base."
        )


def validate_base_model_selection(args):
    if args.base_model_gguf:
        return

    model_name = (args.base_ollama_model or "").strip()
    if not model_name:
        raise RuntimeError(
            "No base model was selected. Set --base-ollama-model or pass --base-model-gguf explicitly."
        )

    if model_name.lower().startswith("qwen3.5:"):
        raise RuntimeError(
            "The local Ollama qwen3.5 GGUF blob is not currently usable with this llama.cpp fine-tune path on this machine. "
            "Your loader error `qwen35.rope.dimension_sections has wrong array length; expected 4, got 3` means the installed Ollama blob metadata does not match what the local llama.cpp loader expects. "
            "Use a different Ollama base model for local llama.cpp training, or pass --base-model-gguf with a compatible GGUF file. "
            "If you need Qwen3.5 specifically, use a compatible GGUF base file instead of the current local Ollama blob."
        )


def resolve_ollama_model_blob_path(ollama_executable, model_name):
    command = [ollama_executable, "show", model_name, "--modelfile"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if not line.startswith("FROM "):
            continue
        candidate = Path(line.removeprefix("FROM ").strip())
        if candidate.exists():
            return candidate
        raise RuntimeError(
            f"Ollama reported a base blob path for {model_name}, but it does not exist: {candidate}"
        )

    raise RuntimeError(
        f"Could not resolve a local GGUF path from `ollama show {model_name} --modelfile`. "
        "Pull the model first or pass --base-model-gguf explicitly."
    )


def main():
    args = parse_args()
    validate_base_model_selection(args)
    dataset_path = Path(args.dataset_file)
    memory_dataset_path = Path(args.memory_dataset_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        raise SystemExit(1)

    base_rows = read_rows(dataset_path)
    rows = list(select_rows(base_rows, args.offset, args.limit))
    if not args.skip_memory_dataset and memory_dataset_path.exists():
        memory_rows, skipped_empty, skipped_truncated = read_memory_rows(memory_dataset_path)
        rows.extend(memory_rows)
        print(
            f"Included {len(memory_rows)} extra memory rows from {memory_dataset_path}",
            file=sys.stderr,
        )
        if skipped_empty:
            print(
                f"Skipped {skipped_empty} extra memory rows with empty question/answer fields",
                file=sys.stderr,
            )
        if skipped_truncated:
            print(
                f"Skipped {skipped_truncated} truncated extra memory rows",
                file=sys.stderr,
            )

    rows, duplicate_rows = deduplicate_training_rows(rows)
    if duplicate_rows:
        print(
            f"Skipped {duplicate_rows} duplicate training rows across the base and memory datasets",
            file=sys.stderr,
        )

    corpus_text, selected_rows = build_corpus(rows)
    corpus_path = output_dir / "train_corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")

    trained_model_path = output_dir / DEFAULT_TRAINED_MODEL_NAME
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        build_modelfile(trained_model_path.resolve(), args.context_length),
        encoding="utf-8",
    )

    print(f"Wrote corpus with {len(selected_rows)} samples to {corpus_path}")
    print(f"Wrote Ollama Modelfile to {modelfile_path}")

    if args.prepare_only:
        print("Preparation complete. No training command was run because --prepare-only was set.")
        return

    if args.base_model_gguf:
        base_model_gguf = Path(args.base_model_gguf)
    else:
        ollama = ensure_ollama_available()
        reject_quantized_ollama_training_base(ollama, args.base_ollama_model)
        print(
            f"Resolving base GGUF from installed Ollama model: {args.base_ollama_model}",
            file=sys.stderr,
        )
        base_model_gguf = resolve_ollama_model_blob_path(ollama, args.base_ollama_model)

    if not base_model_gguf.exists():
        print(f"Base GGUF file not found: {base_model_gguf}", file=sys.stderr)
        raise SystemExit(1)

    warn_if_probably_quantized(base_model_gguf)
    llama_finetune = ensure_llama_finetune_available(args)
    finetune_command = build_finetune_command(llama_finetune, base_model_gguf.resolve(), corpus_path.resolve(), args)
    run_command(finetune_command, cwd=output_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete. No training or Ollama import was executed.")
        return

    if not trained_model_path.exists():
        raise RuntimeError(
            f"llama-finetune finished but {trained_model_path} was not created. Check the llama.cpp output above."
        )

    print(f"Training output is available at {trained_model_path}")

    if args.create_ollama_model:
        ollama = ensure_ollama_available()
        ollama_command = [
            ollama,
            "create",
            args.ollama_model_name,
            "-f",
            str(modelfile_path.resolve()),
        ]
        run_command(ollama_command, dry_run=False)
        print(f"Created Ollama model: {args.ollama_model_name}")
    else:
        print(
            "To register the trained GGUF with Ollama later, run:\n"
            f"  ollama create {args.ollama_model_name} -f {shlex.quote(str(modelfile_path.resolve()))}"
        )

    if args.export_model_dir:
        bundle_dir = export_model_bundle(
            args.export_model_dir,
            args.ollama_model_name,
            trained_model_path,
            args.context_length,
            output_dir,
        )
        print(f"Exported portable model bundle to {bundle_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)