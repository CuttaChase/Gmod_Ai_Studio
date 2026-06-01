#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from train_llamacpp_ollama_macos import (
    DEFAULT_BASE_OLLAMA_MODEL,
    DEFAULT_MEMORY_DATASET,
    SYSTEM_PROMPT,
    build_modelfile as build_generic_modelfile,
    deduplicate_training_rows,
    ensure_ollama_available,
    read_memory_rows,
    read_rows,
    run_command,
    sanitize_model_name_for_path,
    select_rows,
)

DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DEFAULT_OUTPUT_DIR = Path("training_runs/mlx-llama32-1b-mac")
LOW_SIGNAL_EXACT_ANSWERS = {"browser-attachment-turn", "gmod_model_ok", "smoketest", "stars1", "--<id>--"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and run a local MLX LoRA fine-tune on macOS, export GGUF, then optionally register the result with Ollama."
    )
    parser.add_argument("--dataset-file", default="datasets/gmod_lua_quickstart_train.jsonl")
    parser.add_argument("--eval-file", default="datasets/gmod_lua_quickstart_eval.jsonl")
    parser.add_argument("--mlx-model", default=DEFAULT_MLX_MODEL)
    parser.add_argument("--base-ollama-model", default=DEFAULT_BASE_OLLAMA_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--memory-dataset-file", default=str(DEFAULT_MEMORY_DATASET))
    parser.add_argument("--skip-memory-dataset", action="store_true")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "metal", "cuda"], help="Preferred device for MLX training: auto, cpu, metal, or cuda.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-ollama-create", dest="create_ollama_model", action="store_false")
    parser.add_argument("--ollama-model-name", default="gmod-llama32-1b-mlx")
    parser.add_argument("--export-model-dir", default="")
    parser.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--mask-prompt", dest="mask_prompt", action="store_true", default=True)
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false")
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Train using only the chat memory dataset and ignore the base training dataset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_text(text):
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_message_row(row):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalize_text(row.get("question"))},
            {"role": "assistant", "content": normalize_text(row.get("answer"))},
        ]
    }


def write_jsonl_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer_signal_text(text):
    normalized = normalize_text(text).lower()
    return normalized.strip().strip("*`_ ")


def is_low_signal_answer(answer):
    normalized = normalize_answer_signal_text(answer)
    if not normalized:
        return True
    if normalized in {
        "!",
        "?",
        ".",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "thanks",
        "thank you",
        "learned",
        "understood",
    }:
        return True
    if normalized.startswith("--<id>") or normalized.startswith("stars"):
        return True
    if len(normalized) < 16 and len(re.findall(r"[a-zA-Z0-9]{2,}", normalized)) <= 1:
        return True
    return False


def build_row_quality_limits(args):
    return {
        "max_question_chars": max(2000, int(args.context_length) * 7),
        "max_answer_chars": max(2500, int(args.context_length) * 10),
        "max_total_chars": max(4000, int(args.context_length) * 12),
    }


def is_low_signal_training_row(question, answer):
    lower_question = normalize_text(question).lower()
    normalized_answer = normalize_answer_signal_text(answer)

    if "reply with exactly:" in lower_question:
        return True

    if normalized_answer in LOW_SIGNAL_EXACT_ANSWERS:
        return True

    if is_low_signal_answer(answer):
        return True

    learns_with_ack = (
        "reply back with \"learned\"" in lower_question
        or "reply back with 'learned'" in lower_question
        or lower_question.startswith("learn this")
        or lower_question.startswith("learn it")
        or lower_question.startswith("learn this.")
    )
    if learns_with_ack and (normalized_answer == "learned" or len(normalize_text(answer)) <= 64):
        return True

    return False


def filter_training_rows(rows, args, source_name):
    limits = build_row_quality_limits(args)
    filtered_rows = []
    stats = {
        "source": source_name,
        "input_rows": len(rows),
        "skipped_too_long": 0,
        "skipped_low_signal": 0,
    }

    for row in rows:
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        total_chars = len(question) + len(answer)

        if is_low_signal_training_row(question, answer):
            stats["skipped_low_signal"] += 1
            continue

        if (
            len(question) > limits["max_question_chars"]
            or len(answer) > limits["max_answer_chars"]
            or total_chars > limits["max_total_chars"]
        ):
            stats["skipped_too_long"] += 1
            continue

        normalized_row = dict(row)
        normalized_row["question"] = question
        normalized_row["answer"] = answer
        normalized_row["_combined_chars"] = total_chars
        filtered_rows.append(normalized_row)

    return filtered_rows, stats


def limit_memory_rows(memory_rows, base_row_count):
    max_memory_rows = max(0, int(base_row_count or 0))
    if max_memory_rows <= 0 or len(memory_rows) <= max_memory_rows:
        return memory_rows, 0

    ranked_rows = sorted(
        memory_rows,
        key=lambda row: (row.get("_combined_chars", 0), -len(normalize_text(row.get("answer")))),
    )
    selected_rows = ranked_rows[:max_memory_rows]
    return selected_rows, len(memory_rows) - len(selected_rows)


def strip_row_quality_metadata(rows):
    cleaned = []
    for row in rows:
        normalized_row = dict(row)
        normalized_row.pop("_combined_chars", None)
        cleaned.append(normalized_row)
    return cleaned


def resolve_eval_rows(args, train_rows):
    eval_path = Path(args.eval_file)
    if eval_path.exists():
        return read_rows(eval_path), str(eval_path), False

    if not train_rows:
        raise ValueError("No training rows are available to split for validation.")

    split_index = max(1, int(len(train_rows) * (1.0 - max(0.0, min(args.validation_split, 0.5)))))
    split_index = min(split_index, len(train_rows) - 1) if len(train_rows) > 1 else 1
    valid_rows = train_rows[split_index:]
    remaining_train_rows = train_rows[:split_index]
    if not valid_rows:
        valid_rows = train_rows[-1:]
        remaining_train_rows = train_rows[:-1] or train_rows
    return (valid_rows, "split-from-train", True), remaining_train_rows


def build_lora_command(args, data_dir, adapter_path):
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.mlx_model,
        "--train",
        "--data",
        str(data_dir.resolve()),
        "--adapter-path",
        str(adapter_path.resolve()),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--num-layers",
        str(args.num_layers),
        "--max-seq-length",
        str(args.context_length),
        "--steps-per-report",
        str(args.steps_per_report),
        "--save-every",
        str(args.save_every),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--val-batches",
        str(args.val_batches),
        "--seed",
        str(args.seed),
    ]
    if args.mask_prompt:
        command.append("--mask-prompt")
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def build_fuse_command(args, adapter_path, fused_model_dir):
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        args.mlx_model,
        "--adapter-path",
        str(adapter_path.resolve()),
        "--save-path",
        str(fused_model_dir.resolve()),
        "--dequantize",
    ]


def build_mlx_runtime_env(device="auto"):
    env = os.environ.copy()
    if sys.platform == "darwin":
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    normalized_device = str(device or "auto").strip().lower()
    if normalized_device and normalized_device != "auto":
        env["MLX_DEVICE"] = normalized_device
        if normalized_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_mlx_command(command, cwd=None, dry_run=False, device="auto"):
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True, env=build_mlx_runtime_env(device=device))


def fetch_base_modelfile(base_ollama_model):
    if not str(base_ollama_model or "").strip():
        return ""
    try:
        ollama = ensure_ollama_available()
        result = subprocess.run(
            [ollama, "show", base_ollama_model, "--modelfile"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (RuntimeError, subprocess.CalledProcessError):
        return ""


def rebase_modelfile(template_text, model_source, context_length):
    if not template_text.strip():
        return build_generic_modelfile(model_source, context_length)

    rebased_lines = []
    replaced_from = False
    replaced_num_ctx = False
    for line in template_text.splitlines():
        if line.startswith("FROM ") and not replaced_from:
            rebased_lines.append(f"FROM {model_source}")
            replaced_from = True
            continue
        if line.startswith("PARAMETER num_ctx"):
            rebased_lines.append(f"PARAMETER num_ctx {context_length}")
            replaced_num_ctx = True
            continue
        rebased_lines.append(line)

    if not replaced_from:
        rebased_lines.insert(0, f"FROM {model_source}")
    if not replaced_num_ctx:
        rebased_lines.append(f"PARAMETER num_ctx {context_length}")
    return "\n".join(rebased_lines).rstrip() + "\n"


def command_to_string(command):
    return " ".join(shlex.quote(part) for part in command)


def export_model_bundle(export_root, model_name, fused_model_dir, modelfile_text, output_dir):
    bundle_dir = Path(export_root) / sanitize_model_name_for_path(model_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    exported_model_path = bundle_dir / "fused_model"
    if exported_model_path.exists():
        shutil.rmtree(exported_model_path)
    shutil.copytree(fused_model_dir, exported_model_path)

    portable_modelfile_path = bundle_dir / "Modelfile"
    portable_modelfile_path.write_text(
        rebase_modelfile(modelfile_text, "./fused_model", 0).replace("PARAMETER num_ctx 0\n", "", 1),
        encoding="utf-8",
    )

    readme_path = bundle_dir / "README.txt"
    readme_path.write_text(
        textwrap.dedent(
            f"""\
            Portable Ollama bundle for {model_name}

            Files
                - fused_model/: fused MLX model directory for Ollama import
                - Modelfile: portable Ollama Modelfile that references the fused_model folder in this bundle

            Import on another system
            1. Copy this entire folder to the target machine.
            2. Open a terminal in this folder.
            3. Run:
                    ollama create {model_name} -f Modelfile --experimental

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
                "trained_model_file": "fused_model",
                "source_output_dir": str(output_dir.resolve()),
                "import_command": f"ollama create {model_name} -f Modelfile --experimental",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return bundle_dir


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_file)
    memory_dataset_path = Path(args.memory_dataset_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.memory_only:
        args.skip_memory_dataset = False
    elif not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        raise SystemExit(1)

    if args.memory_only:
        train_base_rows = []
        filtered_base_rows = []
        base_filter_stats = {"source": "base", "input_rows": 0, "skipped_too_long": 0, "skipped_low_signal": 0}
        train_rows = []
    else:
        train_base_rows = list(select_rows(read_rows(dataset_path), args.offset, args.limit))
        filtered_base_rows, base_filter_stats = filter_training_rows(train_base_rows, args, "base")
        train_rows = list(filtered_base_rows)

    memory_rows = []
    skipped_empty = 0
    skipped_truncated = 0
    memory_filter_stats = {
        "source": "memory",
        "input_rows": 0,
        "skipped_too_long": 0,
        "skipped_low_signal": 0,
        "capped_rows": 0,
    }
    if not args.skip_memory_dataset and memory_dataset_path.exists():
        raw_memory_rows, skipped_empty, skipped_truncated = read_memory_rows(memory_dataset_path)
        memory_rows, memory_filter_core_stats = filter_training_rows(raw_memory_rows, args, "memory")
        memory_filter_stats.update(memory_filter_core_stats)
        memory_rows, capped_rows = limit_memory_rows(memory_rows, len(filtered_base_rows))
        memory_filter_stats["capped_rows"] = capped_rows
        train_rows.extend(memory_rows)
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
        if memory_filter_stats["skipped_too_long"]:
            print(
                f"Skipped {memory_filter_stats['skipped_too_long']} extra memory rows that were too long for the current context budget",
                file=sys.stderr,
            )
        if memory_filter_stats["skipped_low_signal"]:
            print(
                f"Skipped {memory_filter_stats['skipped_low_signal']} extra memory rows that looked like attachment/import tests or low-signal acknowledgements",
                file=sys.stderr,
            )
        if memory_filter_stats["capped_rows"]:
            print(
                f"Skipped {memory_filter_stats['capped_rows']} extra memory rows to keep memory rows from outnumbering the base dataset",
                file=sys.stderr,
            )

    if base_filter_stats["skipped_too_long"]:
        print(
            f"Skipped {base_filter_stats['skipped_too_long']} base rows that were too long for the current context budget",
            file=sys.stderr,
        )
    if base_filter_stats["skipped_low_signal"]:
        print(
            f"Skipped {base_filter_stats['skipped_low_signal']} base rows that looked like low-signal test data",
            file=sys.stderr,
        )

    train_rows, duplicate_rows = deduplicate_training_rows(train_rows)
    if duplicate_rows:
        print(
            f"Skipped {duplicate_rows} duplicate training rows across the base and memory datasets",
            file=sys.stderr,
        )
    train_rows = strip_row_quality_metadata(train_rows)

    eval_path = Path(args.eval_file)
    fallback_split_used = False
    if eval_path.exists():
        valid_rows = read_rows(eval_path)
        eval_source = str(eval_path)
    else:
        split_index = max(1, int(len(train_rows) * (1.0 - max(0.0, min(args.validation_split, 0.5)))))
        split_index = min(split_index, len(train_rows) - 1) if len(train_rows) > 1 else 1
        valid_rows = train_rows[split_index:]
        train_rows = train_rows[:split_index]
        if not valid_rows:
            valid_rows = train_rows[-1:]
            train_rows = train_rows[:-1] or train_rows
        eval_source = "split-from-train"
        fallback_split_used = True

    if not train_rows:
        raise ValueError("No training rows were available after filtering.")
    if not valid_rows:
        raise ValueError("No validation rows were available. Supply --eval-file or increase the selected train rows.")

    data_dir = output_dir / "data"
    adapter_path = output_dir / "adapters"
    fused_model_dir = output_dir / "fused_model"
    modelfile_path = output_dir / "Modelfile"
    plan_path = output_dir / "mlx_training_plan.json"

    train_message_rows = [build_message_row(row) for row in train_rows]
    valid_message_rows = [build_message_row(row) for row in valid_rows]
    write_jsonl_rows(data_dir / "train.jsonl", train_message_rows)
    write_jsonl_rows(data_dir / "valid.jsonl", valid_message_rows)
    write_jsonl_rows(data_dir / "test.jsonl", valid_message_rows)

    modelfile_template = fetch_base_modelfile(args.base_ollama_model)
    modelfile_text = rebase_modelfile(modelfile_template, str(fused_model_dir.resolve()), args.context_length)
    modelfile_path.write_text(modelfile_text, encoding="utf-8")

    training_command = build_lora_command(args, data_dir, adapter_path)
    fuse_command = build_fuse_command(args, adapter_path, fused_model_dir)
    ollama_command = [
        "ollama",
        "create",
        args.ollama_model_name,
        "-f",
        str(modelfile_path.resolve()),
        "--experimental",
    ]

    plan_payload = {
        "model": args.mlx_model,
        "train_file": str(dataset_path),
        "eval_file": eval_source,
        "memory_dataset_file": str(memory_dataset_path),
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "adapter_path": str(adapter_path),
        "fused_model_dir": str(fused_model_dir),
        "modelfile_path": str(modelfile_path),
        "base_train_samples": len(train_base_rows),
        "filtered_base_train_samples": len(filtered_base_rows),
        "memory_train_samples": len(memory_rows),
        "skipped_memory_empty": skipped_empty,
        "skipped_memory_truncated": skipped_truncated,
        "skipped_base_too_long": base_filter_stats["skipped_too_long"],
        "skipped_base_low_signal": base_filter_stats["skipped_low_signal"],
        "skipped_memory_too_long": memory_filter_stats["skipped_too_long"],
        "skipped_memory_low_signal": memory_filter_stats["skipped_low_signal"],
        "skipped_memory_capped": memory_filter_stats["capped_rows"],
        "train_samples": len(train_rows),
        "valid_samples": len(valid_rows),
        "training_command": command_to_string(training_command),
        "fuse_command": command_to_string(fuse_command),
        "ollama_command": command_to_string(ollama_command),
        "ollama_import_source": str(fused_model_dir),
        "create_ollama_model": args.create_ollama_model,
        "ollama_model_name": args.ollama_model_name,
        "template_ollama_model": args.base_ollama_model,
        "effective_parameters": {
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "max_seq_length": args.context_length,
            "val_batches": args.val_batches,
            "grad_checkpoint": args.grad_checkpoint,
            "mask_prompt": args.mask_prompt,
            "memory_only": args.memory_only,
        },
        "automatic_adjustments": ["validation-split-fallback"] if fallback_split_used else [],
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    print(f"Wrote MLX train data with {len(train_rows)} samples to {data_dir / 'train.jsonl'}")
    print(f"Wrote MLX validation data with {len(valid_rows)} samples to {data_dir / 'valid.jsonl'}")
    print(f"Wrote Ollama Modelfile to {modelfile_path}")
    print(f"Wrote training plan to {plan_path}")

    if args.prepare_only:
        print("Preparation complete. No training command was run because --prepare-only was set.")
        return

    run_mlx_command(training_command, cwd=output_dir, dry_run=args.dry_run, device=args.device)
    run_mlx_command(fuse_command, cwd=output_dir, dry_run=args.dry_run, device=args.device)

    if args.dry_run:
        print("Dry run complete. No MLX training, fusion, or Ollama import was executed.")
        return

    if not fused_model_dir.exists():
        raise RuntimeError(
            f"mlx_lm fuse finished but {fused_model_dir} was not created. Check the MLX output above."
        )

    print(f"Training output is available at {fused_model_dir}")

    if args.create_ollama_model:
        ollama = ensure_ollama_available()
        run_command([
            ollama,
            "create",
            args.ollama_model_name,
            "-f",
            str(modelfile_path.resolve()),
            "--experimental",
        ], dry_run=False)
        print(f"Created Ollama model: {args.ollama_model_name}")
    else:
        print(
            "To register the trained fused model with Ollama later, run:\n"
            f"  ollama create {args.ollama_model_name} -f {shlex.quote(str(modelfile_path.resolve()))} --experimental"
        )

    if args.export_model_dir:
        bundle_dir = export_model_bundle(
            args.export_model_dir,
            args.ollama_model_name,
            fused_model_dir,
            modelfile_text,
            output_dir,
        )
        print(f"Exported portable model bundle to {bundle_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)