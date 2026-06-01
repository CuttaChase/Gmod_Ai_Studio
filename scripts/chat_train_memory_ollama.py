#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SYSTEM_PROMPT = (
    "You are an expert Garry's Mod Lua assistant. Follow the user's latest instruction exactly. "
    "If the user asks for a specific format, length, or exact text, obey that literally and do not add anything else. "
    "Answer using Lua 5.1 and Garry's Mod API behavior accurately, concisely, and with working examples when useful."
)
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_GENERATE_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_KEEP_ALIVE = "15m"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PORTABLE_BUNDLE_MODEL_PREFIX = "bundle:"
DEFAULT_MEMORY_FILE = Path("training_runs/chat_memory/session_memory.json")
DEFAULT_TRAINING_LOG = Path("output/chat_memory_train.jsonl")
DEFAULT_MESSAGES_LOG = Path("output/chat_memory_messages.jsonl")
DEFAULT_NUM_PREDICT = 768
MAIN_TURN_WARM_TIMEOUT = 45.0
MAIN_TURN_RECOVERY_HISTORY_LIMIT = 8
ATTACHMENT_PROMPT_MAX_FILES = float('inf')
ATTACHMENT_PROMPT_TOTAL_CHARS = float('inf')
ATTACHMENT_PROMPT_PER_FILE_CHARS = float('inf')
ATTACHMENT_PROMPT_MIN_FILE_CHARS = 0
ATTACHMENT_PROMPT_HEAD_CHARS = float('inf')
ATTACHMENT_PROMPT_TAIL_CHARS = float('inf')
ATTACHMENT_HISTORY_MAX_FILES = 8
ATTACHMENT_TURN_MAX_HISTORY_MESSAGES = 12
ATTACHMENT_TURN_MAX_NUM_PREDICT = 1_536
ATTACHMENT_TURN_LARGE_PROMPT_THRESHOLD = 6_000
LITERAL_REPLY_PREFIXES = (
    "reply with exactly:",
    "respond with exactly:",
    "answer with exactly:",
    "output exactly:",
    "say exactly:",
)
MEMORY_REFRAME_SYSTEM_PROMPT = (
    "You compress conversation memory for a Garry's Mod coding assistant. Rewrite the durable context as concise bullet points. "
    "Keep only facts that will matter later: user goals, code facts, file paths, constraints, unresolved work, and stable preferences. "
    "Do not copy whole code blocks or full file contents. Do not repeat transient pleasantries or runtime metrics. "
    "Every bullet must be directly supported by the current long-term memory or the recent conversation turns. Do not infer, invent, or generalize beyond the provided text. "
    "If a section has nothing durable yet, write '- None yet.' Use these headings exactly: Goals, Code Facts, Open Work, Preferences."
)
MEMORY_REFRAME_MAX_CHARS = 2400
MEMORY_REFRAME_RECENT_MESSAGES = 12
MEMORY_REFRAME_MESSAGE_CHAR_LIMIT = 600
MEMORY_REFRAME_MAX_NUM_PREDICT = 192
MEMORY_REFRAME_MAX_TIMEOUT = 60.0
MEMORY_REFRAME_IGNORE_TOKENS = {
    "goals",
    "code",
    "facts",
    "open",
    "work",
    "preferences",
    "latest",
    "request",
    "assistant",
    "response",
    "durable",
    "memory",
    "conversation",
    "coding",
    "future",
    "none",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with a local Ollama model, keep persistent memory, and save each exchange as future training data."
    )
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY_FILE))
    parser.add_argument("--training-log-file", default=str(DEFAULT_TRAINING_LOG))
    parser.add_argument("--messages-log-file", default=str(DEFAULT_MESSAGES_LOG))
    parser.add_argument("--max-history-messages", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument("--device", default="auto", help="Inference device preference for Ollama chat requests: auto, cpu, metal, cuda.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--once")
    return parser.parse_args()


def normalize_reframed_memory(text):
    normalized = normalize_text_block(text).strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if len(normalized) <= MEMORY_REFRAME_MAX_CHARS:
        return normalized
    truncated = normalized[:MEMORY_REFRAME_MAX_CHARS].rstrip()
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0].rstrip()
    return (truncated or normalized[:MEMORY_REFRAME_MAX_CHARS].rstrip()) + "\n- Older details truncated."


def load_memory_payload(path):
    if not path.exists():
        return {
            "model": "",
            "updated_at": "",
            "messages": [],
            "reframed_memory": "",
            "reframed_at": "",
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid memory file format: {path}")
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"Invalid memory file format: {path}")
    return {
        "model": str(payload.get("model") or ""),
        "updated_at": str(payload.get("updated_at") or ""),
        "messages": messages,
        "reframed_memory": normalize_reframed_memory(payload.get("reframed_memory") or payload.get("memory_summary") or ""),
        "reframed_at": str(payload.get("reframed_at") or ""),
    }


def load_memory(path):
    return load_memory_payload(path)["messages"]


def save_memory(path, model, messages, reframed_memory=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "model": model,
        "updated_at": created_at,
        "messages": messages,
    }
    cleaned_reframed_memory = normalize_reframed_memory(reframed_memory)
    if cleaned_reframed_memory:
        payload["reframed_memory"] = cleaned_reframed_memory
        payload["reframed_at"] = created_at
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_history(messages, max_history_messages):
    if max_history_messages <= 0:
        return messages
    return messages[-max_history_messages:]


def normalize_text_block(text):
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def normalize_user_text(text):
    return normalize_text_block(text).strip()


def normalize_attachment_path(value):
    cleaned = str(value or "").replace("\\", "/").strip()
    return cleaned.lstrip("/")


def normalize_attachments(raw_attachments):
    if not isinstance(raw_attachments, list):
        return []

    attachments = []
    seen = set()
    for raw_attachment in raw_attachments:
        if not isinstance(raw_attachment, dict):
            continue
        attachment_path = normalize_attachment_path(
            raw_attachment.get("relative_path") or raw_attachment.get("path")
        )
        attachment_name = normalize_attachment_path(raw_attachment.get("name"))
        attachment_label = attachment_path or attachment_name or "attachment.txt"
        attachment_name = Path(attachment_label).name or attachment_label
        attachment_content = normalize_text_block(
            raw_attachment.get("content") or raw_attachment.get("text")
        )
        if not attachment_content.strip():
            continue
        try:
            attachment_size = int(raw_attachment.get("size") or 0)
        except (TypeError, ValueError):
            attachment_size = 0
        if attachment_size <= 0:
            attachment_size = len(attachment_content.encode("utf-8"))
        try:
            attachment_char_count = int(raw_attachment.get("char_count") or 0)
        except (TypeError, ValueError):
            attachment_char_count = 0
        if attachment_char_count <= 0:
            attachment_char_count = len(attachment_content)
        try:
            attachment_line_count = int(raw_attachment.get("line_count") or 0)
        except (TypeError, ValueError):
            attachment_line_count = 0
        if attachment_line_count <= 0:
            attachment_line_count = attachment_content.count("\n") + 1
        dedupe_key = (attachment_label, attachment_size, attachment_content)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        attachments.append(
            {
                "name": attachment_name,
                "path": attachment_label,
                "type": str(raw_attachment.get("type") or raw_attachment.get("mime_type") or "").strip(),
                "size": attachment_size,
                "char_count": attachment_char_count,
                "line_count": attachment_line_count,
                "content": attachment_content,
            }
        )
    return attachments


def compact_attachment_content_for_prompt(content, char_budget):
    normalized = normalize_text_block(content).strip("\n")
    if not normalized:
        return "", False
    if len(normalized) <= char_budget:
        return normalized, False

    marker = "\n... [truncated for live chat] ...\n"
    available = char_budget - len(marker)
    if available <= 240:
        return normalized[:char_budget].rstrip(), True

    head_budget = min(ATTACHMENT_PROMPT_HEAD_CHARS, max(200, (available * 2) // 3))
    tail_budget = min(ATTACHMENT_PROMPT_TAIL_CHARS, max(120, available - head_budget))
    if head_budget + tail_budget > available:
        tail_budget = max(0, available - head_budget)

    excerpt = normalized[:head_budget].rstrip()
    if tail_budget > 0:
        excerpt += marker + normalized[-tail_budget:].lstrip()
    return excerpt.rstrip(), True


def compact_attachments_for_prompt(attachments):
    compacted = []
    remaining_total = ATTACHMENT_PROMPT_TOTAL_CHARS
    omitted_files = 0
    truncated_files = 0

    for attachment in attachments:
        # No file or char budget limits
        char_budget = float('inf')
        compact_content, truncated = compact_attachment_content_for_prompt(
            attachment.get("content"),
            char_budget,
        )
        if not compact_content.strip():
            omitted_files += 1
            continue
        compact_attachment = dict(attachment)
        compact_attachment["content"] = compact_content
        compact_attachment["prompt_truncated"] = truncated
        compacted.append(compact_attachment)
        remaining_total -= len(compact_content)
        if truncated:
            truncated_files += 1

    return compacted, {
        "included_files": len(compacted),
        "omitted_files": omitted_files,
        "truncated_files": truncated_files,
    }


def build_attachment_history_summary(attachments):
    if not attachments:
        return ""

    labels = []
    for attachment in attachments[:ATTACHMENT_HISTORY_MAX_FILES]:
        label = str(attachment.get("path") or attachment.get("name") or "attachment.txt").strip()
        line_count = int(attachment.get("line_count") or 0)
        if line_count > 0:
            label += f" ({line_count} lines)"
        labels.append(label)

    summary = "Attached files: " + "; ".join(labels)
    if len(attachments) > ATTACHMENT_HISTORY_MAX_FILES:
        summary += f"; +{len(attachments) - ATTACHMENT_HISTORY_MAX_FILES} more"
    return summary


def build_attachment_prompt(attachments):
    if not attachments:
        return "", {
            "included_files": 0,
            "omitted_files": 0,
            "truncated_files": 0,
        }

    compacted_attachments, prompt_info = compact_attachments_for_prompt(attachments)
    if not compacted_attachments:
        return "", prompt_info

    parts = ["Attached files:"]
    for index, attachment in enumerate(compacted_attachments, start=1):
        header = f"[FILE {index}: {attachment['path']}"
        line_count = int(attachment.get("line_count") or 0)
        char_count = int(attachment.get("char_count") or len(attachment.get("content") or ""))
        if line_count > 0:
            header += f" | {line_count} lines"
        if char_count > 0:
            header += f" | {char_count} chars"
        if attachment.get("prompt_truncated"):
            header += " | excerpted for live chat"
        header += "]"
        parts.extend(
            [
                "",
                header,
                attachment["content"],
                f"[END FILE {index}]",
            ]
        )
    if prompt_info["omitted_files"]:
        parts.extend(
            [
                "",
                f"[{prompt_info['omitted_files']} more attached file(s) omitted from the live prompt to keep generation stable.]",
            ]
        )
    return "\n".join(parts).strip(), prompt_info


def build_user_turn(user_text, attachments=None):
    normalized_user_text = normalize_user_text(user_text)
    normalized_attachments = normalize_attachments(attachments)

    prompt_parts = []
    attachment_prompt_info = {
        "included_files": 0,
        "omitted_files": 0,
        "truncated_files": 0,
    }
    if normalized_user_text:
        prompt_parts.append(normalized_user_text)
    if normalized_attachments:
        attachment_prompt, attachment_prompt_info = build_attachment_prompt(normalized_attachments)
        if attachment_prompt:
            prompt_parts.append(attachment_prompt)

    prompt_content = "\n\n".join(part for part in prompt_parts if part).strip()
    if not prompt_content:
        raise ValueError("Message cannot be empty.")

    display_content = normalized_user_text
    if not display_content and normalized_attachments:
        count = len(normalized_attachments)
        suffix = "file" if count == 1 else "files"
        display_content = f"Shared {count} attached {suffix}."

    history_parts = []
    if normalized_user_text:
        history_parts.append(normalized_user_text)
    attachment_history_summary = build_attachment_history_summary(normalized_attachments)
    if attachment_history_summary:
        history_parts.append(attachment_history_summary)
    history_content = "\n\n".join(part for part in history_parts if part).strip()

    attachment_metadata = [
        {
            "name": attachment["name"],
            "path": attachment["path"],
            "type": attachment["type"],
            "size": attachment["size"],
            "char_count": attachment["char_count"],
            "line_count": attachment["line_count"],
        }
        for attachment in normalized_attachments
    ]

    return {
        "display_content": display_content or prompt_content,
        "history_content": history_content or display_content or prompt_content,
        "prompt_content": prompt_content,
        "attachments": attachment_metadata,
        "attachment_prompt_compacted": bool(
            attachment_prompt_info["truncated_files"] or attachment_prompt_info["omitted_files"]
        ),
        "attachment_files_included": int(attachment_prompt_info["included_files"]),
        "attachment_files_omitted": int(attachment_prompt_info["omitted_files"]),
    }


def get_user_history_content(message):
    return normalize_user_text(
        message.get("history_content") or message.get("content") or message.get("prompt_content")
    )


def build_runtime_history(history, max_history_messages):
    runtime_history = []
    for message in trim_history(history, max_history_messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            content = get_user_history_content(message)
        else:
            content = normalize_user_text(message.get("content"))
        if not content:
            continue
        runtime_history.append({"role": role, "content": content})
    return runtime_history


def excerpt_memory_text(text, limit=MEMORY_REFRAME_MESSAGE_CHAR_LIMIT):
    normalized = normalize_text_block(text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def build_reframed_memory_source_text(existing_reframed_memory, history):
    recent_lines = []
    for message in trim_history(history, MEMORY_REFRAME_RECENT_MESSAGES):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            content = get_user_history_content(message)
        else:
            content = normalize_user_text(message.get("content"))
        if not content:
            continue
        recent_lines.append(f"{role.upper()}: {excerpt_memory_text(content)}")

    source_parts = []
    if existing_reframed_memory:
        source_parts.append("Current long-term memory:\n" + existing_reframed_memory)
    if recent_lines:
        source_parts.append("Recent conversation turns to merge:\n" + "\n\n".join(recent_lines))
    return "\n\n".join(part for part in source_parts if part).strip()


def build_reframed_memory_messages(existing_reframed_memory, history):
    source_text = build_reframed_memory_source_text(existing_reframed_memory, history)
    user_parts = []
    if source_text:
        user_parts.append(source_text)
    user_parts.append("Rewrite the long-term memory now. Keep it compact, grounded, and useful for future coding requests.")

    return [
        {"role": "system", "content": MEMORY_REFRAME_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(part for part in user_parts if part).strip()},
    ]


def extract_reframed_memory_tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9_./:-]{4,}", str(text or "").lower())
        if token not in MEMORY_REFRAME_IGNORE_TOKENS
    }


def is_reframed_memory_grounded(reframed_memory, existing_reframed_memory, history):
    summary_tokens = extract_reframed_memory_tokens(reframed_memory)
    if not summary_tokens:
        return False
    source_tokens = extract_reframed_memory_tokens(
        build_reframed_memory_source_text(existing_reframed_memory, history)
    )
    if not source_tokens:
        return False
    overlap = summary_tokens & source_tokens
    if not overlap:
        return False
    min_overlap_tokens = 1 if len(summary_tokens) <= 4 else 2
    if len(overlap) < min_overlap_tokens:
        return False
    overlap_ratio = len(overlap) / max(1, len(summary_tokens))
    return overlap_ratio >= 0.2


def build_reframed_memory_fallback(history):
    latest_user = ""
    latest_assistant = ""
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role == "assistant" and not latest_assistant:
            latest_assistant = excerpt_memory_text(message.get("content"), limit=220)
        elif role == "user" and not latest_user:
            latest_user = excerpt_memory_text(get_user_history_content(message), limit=220)
        if latest_user and latest_assistant:
            break

    parts = ["Goals"]
    if latest_user:
        parts.append(f"- Latest request: {latest_user}")
    else:
        parts.append("- No durable user goals captured yet.")
    parts.extend([
        "",
        "Code Facts",
        f"- Latest assistant response: {latest_assistant}" if latest_assistant else "- No assistant output captured yet.",
        "",
        "Open Work",
        "- Review the latest request and decide what should persist as a durable task or constraint.",
        "",
        "Preferences",
        "- No stable user preferences captured yet.",
    ])
    return normalize_reframed_memory("\n".join(parts))


def refresh_reframed_memory(args, history, existing_reframed_memory=""):
    if not history:
        return ""
    if len(build_runtime_history(history, MEMORY_REFRAME_RECENT_MESSAGES)) <= 2:
        return build_reframed_memory_fallback(history)

    summary_messages = build_reframed_memory_messages(existing_reframed_memory, history)
    try:
        result = call_ollama(
            model=args.model,
            messages=summary_messages,
            temperature=min(float(args.temperature), 0.2),
            seed=args.seed,
            timeout=min(float(args.timeout), MEMORY_REFRAME_MAX_TIMEOUT),
            enable_thinking=False,
            num_predict=min(max(int(args.num_predict or MEMORY_REFRAME_MAX_NUM_PREDICT), 64), MEMORY_REFRAME_MAX_NUM_PREDICT),
        )
        reframed_memory = normalize_reframed_memory(result.get("message", {}).get("content", ""))
        if reframed_memory and is_reframed_memory_grounded(reframed_memory, existing_reframed_memory, history):
            return reframed_memory
    except Exception:
        pass

    if existing_reframed_memory:
        return existing_reframed_memory
    return build_reframed_memory_fallback(history)


def extract_literal_reply_text(user_text):
    normalized = " ".join(normalize_user_text(user_text).split())
    lowered = normalized.lower()
    for prefix in LITERAL_REPLY_PREFIXES:
        if lowered.startswith(prefix):
            literal_text = normalized[len(prefix):].strip()
            if literal_text:
                return literal_text
    return ""


def build_turn_override(user_text):
    literal_text = extract_literal_reply_text(user_text)
    if literal_text:
        return (
            "This turn is a literal-output request. Ignore the normal Garry's Mod assistant style and "
            f"reply with exactly this text and nothing else: {literal_text}"
        )
    return ""


def build_request_body(model, messages, temperature, seed, enable_thinking, num_predict, device="auto"):
    options = {
        "num_predict": num_predict,
        "temperature": temperature,
        "seed": seed,
    }
    normalized_device = str(device or "auto").strip().lower()
    if normalized_device and normalized_device != "auto":
        options["device"] = normalized_device
    return {
        "model": model,
        "stream": False,
        "think": enable_thinking,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": messages,
        "options": options,
    }


def request_ollama_json(api_url, payload, timeout, failure_label):
    encoded_payload = json.dumps(payload).encode("utf-8")
    request = Request(
        api_url,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body.strip()
        try:
            parsed = json.loads(body)
            detail = parsed.get("error") or detail
        except json.JSONDecodeError:
            pass
        if detail:
            raise RuntimeError(f"{failure_label} with HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"{failure_label} with HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://127.0.0.1:11434. Start the Ollama app or run `ollama serve`, then try again."
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"{failure_label} timed out after {timeout} seconds. Lower --num-predict, use a smaller model, or raise --timeout."
        ) from exc


def is_portable_bundle_model(model):
    return str(model or "").strip().startswith(PORTABLE_BUNDLE_MODEL_PREFIX)


def resolve_portable_bundle_dir(model):
    cleaned_model = str(model or "").strip()
    if not is_portable_bundle_model(cleaned_model):
        return None
    bundle_path = cleaned_model[len(PORTABLE_BUNDLE_MODEL_PREFIX):].strip()
    if not bundle_path:
        raise RuntimeError("Portable bundle model path cannot be empty.")
    resolved_path = Path(bundle_path)
    if not resolved_path.is_absolute():
        resolved_path = (REPO_ROOT / resolved_path).resolve()
    if not resolved_path.exists() or not resolved_path.is_dir():
        raise RuntimeError(f"Portable bundle not found: {resolved_path}")
    return resolved_path


def resolve_repo_python_executable():
    candidates = []
    if os.name == "nt":
        candidates.extend([
            REPO_ROOT / ".venv" / "Scripts" / "python.exe",
            REPO_ROOT / ".venv" / "Scripts" / "python",
        ])
    else:
        candidates.extend([
            REPO_ROOT / ".venv" / "bin" / "python3",
            REPO_ROOT / ".venv" / "bin" / "python",
        ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable or "python3"


def call_portable_bundle_model(model, messages, temperature, seed, timeout, num_predict, warmup=False):
    bundle_dir = resolve_portable_bundle_dir(model)
    command = [
        resolve_repo_python_executable(),
        str(SCRIPTS_DIR / "run_portable_bundle_chat.py"),
        "--bundle-dir",
        str(bundle_dir),
        "--temperature",
        str(temperature),
        "--seed",
        str(seed),
        "--num-predict",
        str(max(1, int(num_predict))),
    ]
    if warmup:
        command.append("--warmup")
    else:
        command.extend(["--messages-json", json.dumps(messages)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Portable bundle request timed out after {timeout} seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run the portable bundle helper: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Portable bundle request failed. {detail}")
    if not stdout:
        raise RuntimeError("Portable bundle request returned no output.")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Portable bundle request returned invalid JSON. {stdout}") from exc


def call_ollama(model, messages, temperature, seed, timeout, enable_thinking, num_predict, device="auto"):
    if is_portable_bundle_model(model):
        return call_portable_bundle_model(
            model,
            messages,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            num_predict=num_predict,
        )
    return request_ollama_json(
        OLLAMA_API_URL,
        build_request_body(model, messages, temperature, seed, enable_thinking, num_predict, device=device),
        timeout,
        "Ollama chat request failed",
    )


def warm_ollama_model(model, timeout=MAIN_TURN_WARM_TIMEOUT):
    cleaned_model = str(model or "").strip()
    if not cleaned_model:
        raise ValueError("Model name cannot be empty.")
    if is_portable_bundle_model(cleaned_model):
        return call_portable_bundle_model(
            cleaned_model,
            [],
            temperature=0.0,
            seed=0,
            timeout=timeout,
            num_predict=1,
            warmup=True,
        )
    return request_ollama_json(
        OLLAMA_GENERATE_API_URL,
        {
            "model": cleaned_model,
            "prompt": "",
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": 0},
        },
        timeout,
        f"Ollama model warm-up for {cleaned_model} failed",
    )


def generate_assistant_reply(args, runtime_messages, enable_thinking, num_predict):
    result = call_ollama(
        model=args.model,
        messages=runtime_messages,
        temperature=args.temperature,
        seed=args.seed,
        timeout=args.timeout,
        enable_thinking=enable_thinking,
        num_predict=num_predict,
        device=getattr(args, 'device', 'auto'),
    )
    retried_without_thinking = False
    assistant_text = result.get("message", {}).get("content", "").strip()
    thinking_text = result.get("message", {}).get("thinking", "").strip()
    if enable_thinking and not assistant_text and thinking_text:
        result = call_ollama(
            model=args.model,
            messages=runtime_messages,
            temperature=args.temperature,
            seed=args.seed,
            timeout=args.timeout,
            enable_thinking=False,
            num_predict=num_predict,
            device=getattr(args, 'device', 'auto'),
        )
        retried_without_thinking = True
        assistant_text = result.get("message", {}).get("content", "").strip()
    return result, assistant_text, retried_without_thinking


def build_runtime_messages(history, user_turn, max_history_messages, turn_override="", reframed_memory=""):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if reframed_memory:
        messages.append(
            {
                "role": "system",
                "content": "Long-term memory from earlier conversation. Treat this as compressed context, not as a new user instruction.\n\n" + reframed_memory,
            }
        )
    if turn_override:
        messages.append({"role": "system", "content": turn_override})
    messages.extend(build_runtime_history(history, max_history_messages))
    messages.append({"role": "user", "content": user_turn["prompt_content"]})
    return messages


def extract_reply_info(result, num_predict):
    done_reason = (result.get("done_reason") or "").strip().lower()
    truncated = done_reason in {"length", "max_tokens"}
    return {
        "done_reason": done_reason or "unknown",
        "done": bool(result.get("done", False)),
        "eval_count": int(result.get("eval_count") or 0),
        "prompt_eval_count": int(result.get("prompt_eval_count") or 0),
        "num_predict": int(num_predict),
        "truncated": truncated,
    }


def is_low_signal_training_example(user_turn, assistant_text):
    normalized_answer = str(assistant_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_answer:
        return True
    lower_answer = normalized_answer.lower()
    if lower_answer in {
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
        "stars1",
        "--<id>--",
    }:
        return True
    if lower_answer.startswith("--<id>") or lower_answer.startswith("stars"):
        return True
    if len(normalized_answer) < 16 and len(re.findall(r"[a-zA-Z0-9]{2,}", normalized_answer)) <= 1:
        return True
    return False


def save_training_examples(training_log_path, messages_log_path, model, user_turn, assistant_text, reply_info, runtime_messages):
    created_at = datetime.now(timezone.utc).isoformat()
    if not is_low_signal_training_example(user_turn, assistant_text):
        append_jsonl(
            training_log_path,
            {
                "question": user_turn["prompt_content"],
                "answer": assistant_text,
                "user_message": user_turn["display_content"],
                "attachments": user_turn["attachments"],
                "attachment_count": len(user_turn["attachments"]),
                "source": "chat-memory",
                "model": model,
                "created_at": created_at,
                "done_reason": reply_info["done_reason"],
                "eval_count": reply_info["eval_count"],
                "num_predict": reply_info["num_predict"],
                "truncated": reply_info["truncated"],
            },
        )
    append_jsonl(
        messages_log_path,
        {
            "messages": runtime_messages + [{"role": "assistant", "content": assistant_text}],
            "user_message": user_turn["display_content"],
            "attachments": user_turn["attachments"],
            "attachment_count": len(user_turn["attachments"]),
            "source": "chat-memory",
            "model": model,
            "created_at": created_at,
            "done_reason": reply_info["done_reason"],
            "eval_count": reply_info["eval_count"],
            "num_predict": reply_info["num_predict"],
            "truncated": reply_info["truncated"],
        },
    )


def run_turn(args, history, user_text, attachments=None):
    literal_reply_text = extract_literal_reply_text(user_text)
    memory_payload = load_memory_payload(Path(args.memory_file))
    existing_reframed_memory = memory_payload.get("reframed_memory") or ""
    user_turn = build_user_turn(user_text, attachments)
    turn_override = build_turn_override(user_text)
    attachment_heavy_turn = bool(user_turn["attachments"]) and (
        user_turn["attachment_prompt_compacted"]
        or len(user_turn["prompt_content"]) >= ATTACHMENT_TURN_LARGE_PROMPT_THRESHOLD
    )
    effective_history_limit = int(args.max_history_messages)
    effective_num_predict = int(args.num_predict)
    effective_enable_thinking = bool(args.enable_thinking)
    if attachment_heavy_turn:
        effective_history_limit = min(effective_history_limit, ATTACHMENT_TURN_MAX_HISTORY_MESSAGES)
        effective_num_predict = min(effective_num_predict, ATTACHMENT_TURN_MAX_NUM_PREDICT)
        effective_enable_thinking = False

    recovery_history_limit = max(2, min(effective_history_limit, MAIN_TURN_RECOVERY_HISTORY_LIMIT))
    attempt_plans = [
        {
            "label": "primary",
            "warm_model": False,
            "history_limit": effective_history_limit,
            "reframed_memory": existing_reframed_memory,
            "enable_thinking": effective_enable_thinking,
        },
        {
            "label": "warm-retry",
            "warm_model": True,
            "history_limit": effective_history_limit,
            "reframed_memory": existing_reframed_memory,
            "enable_thinking": False,
        },
    ]
    if existing_reframed_memory or effective_history_limit > recovery_history_limit:
        attempt_plans.append(
            {
                "label": "warm-retry-short-history",
                "warm_model": True,
                "history_limit": recovery_history_limit,
                "reframed_memory": "",
                "enable_thinking": False,
            }
        )

    started_at = time.time()
    result = None
    assistant_text = ""
    retried_without_thinking = False
    runtime_messages = []
    successful_attempt = None
    attempt_log = []

    for attempt_plan in attempt_plans:
        warm_error = ""
        if attempt_plan["warm_model"]:
            try:
                warm_ollama_model(args.model, timeout=min(float(args.timeout), MAIN_TURN_WARM_TIMEOUT))
            except RuntimeError as exc:
                warm_error = str(exc)

        runtime_messages = build_runtime_messages(
            history,
            user_turn,
            attempt_plan["history_limit"],
            turn_override=turn_override,
            reframed_memory=attempt_plan["reframed_memory"],
        )
        attempt_entry = {
            "label": attempt_plan["label"],
            "warm_model": attempt_plan["warm_model"],
            "warm_error": warm_error,
            "history_limit": attempt_plan["history_limit"],
            "used_reframed_memory": bool(attempt_plan["reframed_memory"]),
        }
        try:
            result, assistant_text, retried_without_thinking = generate_assistant_reply(
                args,
                runtime_messages,
                enable_thinking=attempt_plan["enable_thinking"],
                num_predict=effective_num_predict,
            )
            if assistant_text:
                attempt_entry["success"] = True
                attempt_entry["retried_without_thinking"] = retried_without_thinking
                attempt_log.append(attempt_entry)
                successful_attempt = attempt_plan
                break
            attempt_entry["success"] = False
            attempt_entry["error"] = "The model returned an empty answer."
        except RuntimeError as exc:
            attempt_entry["success"] = False
            attempt_entry["error"] = str(exc)
        attempt_log.append(attempt_entry)

    elapsed_seconds = round(time.time() - started_at, 3)
    if successful_attempt is None or result is None or not assistant_text:
        failure_reasons = [
            entry.get("error") or entry.get("warm_error")
            for entry in attempt_log
            if entry.get("error") or entry.get("warm_error")
        ]
        detail = " | ".join(failure_reasons).strip()
        if detail:
            raise RuntimeError(f"All prompt generation recovery attempts failed. {detail}")
        raise RuntimeError("All prompt generation recovery attempts failed.")

    reply_info = extract_reply_info(result, effective_num_predict)
    reply_info["retried_without_thinking"] = retried_without_thinking
    reply_info["literal_output_enforced"] = False
    reply_info["prompt_char_count"] = len(user_turn["prompt_content"])
    reply_info["attachment_count"] = len(user_turn["attachments"])
    reply_info["attachment_prompt_compacted"] = bool(user_turn["attachment_prompt_compacted"])
    reply_info["attachment_files_included"] = int(user_turn["attachment_files_included"])
    reply_info["attachment_files_omitted"] = int(user_turn["attachment_files_omitted"])
    reply_info["generation_attempts"] = len(attempt_log)
    reply_info["generation_recovery_used"] = successful_attempt["label"] != "primary"
    reply_info["generation_strategy"] = successful_attempt["label"]
    reply_info["warm_retry_used"] = bool(successful_attempt["warm_model"])
    reply_info["reframed_memory_bypassed"] = bool(existing_reframed_memory) and not bool(successful_attempt["reframed_memory"])
    reply_info["history_limit_used"] = int(successful_attempt["history_limit"])
    reply_info["num_predict_requested"] = int(args.num_predict)
    reply_info["num_predict_clamped"] = effective_num_predict != int(args.num_predict)
    reply_info["thinking_forced_off"] = bool(args.enable_thinking) and not effective_enable_thinking
    if literal_reply_text and assistant_text != literal_reply_text:
        assistant_text = literal_reply_text
        reply_info["literal_output_enforced"] = True
    history.extend(
        [
            {
                "role": "user",
                "content": user_turn["display_content"],
                "history_content": user_turn["history_content"],
                "prompt_content": user_turn["prompt_content"],
                "attachments": user_turn["attachments"],
            },
            {"role": "assistant", "content": assistant_text},
        ]
    )
    if reply_info["generation_recovery_used"]:
        reframed_memory = existing_reframed_memory or build_reframed_memory_fallback(history)
    else:
        reframed_memory = refresh_reframed_memory(args, history, existing_reframed_memory=existing_reframed_memory)
    reply_info["memory_reframed"] = bool(reframed_memory)
    reply_info["memory_reframed_changed"] = reframed_memory != existing_reframed_memory
    reply_info["reframed_memory_chars"] = len(reframed_memory)
    save_memory(Path(args.memory_file), args.model, history, reframed_memory=reframed_memory)
    save_training_examples(
        Path(args.training_log_file),
        Path(args.messages_log_file),
        args.model,
        user_turn,
        assistant_text,
        reply_info,
        runtime_messages,
    )
    return assistant_text, elapsed_seconds, reply_info


def print_help():
    print("Commands: /exit, /clear, /help, /train")


def main():
    args = parse_args()
    memory_path = Path(args.memory_file)
    history = load_memory(memory_path)

    if args.once:
        assistant_text, elapsed_seconds, reply_info = run_turn(args, history, args.once)
        print(assistant_text)
        print(f"\n[{elapsed_seconds}s] saved memory to {memory_path}", file=sys.stderr)
        if reply_info["truncated"]:
            print(
                f"Reply hit the num_predict limit ({reply_info['num_predict']}). Increase --num-predict if you want longer answers.",
                file=sys.stderr,
            )
        return

    print(f"Model: {args.model}")
    print(f"Memory file: {memory_path}")
    print(f"Training log: {args.training_log_file}")
    print(f"Messages log: {args.messages_log_file}")
    print_help()

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/clear":
            history = []
            save_memory(memory_path, args.model, history)
            print("Memory cleared.")
            continue
        if user_text == "/train":
            print("Run: python3 scripts/train_llamacpp_ollama_macos.py")
            continue

        assistant_text, elapsed_seconds, reply_info = run_turn(args, history, user_text)
        print(f"ai> {assistant_text}\n")
        print(f"[{elapsed_seconds}s] memory and training logs updated", file=sys.stderr)
        if reply_info["truncated"]:
            print(
                f"Reply hit the num_predict limit ({reply_info['num_predict']}). Increase --num-predict if you want longer answers.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)