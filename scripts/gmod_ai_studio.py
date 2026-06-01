#!/usr/bin/env python3
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import queue
import re
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from copy import deepcopy
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SETTINGS_PATH = REPO_ROOT / "gmod_ai_studio_settings.json"
PROMPTS_PATH = REPO_ROOT / "readme" / "GMOD_TRAINING_PROMPTS.md"
USERS_PATH = REPO_ROOT / "gmod_ai_studio_users.json"
CHAT_USERS_DIR = REPO_ROOT / "training_runs" / "chat_memory" / "users"
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUT_DIR = REPO_ROOT / "output"
UPLOADED_FILES_DATASET_PATH = DATASETS_DIR / "uploaded_files_train.jsonl"
MERGED_MEMORY_DATASET_PATH = OUTPUT_DIR / "chat_and_uploaded_memory_train.jsonl"
DATASET_SCRIPT_PATH = "scripts/prepare_training_data.py"
MAC_TRAINING_SCRIPT_PATH = "scripts/train_mlx_gmod_macos.py"
WINDOWS_TRAINING_SCRIPT_PATH = "scripts/train_llamacpp_ollama_windows.py"
SESSION_COOKIE_NAME = "gmod_ai_studio_session"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")
PASSWORD_HASH_ITERATIONS = 200_000
OLLAMA_GENERATE_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_LOAD_TIMEOUT = 90.0
PORTABLE_BUNDLE_MODEL_PREFIX = "bundle:"
TRAINING_UPLOAD_ALLOWED_EXTENSIONS = {
  ".lua",
  ".txt",
  ".json",
  ".cfg",
  ".ini",
  ".md",
  ".log",
  ".xml",
  ".yaml",
  ".yml",
  ".toml",
  ".csv",
  ".properties",
}
TEXT_UPLOAD_MIME_TYPES = {
  "application/json",
  "application/xml",
  "text/xml",
  "application/x-yaml",
}
if str(SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_DIR))

from chat_train_memory_ollama import build_user_turn, load_memory, load_memory_payload, normalize_attachments, run_turn, save_memory

TRAINING_PLATFORM_OPTIONS = [
  {"value": "mac", "label": "Mac (MLX / Ollama)"},
  {"value": "windows", "label": "Windows (llama.cpp / Ollama)"},
]
WINDOWS_ACCELERATION_OPTIONS = [
  {"value": "auto", "label": "Auto"},
  {"value": "cpu", "label": "CPU Only"},
  {"value": "cuda", "label": "CUDA GPU"},
]
MAC_ACCELERATION_OPTIONS = [
  {"value": "auto", "label": "Auto"},
  {"value": "cpu", "label": "CPU Only"},
  {"value": "metal", "label": "Metal GPU"},
]
CHAT_DEVICE_OPTIONS = [
  {"value": "auto", "label": "Auto"},
  {"value": "cpu", "label": "CPU Only"},
  {"value": "metal", "label": "Metal GPU"},
  {"value": "cuda", "label": "CUDA GPU"},
]
USER_ROLE_OPTIONS = [
  {"value": "user", "label": "User"},
  {"value": "admin", "label": "Admin"},
]
MLX_SAFE_MAX_SEQ_LENGTH = 32768  # Remove practical cap, set very high
MLX_SAFE_MAX_BATCH_SIZE_SHORT_CONTEXT = 4
MLX_SAFE_MAX_BATCH_SIZE_LONG_CONTEXT = 2
MLX_SAFE_MAX_VAL_BATCHES = 2
DEFAULT_MAC_MLX_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

DEFAULT_SETTINGS = {
    "app": {
    "host": "0.0.0.0",
        "port": 8765,
        "auto_open_browser": True,
        "poll_interval_ms": 2500,
    },
    "chat": {
      "model": "ministral-3:3b",
        "memory_file": "training_runs/chat_memory/session_memory.json",
        "training_log_file": "output/chat_memory_train.jsonl",
        "messages_log_file": "output/chat_memory_messages.jsonl",
        "max_history_messages": 24,
      "temperature": 0.1,
        "seed": 3407,
        "timeout": 240.0,
        "num_predict": 768,
        "enable_thinking": False,
        "device": "auto",
    },
    "dataset": {
      "script": DATASET_SCRIPT_PATH,
        "extra_args": "",
    },
    "training": {
      "script": MAC_TRAINING_SCRIPT_PATH,
        "platform": "mac",
      "dataset_file": "datasets/gmod_lua_coding_train.jsonl",
        "dataset_files": ["datasets/gmod_lua_coding_train.jsonl"],
        "eval_file": "datasets/gmod_lua_coding_eval.jsonl",
        "mlx_model": DEFAULT_MAC_MLX_MODEL,
        "base_model_gguf": "",
        "base_ollama_model": "llama3.2:1b",
        "output_dir": "training_runs/llamacpp-llama32-1b-mac",
        "export_model_bundle": True,
        "export_model_dir": "models",
        "memory_dataset_file": "output/chat_memory_train.jsonl",
        "skip_memory_dataset": False,
        "limit": 256,
        "offset": 0,
        "context_length": 512,
        "batch_size": 64,
        "ubatch_size": 16,
        "epochs": 1,
        "iters": 300,
        "grad_accumulation_steps": 4,
        "num_layers": 8,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "val_batches": 2,
        "save_every": 50,
        "gpu_layers": 0,
        "threads": max(1, (os.cpu_count() or 4) - 1),
        "seed": 3407,
        "validation_split": 0.1,
        "windows_acceleration": "auto",
        "mac_acceleration": "auto",
        "llama_bin_dir": "",
        "prepare_only": False,
        "create_ollama_model": True,
        "ollama_model_name": "gmod-llama32-1b-llamacpp",
        "grad_checkpoint": True,
        "mask_prompt": True,
        "dry_run": False,
        "extra_args": "",
    },
}


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GMod AI Studio</title>
  <style>
    :root {
      --bg: #11161b;
      --panel: #192128;
      --panel-alt: #222d36;
      --line: #30414d;
      --text: #e8f0f4;
      --muted: #9eb0bc;
      --accent: #8ee36b;
      --accent-2: #5ec8ff;
      --warn: #ffc857;
      --danger: #ff6b6b;
      --code: #0c1013;
      --shadow: 0 18px 42px rgba(0, 0, 0, 0.32);
      --radius: 16px;
      --font: "SF Pro Display", "Segoe UI", sans-serif;
      --mono: "SF Mono", "Cascadia Code", monospace;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(142, 227, 107, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(94, 200, 255, 0.14), transparent 24%),
        linear-gradient(180deg, #0c1116, #121921 24%, #0f151b 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 24px;
      padding: 24px;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(25,33,40,0.96), rgba(34,45,54,0.92));
      box-shadow: var(--shadow);
    }

    .hero h1 {
      margin: 0 0 10px 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }

    .hero-card {
      min-width: 280px;
      padding: 18px;
      border-radius: 18px;
      background: rgba(11, 15, 19, 0.58);
      border: 1px solid rgba(255,255,255,0.06);
    }

    .hero-label {
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 24px;
      align-items: start;
    }

    .chat-panel {
      grid-column: 1 / -1;
    }

    .stack {
      display: grid;
      gap: 24px;
    }

    .panel {
      border: 1px solid rgba(255,255,255,0.06);
      background: linear-gradient(180deg, rgba(25,33,40,0.96), rgba(19,25,31,0.95));
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 18px 20px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
    }

    .panel-title {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
    }

    .panel-body {
      padding: 18px 20px 20px;
    }

    .muted {
      color: var(--muted);
      font-size: 0.93rem;
      line-height: 1.5;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 0.84rem;
      background: rgba(142, 227, 107, 0.09);
      border: 1px solid rgba(142, 227, 107, 0.18);
      color: var(--text);
    }

    .status-pill.running {
      background: rgba(94, 200, 255, 0.12);
      border-color: rgba(94, 200, 255, 0.22);
    }

    .status-pill.error {
      background: rgba(255, 107, 107, 0.12);
      border-color: rgba(255, 107, 107, 0.22);
    }

    .row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }

    .row.tight { gap: 8px; }

    button {
      appearance: none;
      border: 0;
      border-radius: 12px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      color: #081015;
      background: linear-gradient(135deg, var(--accent), #b8ff96);
      transition: transform 120ms ease, filter 120ms ease;
      font-weight: 700;
    }

    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #b6e7ff);
    }

    button.ghost {
      color: var(--text);
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }

    button.warn {
      background: linear-gradient(135deg, var(--warn), #ffd88e);
    }

    button:disabled {
      filter: grayscale(0.45);
      opacity: 0.6;
      cursor: wait;
    }

    button:hover:not(:disabled) { transform: translateY(-1px); }

    textarea, input[type="text"], input[type="password"], input[type="number"], select {
      width: 100%;
      padding: 12px 13px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(8, 12, 16, 0.72);
      color: var(--text);
      font: inherit;
    }

    select {
      appearance: none;
    }

    textarea {
      resize: vertical;
      min-height: 110px;
    }

    label {
      display: block;
      font-size: 0.86rem;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .field-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .field-grid.compact {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .field-grid .full { grid-column: 1 / -1; }

    .checkbox {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.05);
    }

    .checkbox input { width: auto; }

    details {
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      margin-top: 14px;
      background: rgba(255,255,255,0.02);
    }

    summary {
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-weight: 700;
    }

    summary::-webkit-details-marker { display: none; }

    .details-body {
      padding: 0 16px 16px;
    }

    .chat-log {
      min-height: 320px;
      max-height: 560px;
      overflow: auto;
      display: grid;
      gap: 12px;
      padding-right: 4px;
      margin-bottom: 14px;
    }

    .bubble {
      padding: 14px 15px;
      border-radius: 16px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .bubble.user {
      background: rgba(94, 200, 255, 0.14);
      border: 1px solid rgba(94, 200, 255, 0.18);
    }

    .bubble.assistant {
      background: rgba(142, 227, 107, 0.12);
      border: 1px solid rgba(142, 227, 107, 0.18);
    }

    .bubble-meta {
      font-size: 0.77rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .bubble-files {
      margin-top: 10px;
      font-size: 0.78rem;
      color: var(--muted);
    }

    .attachment-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
      max-height: 220px;
      overflow: auto;
    }

    .attachment-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
    }

    .attachment-row strong {
      display: block;
      font-size: 0.88rem;
      color: var(--text);
      word-break: break-all;
    }

    .attachment-row span {
      display: block;
      margin-top: 4px;
      font-size: 0.76rem;
      color: var(--muted);
      word-break: break-all;
    }

    .attachment-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-shrink: 0;
    }

    .attachment-actions button {
      padding: 8px 12px;
      font-size: 0.82rem;
    }

    .composer-actions {
      margin-top: 12px;
    }

    .composer-meta {
      flex: 1 1 260px;
      min-width: 0;
    }

    .attachment-trigger {
      min-width: 48px;
      padding-left: 0;
      padding-right: 0;
      font-size: 1.35rem;
      line-height: 1;
    }

    pre {
      margin: 0;
      padding: 16px;
      border-radius: 14px;
      background: var(--code);
      border: 1px solid rgba(255,255,255,0.05);
      color: #cde4ee;
      overflow: auto;
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.85rem;
      line-height: 1.45;
    }

    .prompt-library {
      max-height: 500px;
      overflow: auto;
      padding-right: 4px;
      white-space: pre-wrap;
      line-height: 1.55;
      color: #d9e4ea;
    }

    .split {
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr 1fr;
    }

    .tiny {
      font-size: 0.78rem;
      color: var(--muted);
    }

    [hidden] {
      display: none !important;
    }

    .login-panel {
      max-width: 760px;
      margin: 0 auto 24px;
    }

    .user-list {
      display: grid;
      gap: 10px;
      max-height: 240px;
      overflow: auto;
      margin-bottom: 14px;
    }

    .user-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.03);
    }

    .user-row strong {
      display: block;
      margin-bottom: 4px;
    }

    .user-role {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 760px) {
      .shell { padding: 16px; }
      .hero { flex-direction: column; align-items: stretch; }
      .field-grid, .field-grid.compact, .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="hero-label">Local Cross-Platform Studio</div>
        <h1>GMod AI Studio</h1>
        <p>Chat with the model, collect future training rows, rebuild the dataset, and launch training from one browser panel. Settings are saved to a JSON file so you can keep custom arguments and paths for macOS or Windows.</p>
      </div>
      <div class="hero-card">
        <div class="hero-label">Current Setup</div>
        <div id="currentModel">Model: loading...</div>
        <div style="margin-top:10px;"><span id="modelLoadStatus" class="status-pill">Checking model...</span></div>
        <div id="currentDataset" class="muted" style="margin-top:8px;">Dataset: loading...</div>
        <div id="authSummary" class="muted" style="margin-top:8px;">Session: loading...</div>
        <div class="row" style="margin-top:14px;">
          <button id="adminUsersButton" class="ghost" hidden>Admin</button>
          <button id="logoutButton" class="ghost" hidden>Logout</button>
        </div>
      </div>
    </section>

    <section id="loginPanel" class="panel login-panel" hidden>
      <div class="panel-header">
        <h2 class="panel-title">Sign In</h2>
        <span id="loginStatus" class="status-pill">Ready</span>
      </div>
      <div class="panel-body field-grid">
        <div><label>Username<input id="loginUsername" type="text" autocomplete="username"></label></div>
        <div><label>Password<input id="loginPassword" type="password" autocomplete="current-password"></label></div>
        <div class="full row">
          <button id="loginButton">Sign In</button>
          <span class="tiny">Localhost connections stay admin automatically.</span>
        </div>
      </div>
    </section>

    <div id="mainGrid" class="grid">
      <section class="panel chat-panel">
        <div class="panel-header">
          <h2 class="panel-title">Chat Trainer</h2>
          <div class="row tight">
            <span id="chatStatus" class="status-pill">Ready</span>
            <button id="clearChatButton" class="ghost">Clear Memory</button>
          </div>
        </div>
        <div class="panel-body">
          <div id="chatLog" class="chat-log"></div>
          <label for="chatInput">Message</label>
          <textarea id="chatInput" placeholder="Ask for Garry's Mod Lua code, debugging help, or addon ideas. Add files when you want the model and future training rows to include your code."></textarea>
          <input id="chatFileInput" type="file" multiple accept=".lua,.txt,.json,.cfg,.ini,.md,.log,.xml,.yaml,.yml,.toml" hidden>
          <input id="chatFolderInput" type="file" webkitdirectory multiple hidden>
          <div class="row composer-actions">
            <button id="chatAttachFilesButton" class="secondary attachment-trigger" type="button" title="Attach Lua or text files" aria-label="Attach files">+</button>
            <button id="sendChatButton">Send Message</button>
            <button id="chatAttachFolderButton" class="ghost" type="button" title="Attach a folder so many addon files can be included at once">Folder</button>
            <button id="chatClearFilesButton" class="ghost" type="button" hidden>Clear Files</button>
            <span id="chatAttachmentMeta" class="tiny composer-meta">No files attached.</span>
          </div>
          <div id="chatAttachmentList" class="attachment-list" hidden></div>
          <div class="tiny" style="margin-top:10px;">Use + to attach Lua or text files. Folder skips obvious binary assets, prioritizes addon code files, and saves supported files into the training store even before you send a chat message.</div>
          <div class="row tight" style="margin-top:12px;">
            <span id="chatMeta" class="tiny"></span>
          </div>
        </div>
      </section>

      <div id="adminPanelStack" class="stack admin-only">
        <section class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Dataset And Training</h2>
            <span id="taskStatus" class="status-pill">Idle</span>
          </div>
          <div class="panel-body">
            <div class="split">
              <div>
                <div class="muted">Rebuild the wiki-derived dataset and curated concept rows before training.</div>
                <div class="row" style="margin-top:14px;">
                  <button id="rebuildDatasetButton" class="secondary">Rebuild Dataset</button>
                </div>
                <div class="tiny" style="margin-top:10px;">Command preview</div>
                <pre id="datasetCommandPreview"></pre>
              </div>
              <div>
                <div class="muted">Launch the local training wrapper with the current settings and extra arguments.</div>
                <div class="row" style="margin-top:14px;">
                  <button id="startTrainingButton" class="warn">Start Training</button>
                </div>
                <div class="tiny" style="margin-top:10px;">Command preview</div>
                <pre id="trainingCommandPreview"></pre>
              </div>
            </div>
            <div class="tiny" style="margin-top:14px;">Task output</div>
            <pre id="taskLog">No task output yet.</pre>
          </div>
        </section>
      </div>

      <div id="adminSettingsStack" class="stack admin-only">
        <section id="userManagementPanel" class="panel" hidden>
          <div class="panel-header">
            <h2 class="panel-title">Admin Users</h2>
            <span class="status-pill">Access</span>
          </div>
          <div class="panel-body">
            <div class="muted" style="margin-bottom:12px;">Create remote users, reset passwords, and assign admin or user rights. Localhost always keeps admin access.</div>
            <div id="userList" class="user-list"></div>
            <div class="tiny" style="margin-bottom:10px;">Saving an existing username updates its role. Leave the password blank to keep the current password.</div>
            <div class="field-grid compact">
              <div><label>Username<input id="managedUsername" type="text" autocomplete="off"></label></div>
              <div><label>Password<input id="managedPassword" type="password" autocomplete="new-password"></label></div>
              <div><label>Role<select id="managedRole"></select></label></div>
              <div class="full row">
                <button id="saveUserButton" class="secondary">Save User</button>
                <button id="resetUserFormButton" class="ghost">Clear Form</button>
                <span id="userManagerStatus" class="tiny"></span>
              </div>
            </div>
          </div>
        </section>

        <section id="settingsPanel" class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Settings</h2>
            <div class="row tight">
              <button id="saveSettingsButton">Save Settings</button>
            </div>
          </div>
          <div class="panel-body">
            <details open>
              <summary>Chat Settings</summary>
              <div class="details-body field-grid compact">
                <div class="full"><label>Model<input id="chatModel" list="chatModelList" type="text" placeholder="Type or pick an Ollama model"></label><datalist id="chatModelList"></datalist></div>
                <div><label>Timeout<input id="chatTimeout" type="number" step="0.1"></label></div>
                <div><label>Num Predict<input id="chatNumPredict" type="number"></label></div>
                <div><label>Device<select id="chatDevice"></select></label></div>
                <div><label>History Messages<input id="chatHistory" type="number"></label></div>
                <div><label>Temperature<input id="chatTemperature" type="number" step="0.05"></label></div>
                <div><label>Seed<input id="chatSeed" type="number"></label></div>
                <div class="full checkbox"><input id="chatEnableThinking" type="checkbox"><label style="margin:0;">Enable Thinking</label></div>
                <div class="full"><label>Memory File<input id="chatMemoryFile" type="text"></label></div>
                <div class="full"><label>Training Log File<input id="chatTrainingLogFile" type="text"></label></div>
                <div class="full"><label>Messages Log File<input id="chatMessagesLogFile" type="text"></label></div>
              </div>
            </details>

            <details>
              <summary>Dataset Settings</summary>
              <div class="details-body field-grid">
                <div class="full"><label>Dataset Builder (automatic)<input id="datasetScript" type="text" readonly></label></div>
                <div class="full"><div class="muted">Use Training Settings to pick one or more train datasets. Rebuild Dataset uses the built-in dataset builder automatically.</div></div>
                <div class="full"><label>Extra Args<input id="datasetExtraArgs" type="text" placeholder="Example: --dry-run"></label></div>
              </div>
            </details>

            <details>
              <summary>Training Settings</summary>
              <div class="details-body field-grid compact">
                <div><label>Training Target<select id="trainingPlatform"></select></label></div>
                <div class="full"><div id="trainingPlatformHint" class="muted">Mac uses MLX LoRA fine-tuning and imports the fused model into Ollama. Windows uses the llama.cpp / Ollama wrapper.</div></div>
                <div class="full platform-windows"><label>Training Script<input id="trainingScript" type="text"></label></div>
                <div class="full"><label>Train Datasets<select id="trainingDatasetFile" multiple size="6"></select></label></div>
                <div class="full platform-mac"><label>MLX Base Model<input id="trainingMlxModel" type="text" placeholder="Example: mlx-community/Llama-3.2-1B-Instruct-4bit"></label></div>
                <div class="full platform-mac"><label>Mac Acceleration<select id="trainingMacAcceleration"></select></label></div>
                <div class="full platform-windows"><label>Base GGUF Path<input id="trainingBaseModelGguf" type="text"></label></div>
                <div><label>Base Ollama Model<input id="trainingBaseOllamaModel" list="trainingBaseOllamaModelList" type="text" placeholder="Type or pick an Ollama model"></label><datalist id="trainingBaseOllamaModelList"></datalist></div>
                <div class="platform-windows"><label>Windows Acceleration<select id="trainingWindowsAcceleration"></select></label></div>
                <div><label>Output Dir<input id="trainingOutputDir" type="text"></label></div>
                <div><label>Memory Dataset File<input id="trainingMemoryDatasetFile" type="text"></label></div>
                <div><label>Limit<input id="trainingLimit" type="number"></label></div>
                <div><label>Offset<input id="trainingOffset" type="number"></label></div>
                <div><label>Max Seq / Context<input id="trainingContextLength" type="number" min="1" max="32768"></label></div>
                <div><label>Batch Size<input id="trainingBatchSize" type="number"></label></div>
                <div class="platform-mac"><label>Iterations<input id="trainingIters" type="number"></label></div>
                <div class="platform-mac"><label>Grad Accumulation<input id="trainingGradAccumulationSteps" type="number"></label></div>
                <div class="platform-mac"><label>Train Layers<input id="trainingNumLayers" type="number"></label></div>
                <div class="platform-mac"><label>Report Every<input id="trainingStepsPerReport" type="number"></label></div>
                <div class="platform-mac"><label>Eval Every<input id="trainingStepsPerEval" type="number"></label></div>
                <div class="platform-mac"><label>Val Batches<input id="trainingValBatches" type="number"></label></div>
                <div class="platform-mac"><label>Save Every<input id="trainingSaveEvery" type="number"></label></div>
                <div class="platform-windows"><label>UBatch Size<input id="trainingUBatchSize" type="number"></label></div>
                <div class="platform-windows"><label>Epochs<input id="trainingEpochs" type="number"></label></div>
                <div class="platform-windows"><label>GPU Layers<input id="trainingGpuLayers" type="number"></label></div>
                <div class="platform-windows"><label>Threads<input id="trainingThreads" type="number"></label></div>
                <div><label>Seed<input id="trainingSeed" type="number"></label></div>
                <div class="platform-windows"><label>Validation Split<input id="trainingValidationSplit" type="number" step="0.01"></label></div>
                <div class="full platform-windows"><label>llama.cpp Bin Dir<input id="trainingLlamaBinDir" type="text"></label></div>
                <div><label>Ollama Model Name<input id="trainingOllamaModelName" type="text"></label></div>
                <div class="full"><label>Portable Export Dir<input id="trainingExportModelDir" type="text"></label></div>
                <div class="full"><label>Extra Args<input id="trainingExtraArgs" type="text" placeholder="Example: --dry-run --gpu-layers 20"></label></div>
                <div class="checkbox"><input id="trainingSkipMemoryDataset" type="checkbox"><label style="margin:0;">Skip Memory Dataset</label></div>
                <div class="checkbox"><input id="trainingMemoryOnly" type="checkbox"><label style="margin:0;">Memory-only Training</label></div>
                <div class="checkbox"><input id="trainingPrepareOnly" type="checkbox"><label style="margin:0;">Prepare Only</label></div>
                <div class="checkbox"><input id="trainingCreateOllamaModel" type="checkbox"><label style="margin:0;">Create Ollama Model</label></div>
                <div class="checkbox"><input id="trainingExportModelBundle" type="checkbox"><label style="margin:0;">Export Portable Model Bundle</label></div>
                <div class="checkbox"><input id="trainingDryRun" type="checkbox"><label style="margin:0;">Dry Run</label></div>
              </div>
            </details>

            <details>
              <summary>Application Settings</summary>
              <div class="details-body field-grid compact">
                <div><label>Host<input id="appHost" type="text"></label></div>
                <div><label>Port<input id="appPort" type="number"></label></div>
                <div><label>Poll Interval (ms)<input id="appPollInterval" type="number"></label></div>
                <div class="full checkbox"><input id="appAutoOpenBrowser" type="checkbox"><label style="margin:0;">Auto Open Browser On Start</label></div>
              </div>
            </details>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Prompt Library</h2>
            <span class="status-pill">Reusable</span>
          </div>
          <div class="panel-body">
            <div class="muted" style="margin-bottom:12px;">These prompts can be turned into new question/answer rows for future training data.</div>
            <div id="promptLibrary" class="prompt-library">Loading prompts...</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Saved Models</h2>
            <span id="localModelStatus" class="status-pill">Ready</span>
          </div>
          <div class="panel-body">
            <div class="muted" style="margin-bottom:12px;">Load trained bundles directly from the configured models folder and switch Studio chat to the portable model without copying it into Ollama.</div>
            <div id="localModelLibrary" class="user-list"></div>
          </div>
        </section>
      </div>
    </div>
  </div>

  <script>
    const state = {
      auth: null,
      settings: null,
      options: null,
      pollTimer: null,
      chatAttachments: [],
      settingsLoaded: false,
      settingsDirty: false,
      optionsLoaded: false,
      promptsLoaded: false,
      lastChatSignature: '',
      lastTaskSignature: '',
    };

    const CHAT_REQUEST_ATTACHMENT_MAX_FILES = Infinity;
    const CHAT_REQUEST_ATTACHMENT_TOTAL_CHARS = Infinity;
    const CHAT_REQUEST_ATTACHMENT_PER_FILE_CHARS = Infinity;
    const CHAT_REQUEST_ATTACHMENT_MIN_FILE_CHARS = 0;
    const CHAT_REQUEST_ATTACHMENT_HEAD_CHARS = Infinity;
    const CHAT_REQUEST_ATTACHMENT_TAIL_CHARS = Infinity;

    const FIELD_TOOLTIPS = Object.freeze({
      chatModel: 'Type or select an Ollama model for chat replies in the studio. Supports both local models and cloud model names.',
      chatTimeout: 'Maximum seconds to wait for a chat reply before the request fails.',
      chatNumPredict: 'Upper limit for generated tokens in each assistant reply.',
      chatHistory: 'How many recent messages are sent back to the model for context.',
      chatTemperature: 'Controls randomness. Lower is steadier, higher is more varied.',
      chatSeed: 'Repeatable random seed for chat generation when the model supports it.',
      chatEnableThinking: 'Allows models with thinking support to return reasoning-style output when available.',
      chatMemoryFile: 'Local JSON file used for the localhost admin chat history.',
      chatTrainingLogFile: 'Shared JSONL file where prompt-response training rows are appended.',
      chatMessagesLogFile: 'Shared JSONL file where raw chat messages are logged turn by turn.',
      datasetScript: 'Built-in dataset builder used by the Rebuild Dataset button. This is kept visible for advanced troubleshooting but is not the main dataset selection control.',
      datasetExtraArgs: 'Extra command-line flags passed directly to the dataset script.',
      trainingPlatform: 'Chooses whether the studio should use the Mac MLX trainer or the Windows llama.cpp wrapper.',
      trainingScript: 'Training launcher script used for the selected platform.',
      trainingDatasetFile: 'Select one or more training datasets. If you select multiple files, the studio combines them into one prepared training file before training starts.',
      trainingMlxModel: 'Mac-only MLX base model repo or local path used for LoRA fine-tuning on Apple Silicon.',
      trainingBaseModelGguf: 'Optional path to a base GGUF model file for the llama.cpp training path.',
      trainingBaseOllamaModel: 'Type or select an Ollama model to use as the base for training. Supports both local and cloud models.',
      trainingWindowsAcceleration: 'Windows-only training backend preference. Auto falls back to CPU if CUDA is not available.',
      trainingOutputDir: 'Folder where checkpoints, merged weights, and import files are written.',
      trainingMemoryDatasetFile: 'Extra JSONL dataset built from chat memory. Stored uploaded files are merged into this automatically at training time.',
      trainingLimit: 'Maximum number of base dataset rows to include before memory rows are appended.',
      trainingOffset: 'Number of base dataset rows to skip before training rows are selected.',
      trainingContextLength: 'Maximum token context window used during training.',
      trainingBatchSize: 'Mini-batch size per training step. Higher uses more memory.',
      trainingIters: 'Mac-only MLX training iterations. Lower values are better for a quick first run.',
      trainingGradAccumulationSteps: 'Mac-only MLX gradient accumulation steps before each optimizer update.',
      trainingNumLayers: 'Mac-only number of layers fine-tuned by MLX LoRA.',
      trainingStepsPerReport: 'Mac-only log reporting interval in training steps.',
      trainingStepsPerEval: 'Mac-only validation interval in training steps.',
      trainingValBatches: 'Mac-only number of validation batches used by MLX.',
      trainingSaveEvery: 'Mac-only checkpoint save interval in training steps.',
      trainingUBatchSize: 'Micro-batch size for the Windows llama.cpp training path.',
      trainingEpochs: 'How many passes over the selected dataset to run on the Windows path.',
      trainingGpuLayers: 'How many model layers to place on GPU in the llama.cpp training path.',
      trainingThreads: 'CPU thread count used by the llama.cpp training tools.',
      trainingSeed: 'Random seed for dataset ordering and training reproducibility.',
      trainingValidationSplit: 'Fraction of the selected dataset reserved for validation inside the llama.cpp wrapper.',
      trainingLlamaBinDir: 'Folder containing llama.cpp binaries if they are not already on PATH.',
      trainingOllamaModelName: 'Name used when creating or importing the trained Ollama model.',
      trainingExportModelDir: 'Directory where the app writes a portable fused-model bundle plus Modelfile for moving the trained model to other systems.',
      trainingExtraArgs: 'Extra command-line flags appended to the training command.',
      trainingSkipMemoryDataset: 'Disables appending the extra memory datasets, including chat memory and stored uploaded files, into the training run.',
      trainingMemoryOnly: 'Use only the chat/memory dataset for training, ignoring the base dataset selection.',
      trainingPrepareOnly: 'Prepares dataset/output files but skips the actual training run.',
      trainingCreateOllamaModel: 'Automatically imports the finished model into Ollama after training.',
      trainingExportModelBundle: 'Copies the trained fused-model bundle, a portable Modelfile, and import notes into the models folder after training completes.',
      trainingDryRun: 'Prints the training command without launching the real run.',
      appHost: 'Network interface the studio web server binds to.',
      appPort: 'Port the studio web server listens on.',
      appPollInterval: 'How often the browser refreshes studio state from the server.',
      appAutoOpenBrowser: 'Automatically opens the studio in your browser when the server starts.',
    });

    const els = {};

    function $(id) {
      if (!els[id]) {
        els[id] = document.getElementById(id);
      }
      return els[id];
    }

    function setStatus(el, text, kind) {
      if (!el) {
        return;
      }
      el.textContent = text;
      el.className = 'status-pill' + (kind ? ' ' + kind : '');
    }

    function setUserManagerStatus(text, isError = false) {
      const el = $("userManagerStatus");
      if (!el) {
        return;
      }
      el.textContent = text || '';
      el.style.color = isError ? '#ffb6b6' : '';
    }

    function applyFieldTooltips() {
      Object.entries(FIELD_TOOLTIPS).forEach(([id, text]) => {
        const field = $(id);
        if (!field) {
          return;
        }
        field.title = text;
        const parentLabel = field.closest('label');
        if (parentLabel) {
          parentLabel.title = text;
        }
        const siblingLabel = field.nextElementSibling && field.nextElementSibling.tagName === 'LABEL'
          ? field.nextElementSibling
          : null;
        if (siblingLabel) {
          siblingLabel.title = text;
        }
      });
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function formatBytes(bytes) {
      const value = Number(bytes) || 0;
      if (value <= 0) {
        return '0 B';
      }
      const units = ['B', 'KB', 'MB', 'GB'];
      let unitIndex = 0;
      let size = value;
      while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
      }
      const digits = size >= 10 || unitIndex === 0 ? 0 : 1;
      return size.toFixed(digits) + ' ' + units[unitIndex];
    }

    function countTextLines(text) {
      const normalized = String(text || '').replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n');
      if (!normalized) {
        return 0;
      }
      return normalized.split('\\n').length;
    }

    function compactAttachmentContentForChat(content, charBudget) {
      const normalized = String(content || '').replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n').trim();
      if (!normalized) {
        return {text: '', truncated: false};
      }
      if (normalized.length <= charBudget) {
        return {text: normalized, truncated: false};
      }

      const marker = '\\n... [truncated for live chat] ...\\n';
      const available = charBudget - marker.length;
      if (available <= 240) {
        return {text: normalized.slice(0, charBudget).trimEnd(), truncated: true};
      }

      const headBudget = Math.min(CHAT_REQUEST_ATTACHMENT_HEAD_CHARS, Math.max(200, Math.floor((available * 2) / 3)));
      let tailBudget = Math.min(CHAT_REQUEST_ATTACHMENT_TAIL_CHARS, Math.max(120, available - headBudget));
      if (headBudget + tailBudget > available) {
        tailBudget = Math.max(0, available - headBudget);
      }

      let text = normalized.slice(0, headBudget).trimEnd();
      if (tailBudget > 0) {
        text += marker + normalized.slice(-tailBudget).trimStart();
      }
      return {text: text.trimEnd(), truncated: true};
    }

    function buildChatRequestAttachments(attachments) {
      const rawAttachments = Array.isArray(attachments) ? attachments : [];
      const compacted = [];
      let remainingTotal = CHAT_REQUEST_ATTACHMENT_TOTAL_CHARS;
      let omittedCount = 0;
      let truncatedCount = 0;

      for (const attachment of rawAttachments) {
        // No file or char budget limits
        const charBudget = Infinity;
        const rawContent = String(attachment.content || attachment.text || '');
        const compactedContent = compactAttachmentContentForChat(rawContent, charBudget);
        if (!compactedContent.text.trim()) {
          omittedCount += 1;
          continue;
        }
        compacted.push({
          name: attachment.name || 'attachment.txt',
          relative_path: attachment.relative_path || attachment.path || '',
          path: attachment.path || attachment.relative_path || attachment.name || 'attachment.txt',
          type: attachment.type || '',
          size: Number(attachment.size || 0),
          char_count: rawContent.length,
          line_count: countTextLines(rawContent),
          content: compactedContent.text,
        });
        remainingTotal -= compactedContent.text.length;
        if (compactedContent.truncated) {
          truncatedCount += 1;
        }
      }

      return {
        attachments: compacted,
        omittedCount,
        truncatedCount,
      };
    }

    const CHAT_UPLOAD_ALLOWED_EXTENSIONS = new Set([
      '.lua', '.txt', '.json', '.cfg', '.ini', '.md', '.log', '.xml', '.yaml', '.yml', '.toml', '.csv', '.properties',
    ]);
    const CHAT_UPLOAD_TEXT_MIME_TYPES = new Set(['application/json', 'application/xml', 'text/xml', 'application/x-yaml']);

    function getChatFileLabel(file) {
      return String((file && (file.webkitRelativePath || file.name)) || '').trim();
    }

    function getChatFileExtension(file) {
      const label = getChatFileLabel(file).toLowerCase();
      const dotIndex = label.lastIndexOf('.');
      if (dotIndex < 0) {
        return '';
      }
      return label.slice(dotIndex);
    }

    function isChatFileSupported(file) {
      const extension = getChatFileExtension(file);
      const mimeType = String((file && file.type) || '').trim().toLowerCase();
      if (CHAT_UPLOAD_ALLOWED_EXTENSIONS.has(extension)) {
        return true;
      }
      if (!extension && (mimeType.startsWith('text/') || CHAT_UPLOAD_TEXT_MIME_TYPES.has(mimeType))) {
        return true;
      }
      return false;
    }

    function getChatFilePriority(file) {
      const label = getChatFileLabel(file).toLowerCase().replaceAll('\\\\', '/');
      const extension = getChatFileExtension(file);
      let score = 100;
      if (extension === '.lua') {
        score -= 60;
      } else if (['.json', '.cfg', '.ini', '.txt', '.md'].includes(extension)) {
        score -= 24;
      }
      if (label.includes('/lua/') || label.startsWith('lua/')) {
        score -= 20;
      }
      if (label.includes('/weapons/') || label.includes('weapon_')) {
        score -= 12;
      }
      if (label.includes('/entities/') || label.includes('/autorun/')) {
        score -= 10;
      }
      return score;
    }

    function attachmentKey(attachment) {
      return [
        attachment.relative_path || attachment.path || '',
        attachment.name || '',
        String(attachment.size || 0),
      ].join('::');
    }

    function clearChatAttachments() {
      state.chatAttachments = [];
      renderChatAttachments();
    }

    function removeChatAttachment(index) {
      state.chatAttachments = state.chatAttachments.filter((_, attachmentIndex) => attachmentIndex !== index);
      renderChatAttachments();
    }

    function summarizeHistoryAttachments(attachments) {
      if (!Array.isArray(attachments) || !attachments.length) {
        return '';
      }
      const names = attachments
        .map((attachment) => attachment.path || attachment.name)
        .filter(Boolean);
      const preview = names.slice(0, 3).join(', ');
      const more = names.length > 3 ? ' +' + (names.length - 3) + ' more' : '';
      const totalBytes = attachments.reduce((sum, attachment) => sum + (Number(attachment.size) || 0), 0);
      return 'Files: ' + preview + more + ' · ' + formatBytes(totalBytes);
    }

    function renderChatAttachments() {
      const attachments = Array.isArray(state.chatAttachments) ? state.chatAttachments : [];
      const list = $("chatAttachmentList");
      const meta = $("chatAttachmentMeta");
      const clearButton = $("chatClearFilesButton");
      if (!attachments.length) {
        list.innerHTML = '';
        list.hidden = true;
        clearButton.hidden = true;
        meta.textContent = 'No files attached.';
        return;
      }

      list.innerHTML = '';
      list.hidden = false;
      clearButton.hidden = false;
      const totalBytes = attachments.reduce((sum, attachment) => sum + (Number(attachment.size) || 0), 0);
      meta.textContent = attachments.length + ' file' + (attachments.length === 1 ? '' : 's') + ' attached · ' + formatBytes(totalBytes);

      attachments.forEach((attachment, index) => {
        const row = document.createElement('div');
        row.className = 'attachment-row';

        const metaWrap = document.createElement('div');
        const title = document.createElement('strong');
        title.textContent = attachment.relative_path || attachment.path || attachment.name;
        const detail = document.createElement('span');
        const detailParts = [formatBytes(attachment.size)];
        if (attachment.type) {
          detailParts.push(attachment.type);
        }
        metaWrap.appendChild(title);
        detail.textContent = detailParts.join(' · ');
        metaWrap.appendChild(detail);

        const actions = document.createElement('div');
        actions.className = 'attachment-actions';
        const removeButton = document.createElement('button');
        removeButton.className = 'ghost';
        removeButton.type = 'button';
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', () => removeChatAttachment(index));
        actions.appendChild(removeButton);

        row.appendChild(metaWrap);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }

    async function readChatFile(file) {
      return {
        name: file.name || 'attachment.txt',
        relative_path: file.webkitRelativePath || '',
        type: file.type || '',
        size: Number(file.size || 0),
        content: await file.text(),
      };
    }

    async function addChatFiles(fileList) {
      const incomingFiles = Array.from(fileList || []);
      if (!incomingFiles.length) {
        return;
      }

      const supportedFiles = incomingFiles
        .filter((file) => isChatFileSupported(file))
        .sort((left, right) => {
          const scoreDiff = getChatFilePriority(left) - getChatFilePriority(right);
          if (scoreDiff) {
            return scoreDiff;
          }
          return getChatFileLabel(left).localeCompare(getChatFileLabel(right));
        });
      const skippedUnsupported = incomingFiles.length - supportedFiles.length;
      if (!supportedFiles.length) {
        setStatus($("chatStatus"), 'Skipped ' + skippedUnsupported + ' unsupported or binary file' + (skippedUnsupported === 1 ? '' : 's'), 'error');
        return;
      }

      const nextAttachments = Array.isArray(state.chatAttachments) ? [...state.chatAttachments] : [];
      const seen = new Set(nextAttachments.map((attachment) => attachmentKey(attachment)));
      let added = 0;
      let duplicates = 0;
      let failed = 0;
      const newlyAddedAttachments = [];
      setStatus($("chatStatus"), 'Loading ' + supportedFiles.length + ' file' + (supportedFiles.length === 1 ? '' : 's') + '...', 'running');
      for (const file of supportedFiles) {
        try {
          const attachment = await readChatFile(file);
          const key = attachmentKey(attachment);
          if (seen.has(key)) {
            duplicates += 1;
            continue;
          }
          seen.add(key);
          nextAttachments.push(attachment);
          newlyAddedAttachments.push(attachment);
          added += 1;
        } catch (error) {
          failed += 1;
        }
      }

      let storeResult = null;
      let storeError = '';
      if (newlyAddedAttachments.length) {
        try {
          storeResult = await getJson('/api/uploads/store', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({attachments: newlyAddedAttachments}),
          });
        } catch (error) {
          storeError = error.message || 'Could not save files for training.';
        }
      }

      state.chatAttachments = nextAttachments;
      renderChatAttachments();
      if (storeResult && storeResult.dataset_file) {
        $("chatMeta").textContent = 'Stored for training in ' + storeResult.dataset_file + ' · staged files: ' + nextAttachments.length;
      }

      if (storeError) {
        setStatus($("chatStatus"), storeError, 'error');
      } else if (failed) {
        setStatus($("chatStatus"), 'Skipped ' + failed + ' unreadable file' + (failed === 1 ? '' : 's'), 'error');
      } else if (added) {
        const parts = [];
        const storedCount = Number((storeResult && storeResult.stored_count) || added || 0);
        parts.push('Saved ' + storedCount + ' file' + (storedCount === 1 ? '' : 's') + ' for training');
        if (duplicates) {
          parts.push('skipped ' + duplicates + ' duplicate' + (duplicates === 1 ? '' : 's'));
        }
        if (skippedUnsupported) {
          parts.push('skipped ' + skippedUnsupported + ' unsupported/binary');
        }
        setStatus($("chatStatus"), parts.join(' · '), '');
        setTimeout(() => {
          if (String($("chatStatus").textContent || '').startsWith('Saved ')) {
            setStatus($("chatStatus"), 'Ready', '');
          }
        }, 1600);
      } else {
        const baseMessage = duplicates
          ? 'No new supported files added · skipped ' + duplicates + ' duplicate' + (duplicates === 1 ? '' : 's')
          : 'No new files added';
        const fullMessage = skippedUnsupported ? baseMessage + ' · skipped ' + skippedUnsupported + ' unsupported/binary' : baseMessage;
        setStatus($("chatStatus"), fullMessage, skippedUnsupported ? 'error' : '');
        setTimeout(() => {
          if (String($("chatStatus").textContent || '').startsWith('No new')) {
            setStatus($("chatStatus"), 'Ready', '');
          }
        }, 1200);
      }
    }

    function normalizeOption(option) {
      if (typeof option === 'string') {
        return {value: option, label: option};
      }
      return {
        value: String(option.value ?? ''),
        label: String(option.label ?? option.value ?? ''),
      };
    }

    function populateSelect(id, rawOptions, currentValue, placeholder = '') {
      const select = $(id);
      if (!select) {
        return;
      }
      const options = [];
      const seen = new Set();

      function appendOption(value, label) {
        const normalizedValue = String(value ?? '');
        if (seen.has(normalizedValue)) {
          return;
        }
        seen.add(normalizedValue);
        options.push({value: normalizedValue, label: String(label ?? normalizedValue)});
      }

      if (placeholder !== null) {
        appendOption('', placeholder);
      }
      if (currentValue) {
        appendOption(currentValue, currentValue);
      }
      (rawOptions || []).forEach((option) => {
        const normalized = normalizeOption(option);
        appendOption(normalized.value, normalized.label);
      });

      select.innerHTML = '';
      options.forEach((option) => {
        const element = document.createElement('option');
        element.value = option.value;
        element.textContent = option.label;
        select.appendChild(element);
      });

      if (currentValue !== undefined && currentValue !== null) {
        select.value = String(currentValue);
      }
    }

    function populateDatalist(id, rawOptions) {
      const datalist = $(id);
      if (!datalist) {
        return;
      }
      const options = [];
      const seen = new Set();

      function appendOption(value) {
        const normalizedValue = String(value ?? '');
        if (!normalizedValue || seen.has(normalizedValue)) {
          return;
        }
        seen.add(normalizedValue);
        options.push(normalizedValue);
      }

      (rawOptions || []).forEach((option) => {
        const normalized = normalizeOption(option);
        appendOption(normalized.value);
      });

      datalist.innerHTML = '';
      options.forEach((value) => {
        const element = document.createElement('option');
        element.value = value;
        datalist.appendChild(element);
      });
    }

    function setMultiSelectValues(id, values) {
      const select = $(id);
      if (!select) {
        return;
      }
      const wanted = new Set((Array.isArray(values) ? values : []).map((value) => String(value)));
      Array.from(select.options).forEach((option) => {
        option.selected = wanted.has(option.value);
      });
    }

    function getMultiSelectValues(id) {
      const select = $(id);
      if (!select) {
        return [];
      }
      return Array.from(select.selectedOptions)
        .map((option) => option.value)
        .filter((value) => String(value || '').trim() !== '');
    }

    function populateMultiSelect(id, rawOptions, currentValues) {
      const select = $(id);
      if (!select) {
        return;
      }

      const values = Array.isArray(currentValues) ? currentValues : [];
      const options = [];
      const seen = new Set();

      function appendOption(value, label) {
        const normalizedValue = String(value ?? '');
        if (!normalizedValue || seen.has(normalizedValue)) {
          return;
        }
        seen.add(normalizedValue);
        options.push({value: normalizedValue, label: String(label ?? normalizedValue)});
      }

      values.forEach((value) => appendOption(value, value));
      (rawOptions || []).forEach((option) => {
        const normalized = normalizeOption(option);
        appendOption(normalized.value, normalized.label);
      });

      select.innerHTML = '';
      options.forEach((option) => {
        const element = document.createElement('option');
        element.value = option.value;
        element.textContent = option.label;
        select.appendChild(element);
      });

      setMultiSelectValues(id, values.length ? values : (options[0] ? [options[0].value] : []));
    }

    function summarizeTrainingDatasets(training) {
      const datasetFiles = Array.isArray(training.dataset_files) && training.dataset_files.length
        ? training.dataset_files
        : [training.dataset_file].filter(Boolean);
      if (!datasetFiles.length) {
        return '';
      }
      if (datasetFiles.length === 1) {
        return datasetFiles[0];
      }
      return datasetFiles.length + ' datasets selected';
    }

    function currentPollInterval() {
      return Number((state.settings && state.settings.app && state.settings.app.poll_interval_ms) || 2500);
    }

    function renderLocalModelBundles(bundles, settings) {
      const list = $("localModelLibrary");
      if (!list) {
        return;
      }
      list.innerHTML = '';

      const modelBundles = Array.isArray(bundles) ? bundles : [];
      if (!modelBundles.length) {
        const empty = document.createElement('div');
        empty.className = 'tiny';
        const exportDir = (settings && settings.training && settings.training.export_model_dir) || 'models';
        empty.textContent = 'No portable model bundles found in ' + exportDir + '.';
        list.appendChild(empty);
        return;
      }

      modelBundles.forEach((bundle) => {
        const row = document.createElement('div');
        row.className = 'user-row';

        const meta = document.createElement('div');
        const title = document.createElement('strong');
        title.textContent = bundle.ollama_model_name;

        const status = document.createElement('div');
        status.className = 'user-role';
        if (bundle.current_chat) {
          status.textContent = 'Current chat model';
        } else {
          status.textContent = 'Portable ' + ((bundle.portable_backend || 'bundle').toUpperCase()) + ' bundle';
        }

        const details = document.createElement('div');
        details.className = 'tiny';
        const detailParts = [bundle.bundle_path];
        if (bundle.source_output_dir) {
          detailParts.push('source: ' + bundle.source_output_dir);
        }
        details.textContent = detailParts.join(' · ');

        meta.appendChild(title);
        meta.appendChild(status);
        meta.appendChild(details);

        const actions = document.createElement('div');
        actions.className = 'row tight';

        const importButton = document.createElement('button');
        importButton.className = bundle.current_chat ? 'secondary' : '';
        importButton.textContent = bundle.current_chat ? 'Reload Portable Model' : 'Load From Models Folder';
        importButton.addEventListener('click', () => loadModelBundle(bundle.bundle_name, bundle.ollama_model_name));
        actions.appendChild(importButton);

        const deleteButton = document.createElement('button');
        deleteButton.className = 'ghost danger';
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', () => deleteModelBundle(bundle.bundle_name));
        actions.appendChild(deleteButton);

        row.appendChild(meta);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }


    function updateLocalModelTaskStatus(task) {
      const statusEl = $("localModelStatus");
      if (!statusEl || !task || task.name !== 'model-import') {
        return;
      }
      if (task.status === 'running') {
        setStatus(statusEl, 'Loading portable bundle...', 'running');
      } else if (task.status === 'completed') {
        setStatus(statusEl, 'Portable bundle loaded', '');
      } else if (task.status === 'failed') {
        setStatus(statusEl, 'Portable bundle load failed', 'error');
      }
    }

    function renderPublic(publicState) {
      const model = publicState && publicState.model ? publicState.model : 'loading...';
      const dataset = publicState && publicState.dataset ? publicState.dataset : 'loading...';
      $("currentModel").textContent = 'Model: ' + model;
      $("currentDataset").textContent = 'Dataset: ' + dataset;
    }

    function renderModelLoadStatus(modelLoad, fallbackModel) {
      const model = (modelLoad && modelLoad.model) || fallbackModel || '';
      const status = (modelLoad && modelLoad.status) || 'idle';
      let text = 'Model not loaded yet';
      let kind = '';

      if (status === 'loading') {
        text = (modelLoad && modelLoad.message) || ('Loading ' + model + '...');
        kind = 'running';
      } else if (status === 'loaded') {
        text = (modelLoad && modelLoad.message) || ('Loaded ' + model);
      } else if (status === 'error') {
        text = (modelLoad && modelLoad.message) || ('Failed to load ' + model);
        kind = 'error';
      } else if (model) {
        text = 'Model selected: ' + model;
      }

      setStatus($("modelLoadStatus"), text, kind);

      const chatStatus = $("chatStatus");
      if (!chatStatus) {
        return;
      }
      if (chatStatus.textContent === 'Model loading...') {
        if (status === 'loaded') {
          setStatus(chatStatus, 'Model loaded', '');
          setTimeout(() => {
            if ($("chatStatus").textContent === 'Model loaded') {
              setStatus($("chatStatus"), 'Ready', '');
            }
          }, 1600);
        } else if (status === 'error') {
          setStatus(chatStatus, 'Model load failed', 'error');
        }
      }
    }

    function updateTrainingPlatformUI() {
      const select = $("trainingPlatform");
      const platform = (select && select.value) || 'mac';
      const isMac = platform === 'mac';
      document.querySelectorAll('.platform-mac').forEach((element) => {
        element.hidden = !isMac;
      });
      document.querySelectorAll('.platform-windows').forEach((element) => {
        element.hidden = isMac;
      });
      const hint = $("trainingPlatformHint");
      if (hint) {
        hint.textContent = platform === 'windows'
          ? 'Windows mode uses the llama.cpp / Ollama wrapper and looks for llama-finetune.exe.'
          : 'Mac mode uses MLX LoRA fine-tuning on Apple Silicon and can load portable bundles directly from the models folder.';
      }
    }

    function applyOptions(options, settings) {
      state.options = options || null;
      if (!options || !settings) {
        return;
      }
      populateDatalist("chatModelList", options.ollama_models);
      populateDatalist("trainingBaseOllamaModelList", options.ollama_models);
      $("chatModel").value = settings.chat.model || '';
      populateSelect("chatDevice", options.chat_device_options, settings.chat.device, null);
      $("trainingBaseOllamaModel").value = settings.training.base_ollama_model || '';
      populateMultiSelect("trainingDatasetFile", options.train_datasets, settings.training.dataset_files);
      populateSelect("trainingPlatform", options.training_platforms, settings.training.platform, null);
      populateSelect("trainingWindowsAcceleration", options.windows_acceleration_options, settings.training.windows_acceleration, null);
      populateSelect("trainingMacAcceleration", options.mac_acceleration_options, settings.training.mac_acceleration, null);
      populateSelect("managedRole", options.user_roles, $("managedRole").value || 'user', null);
    }

    function updateSaveButton() {
      const button = $("saveSettingsButton");
      if (!button) {
        return;
      }
      button.textContent = state.settingsDirty ? 'Save Settings *' : 'Save Settings';
    }

    function markSettingsDirty() {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      state.settingsDirty = true;
      updateSaveButton();
    }

    function markSettingsClean() {
      state.settingsDirty = false;
      updateSaveButton();
    }

    function getJson(path, options) {
      return fetch(path, options).then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || 'Request failed');
        }
        return payload;
      });
    }

    function applySettings(settings) {
      state.settings = settings;
      renderPublic({model: settings.chat.model, dataset: summarizeTrainingDatasets(settings.training)});

      $("chatModel").value = settings.chat.model;
      $("chatTimeout").value = settings.chat.timeout;
      $("chatNumPredict").value = settings.chat.num_predict;
      $("chatDevice").value = settings.chat.device;
      $("chatHistory").value = settings.chat.max_history_messages;
      $("chatTemperature").value = settings.chat.temperature;
      $("chatSeed").value = settings.chat.seed;
      $("chatEnableThinking").checked = settings.chat.enable_thinking;
      $("chatMemoryFile").value = settings.chat.memory_file;
      $("chatTrainingLogFile").value = settings.chat.training_log_file;
      $("chatMessagesLogFile").value = settings.chat.messages_log_file;

      $("datasetScript").value = settings.dataset.script;
      $("datasetExtraArgs").value = settings.dataset.extra_args;

      $("trainingScript").value = settings.training.script;
      $("trainingPlatform").value = settings.training.platform;
      setMultiSelectValues("trainingDatasetFile", settings.training.dataset_files);
      $("trainingMlxModel").value = settings.training.mlx_model || '';
      $("trainingBaseModelGguf").value = settings.training.base_model_gguf;
      $("trainingBaseOllamaModel").value = settings.training.base_ollama_model;
      $("trainingWindowsAcceleration").value = settings.training.windows_acceleration;
      $("trainingMacAcceleration").value = settings.training.mac_acceleration;
      $("trainingOutputDir").value = settings.training.output_dir;
      $("trainingMemoryDatasetFile").value = settings.training.memory_dataset_file;
      $("trainingMemoryOnly").checked = Boolean(settings.training.memory_only);
      $("trainingLimit").value = settings.training.limit;
      $("trainingOffset").value = settings.training.offset;
      $("trainingContextLength").value = settings.training.context_length;
      $("trainingBatchSize").value = settings.training.batch_size;
      $("trainingIters").value = settings.training.iters;
      $("trainingGradAccumulationSteps").value = settings.training.grad_accumulation_steps;
      $("trainingNumLayers").value = settings.training.num_layers;
      $("trainingStepsPerReport").value = settings.training.steps_per_report;
      $("trainingStepsPerEval").value = settings.training.steps_per_eval;
      $("trainingValBatches").value = settings.training.val_batches;
      $("trainingSaveEvery").value = settings.training.save_every;
      $("trainingUBatchSize").value = settings.training.ubatch_size;
      $("trainingEpochs").value = settings.training.epochs;
      $("trainingGpuLayers").value = settings.training.gpu_layers;
      $("trainingThreads").value = settings.training.threads;
      $("trainingSeed").value = settings.training.seed;
      $("trainingValidationSplit").value = settings.training.validation_split;
      $("trainingLlamaBinDir").value = settings.training.llama_bin_dir;
      $("trainingOllamaModelName").value = settings.training.ollama_model_name;
      $("trainingExportModelDir").value = settings.training.export_model_dir;
      $("trainingExtraArgs").value = settings.training.extra_args;
      $("trainingSkipMemoryDataset").checked = settings.training.skip_memory_dataset;
      $("trainingPrepareOnly").checked = settings.training.prepare_only;
      $("trainingCreateOllamaModel").checked = settings.training.create_ollama_model;
      $("trainingExportModelBundle").checked = settings.training.export_model_bundle;
      $("trainingDryRun").checked = settings.training.dry_run;

      $("appHost").value = settings.app.host;
      $("appPort").value = settings.app.port;
      $("appPollInterval").value = settings.app.poll_interval_ms;
      $("appAutoOpenBrowser").checked = settings.app.auto_open_browser;
      updateTrainingPlatformUI();
      markSettingsClean();
    }

    function updateCommandPreviews(commands) {
      $("datasetCommandPreview").textContent = commands && commands.dataset ? commands.dataset : 'Admin only.';
      $("trainingCommandPreview").textContent = commands && commands.training ? commands.training : 'Admin only.';
    }

    function historySignature(history) {
      const last = history.length ? history[history.length - 1] : null;
      if (!last) {
        return '0';
      }
      return history.length + ':' + last.role + ':' + String(last.content || '').length + ':' + String(last.content || '').slice(0, 96);
    }

    function taskSignature(task) {
      if (!task || !task.name) {
        return 'idle';
      }
      return [task.name, task.status, task.exit_code, (task.log || '').length].join(':');
    }

    function collectSettings() {
      const selectedDatasetFiles = getMultiSelectValues("trainingDatasetFile");
      const existingTraining = (state.settings && state.settings.training) || {};

      return {
        app: {
          host: $("appHost").value.trim(),
          port: Number($("appPort").value),
          poll_interval_ms: Number($("appPollInterval").value),
          auto_open_browser: $("appAutoOpenBrowser").checked,
        },
        chat: {
          model: $("chatModel").value.trim(),
          timeout: Number($("chatTimeout").value),
          num_predict: Number($("chatNumPredict").value),
          device: $("chatDevice").value.trim(),
          max_history_messages: Number($("chatHistory").value),
          temperature: Number($("chatTemperature").value),
          seed: Number($("chatSeed").value),
          enable_thinking: $("chatEnableThinking").checked,
          memory_file: $("chatMemoryFile").value.trim(),
          training_log_file: $("chatTrainingLogFile").value.trim(),
          messages_log_file: $("chatMessagesLogFile").value.trim(),
        },
        dataset: {
          script: $("datasetScript").value.trim(),
          extra_args: $("datasetExtraArgs").value.trim(),
        },
        training: {
          script: $("trainingScript").value.trim(),
          platform: $("trainingPlatform").value.trim(),
          dataset_files: selectedDatasetFiles,
          dataset_file: selectedDatasetFiles[0] || '',
          eval_file: String(existingTraining.eval_file || ''),
          mlx_model: $("trainingMlxModel").value.trim(),
          base_model_gguf: $("trainingBaseModelGguf").value.trim(),
          base_ollama_model: $("trainingBaseOllamaModel").value.trim(),
          windows_acceleration: $("trainingWindowsAcceleration").value.trim(),
          mac_acceleration: $("trainingMacAcceleration").value.trim(),
          output_dir: $("trainingOutputDir").value.trim(),
          memory_dataset_file: $("trainingMemoryDatasetFile").value.trim(),
          skip_memory_dataset: $("trainingSkipMemoryDataset").checked,
          memory_only: $("trainingMemoryOnly").checked,
          limit: Number($("trainingLimit").value),
          offset: Number($("trainingOffset").value),
          context_length: Number($("trainingContextLength").value),
          batch_size: Number($("trainingBatchSize").value),
          ubatch_size: Number($("trainingUBatchSize").value),
          epochs: Number($("trainingEpochs").value),
          iters: Number($("trainingIters").value),
          grad_accumulation_steps: Number($("trainingGradAccumulationSteps").value),
          num_layers: Number($("trainingNumLayers").value),
          steps_per_report: Number($("trainingStepsPerReport").value),
          steps_per_eval: Number($("trainingStepsPerEval").value),
          val_batches: Number($("trainingValBatches").value),
          save_every: Number($("trainingSaveEvery").value),
          gpu_layers: Number($("trainingGpuLayers").value),
          threads: Number($("trainingThreads").value),
          seed: Number($("trainingSeed").value),
          validation_split: Number($("trainingValidationSplit").value),
          llama_bin_dir: $("trainingLlamaBinDir").value.trim(),
          grad_checkpoint: Boolean(existingTraining.grad_checkpoint),
          mask_prompt: Boolean(existingTraining.mask_prompt),
          prepare_only: $("trainingPrepareOnly").checked,
          create_ollama_model: $("trainingCreateOllamaModel").checked,
          ollama_model_name: $("trainingOllamaModelName").value.trim(),
          export_model_bundle: $("trainingExportModelBundle").checked,
          export_model_dir: $("trainingExportModelDir").value.trim(),
          dry_run: $("trainingDryRun").checked,
          extra_args: $("trainingExtraArgs").value.trim(),
        },
      };
    }

    function renderChat(history, force = false) {
      const signature = historySignature(history);
      if (!force && signature === state.lastChatSignature) {
        return;
      }
      state.lastChatSignature = signature;
      const log = $("chatLog");
      log.innerHTML = "";
      if (!history.length) {
        const div = document.createElement("div");
        div.className = "bubble assistant";
        div.innerHTML = '<div class="bubble-meta">Assistant</div>No saved chat yet. Start talking and each turn will be logged as future training data.';
        log.appendChild(div);
        return;
      }
      history.forEach((item) => {
        const bubble = document.createElement("div");
        bubble.className = "bubble " + (item.role === "user" ? "user" : "assistant");
        const meta = item.role === "user" ? "You" : "Assistant";
        bubble.innerHTML = '<div class="bubble-meta">' + meta + '</div>' + escapeHtml(item.content);
        if (item.role === 'user' && Array.isArray(item.attachments) && item.attachments.length) {
          const fileMeta = document.createElement('div');
          fileMeta.className = 'bubble-files';
          fileMeta.textContent = summarizeHistoryAttachments(item.attachments);
          bubble.appendChild(fileMeta);
        }
        log.appendChild(bubble);
      });
      log.scrollTop = log.scrollHeight;
    }

    function updateTask(task) {
      const signature = taskSignature(task);
      if (signature === state.lastTaskSignature) {
        return;
      }
      state.lastTaskSignature = signature;
      if (!task || !task.name) {
        setStatus($("taskStatus"), "Idle", "");
        $("taskLog").textContent = "No task output yet.";
        return;
      }
      const label = task.name + ' · ' + task.status;
      const kind = task.status === 'running' ? 'running' : (task.status === 'failed' ? 'error' : '');
      setStatus($("taskStatus"), label, kind);
      $("taskLog").textContent = task.log || "No task output yet.";
    }

    function resetUserForm() {
      $("managedUsername").value = '';
      $("managedPassword").value = '';
      $("managedRole").value = 'user';
      setUserManagerStatus('');
    }

    function loadUserIntoForm(user) {
      $("managedUsername").value = user.username;
      $("managedPassword").value = '';
      $("managedRole").value = user.role;
      setUserManagerStatus('Editing ' + user.username);
    }

    function renderUsers(users) {
      const list = $("userList");
      if (!list) {
        return;
      }
      list.innerHTML = '';
      if (!users || !users.length) {
        const empty = document.createElement('div');
        empty.className = 'tiny';
        empty.textContent = 'No remote users created yet.';
        list.appendChild(empty);
        return;
      }
      users.forEach((user) => {
        const row = document.createElement('div');
        row.className = 'user-row';

        const meta = document.createElement('div');
        const name = document.createElement('strong');
        name.textContent = user.username;
        const role = document.createElement('div');
        role.className = 'user-role';
        role.textContent = user.role;
        meta.appendChild(name);
        meta.appendChild(role);

        const actions = document.createElement('div');
        actions.className = 'row tight';
        const editButton = document.createElement('button');
        editButton.className = 'ghost';
        editButton.textContent = 'Edit';
        editButton.addEventListener('click', () => loadUserIntoForm(user));
        const deleteButton = document.createElement('button');
        deleteButton.className = 'ghost';
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', () => deleteUser(user.username));
        actions.appendChild(editButton);
        actions.appendChild(deleteButton);

        row.appendChild(meta);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }

    function toggleUserManager(forceOpen) {
      const panel = $("userManagementPanel");
      const button = $("adminUsersButton");
      if (!panel || !button) {
        return;
      }
      const open = typeof forceOpen === 'boolean' ? forceOpen : panel.hidden;
      panel.hidden = !open;
      button.textContent = open ? 'Close Admin' : 'Admin';
      if (!open) {
        resetUserForm();
      }
    }

    function resetAdminViews() {
      state.settings = null;
      state.options = null;
      state.settingsLoaded = false;
      state.optionsLoaded = false;
      state.promptsLoaded = false;
      state.settingsDirty = false;
      updateSaveButton();
      updateCommandPreviews(null);
      $("promptLibrary").textContent = '';
      $("localModelLibrary").textContent = '';
      setStatus($("localModelStatus"), 'Ready', '');
      renderUsers([]);
      updateTask(null);
      toggleUserManager(false);
    }

    function renderAuth(auth, publicState) {
      state.auth = auth || {
        authenticated: false,
        username: '',
        role: 'guest',
        can_manage: false,
        is_local_admin: false,
      };
      renderPublic(publicState);
      const authenticated = Boolean(state.auth.authenticated);
      const isAdmin = Boolean(state.auth.can_manage);
      $("loginPanel").hidden = authenticated;
      $("mainGrid").hidden = !authenticated;
      $("adminPanelStack").hidden = !isAdmin;
      $("adminSettingsStack").hidden = !isAdmin;
      $("adminUsersButton").hidden = !isAdmin;
      $("logoutButton").hidden = !authenticated || Boolean(state.auth.is_local_admin);
      if (!isAdmin) {
        toggleUserManager(false);
      }

      if (!authenticated) {
        $("authSummary").textContent = 'Sign in to access your chat.';
        setStatus($("chatStatus"), 'Login required', 'error');
      } else if (state.auth.is_local_admin) {
        $("authSummary").textContent = 'Localhost admin session';
        if ($("chatStatus").textContent === 'Login required') {
          setStatus($("chatStatus"), 'Ready', '');
        }
      } else {
        $("authSummary").textContent = 'Signed in as ' + state.auth.username + ' · ' + state.auth.role;
        if ($("chatStatus").textContent === 'Login required') {
          setStatus($("chatStatus"), 'Ready', '');
        }
      }
    }

    function renderStatePayload(payload, options = {}) {
      renderAuth(payload.auth, payload.public);
      renderModelLoadStatus(payload.model_load, payload.public && payload.public.model);
      const auth = payload.auth || {};
      if (!auth.authenticated) {
        resetAdminViews();
        clearChatAttachments();
        state.lastChatSignature = '';
        renderChat([], true);
        $("chatMeta").textContent = '';
        return;
      }

      renderChat((payload.chat && payload.chat.history) || [], Boolean(options.forceChat));

      if (auth.can_manage) {
        if (payload.options && payload.settings) {
          renderLocalModelBundles(payload.options.local_model_bundles, payload.settings);
        }
        if (payload.options && payload.settings && (!state.optionsLoaded || options.hydrateOptions || !state.settingsDirty)) {
          applyOptions(payload.options, payload.settings);
          state.optionsLoaded = true;
        }
        if (payload.settings) {
          if (!state.settingsLoaded || options.hydrateSettings) {
            applySettings(payload.settings);
            state.settingsLoaded = true;
          } else {
            state.settings = payload.settings;
            if (!state.settingsDirty) {
              renderPublic(payload.public);
            }
          }
        }
        if (!state.promptsLoaded || options.reloadPrompts) {
          $("promptLibrary").textContent = payload.prompts || 'Prompt file not found.';
          state.promptsLoaded = true;
        }
        if (!state.settingsDirty) {
          updateCommandPreviews(payload.commands);
        }
        renderUsers(payload.users || []);
        updateTask(payload.task);
        updateLocalModelTaskStatus(payload.task);
      } else {
        resetAdminViews();
        renderPublic(payload.public);
      }

      const chat = payload.chat || {turns: 0, memory_file: ''};
      if (auth.can_manage) {
        $("chatMeta").textContent = 'Saved turns: ' + chat.turns + ' · Memory file: ' + chat.memory_file;
      } else {
        $("chatMeta").textContent = 'Saved turns: ' + chat.turns + ' · Your chat is logged for future training.';
      }
    }

    async function refreshState(options = {}) {
      const payload = await getJson('/api/state');
      renderStatePayload(payload, options);
    }

    async function saveSettings() {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      const settings = collectSettings();
      const currentModel = state.settings && state.settings.chat ? state.settings.chat.model : '';
      const modelChanged = currentModel !== settings.chat.model;
      setStatus($("chatStatus"), modelChanged ? 'Saving settings and loading model...' : 'Saving settings...', 'running');
      const payload = await getJson('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(settings),
      });
      renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true, reloadPrompts: true});
      clearInterval(state.pollTimer);
      state.pollTimer = setInterval(refreshState, currentPollInterval());
      if (payload.model_load && payload.model_load.status === 'loading') {
        setStatus($("chatStatus"), 'Model loading...', 'running');
      } else {
        setStatus($("chatStatus"), 'Settings saved', '');
        setTimeout(() => setStatus($("chatStatus"), 'Ready', ''), 1600);
      }
    }

    async function login() {
      const username = $("loginUsername").value.trim();
      const password = $("loginPassword").value;
      if (!username || !password) {
        setStatus($("loginStatus"), 'Enter username and password', 'error');
        return;
      }
      $("loginButton").disabled = true;
      setStatus($("loginStatus"), 'Signing in', 'running');
      try {
        const payload = await getJson('/api/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username, password}),
        });
        $("loginPassword").value = '';
        state.lastChatSignature = '';
        state.lastTaskSignature = '';
        renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true, reloadPrompts: true, forceChat: true});
        clearInterval(state.pollTimer);
        state.pollTimer = setInterval(refreshState, currentPollInterval());
      } catch (error) {
        setStatus($("loginStatus"), error.message, 'error');
      } finally {
        $("loginButton").disabled = false;
      }
    }

    async function logout() {
      const payload = await getJson('/api/logout', {method: 'POST'});
      state.lastChatSignature = '';
      state.lastTaskSignature = '';
      renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true, reloadPrompts: true, forceChat: true});
      clearInterval(state.pollTimer);
      state.pollTimer = setInterval(refreshState, currentPollInterval());
      setStatus($("loginStatus"), 'Signed out', '');
    }

    async function deleteModelBundle(bundleName) {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      if (!confirm('Are you sure you want to delete this model bundle? This will remove it from the models folder and delete it from Ollama if it was imported.')) {
        return;
      }
      setStatus($("localModelStatus"), 'Deleting bundle ' + bundleName + '...', 'running');
      try {
        const payload = await getJson('/api/models/delete', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({bundle_name: bundleName}),
        });
        renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true});
        setStatus($("localModelStatus"), 'Bundle deleted', '');
      } catch (error) {
        setStatus($("localModelStatus"), error.message, 'error');
      }
    }

    async function loadModelBundle(bundleName, modelName) {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      setStatus($("localModelStatus"), 'Loading ' + modelName + ' from the models folder...', 'running');
      try {
        const payload = await getJson('/api/models/load', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({bundle_name: bundleName}),
        });
        renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true});
        if (payload.model_load && payload.model_load.status === 'loading') {
          setStatus($("localModelStatus"), 'Loading ' + modelName + '...', 'running');
        } else {
          setStatus($("localModelStatus"), 'Using portable model ' + modelName, '');
        }
      } catch (error) {
        setStatus($("localModelStatus"), error.message, 'error');
      }
    }

    async function useImportedModel(modelName) {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      setStatus($("localModelStatus"), 'Switching chat to ' + modelName + '...', 'running');
      try {
        const payload = await getJson('/api/models/use', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({model_name: modelName}),
        });
        renderStatePayload(payload, {hydrateSettings: true, hydrateOptions: true});
        if (payload.model_load && payload.model_load.status === 'loading') {
          setStatus($("localModelStatus"), 'Loading ' + modelName + '...', 'running');
          setStatus($("chatStatus"), 'Model loading...', 'running');
        } else {
          setStatus($("localModelStatus"), 'Using ' + modelName, '');
        }
      } catch (error) {
        setStatus($("localModelStatus"), error.message, 'error');
      }
    }

    async function sendChat() {
      if (!(state.auth && state.auth.authenticated)) {
        return;
      }
      const input = $("chatInput");
      const message = input.value.trim();
      const rawAttachments = Array.isArray(state.chatAttachments) ? state.chatAttachments : [];
      const attachmentPayload = buildChatRequestAttachments(rawAttachments);
      const attachments = attachmentPayload.attachments;
      if (!message && !attachments.length) {
        if (rawAttachments.length) {
          setStatus($("chatStatus"), 'Attached files were too large to include in the live prompt. Try fewer files or smaller files.', 'error');
        }
        return;
      }
      $("sendChatButton").disabled = true;
      setStatus($("chatStatus"), 'Generating', 'running');
      try {
        const payload = await getJson('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message, attachments}),
        });
        const attachmentCount = rawAttachments.length;
        input.value = '';
        clearChatAttachments();
        renderChat(payload.history, true);
        const truncated = Boolean(payload.reply_info && payload.reply_info.truncated);
        const doneReason = payload.reply_info ? payload.reply_info.done_reason : 'unknown';
        const evalCount = payload.reply_info ? payload.reply_info.eval_count : 0;
        const numPredict = payload.reply_info ? payload.reply_info.num_predict : 0;
        const compactedPrompt = Boolean(payload.reply_info && payload.reply_info.attachment_prompt_compacted);
        const compactedPromptNote = compactedPrompt || attachmentPayload.truncatedCount || attachmentPayload.omittedCount
          ? ' · live prompt compacted'
          : '';
        const trainingNote = attachmentCount ? ' · file chat saved for future training' : '';
        if (state.auth && state.auth.can_manage) {
          $("chatMeta").textContent = 'Saved turns: ' + payload.turns + ' · Last reply: ' + payload.elapsed_seconds + 's · tokens: ' + evalCount + '/' + numPredict + ' · stop: ' + doneReason + (attachmentCount ? ' · files: ' + attachmentCount : '') + compactedPromptNote + trainingNote;
        } else {
          $("chatMeta").textContent = 'Saved turns: ' + payload.turns + ' · Your chat is logged for future training.' + (attachmentCount ? ' · files: ' + attachmentCount : '') + compactedPromptNote + trainingNote;
        }
        if (truncated) {
          setStatus($("chatStatus"), 'Reply hit num_predict limit. Raise Num Predict in Settings.', 'error');
        } else {
          setStatus($("chatStatus"), 'Reply saved', '');
        }
      } catch (error) {
        setStatus($("chatStatus"), error.message, 'error');
      } finally {
        $("sendChatButton").disabled = false;
      }
    }

    async function clearChat() {
      if (!(state.auth && state.auth.authenticated)) {
        return;
      }
      if (!confirm('Clear saved chat memory for the current user?')) {
        return;
      }
      const payload = await getJson('/api/chat/clear', {method: 'POST'});
      renderChat(payload.history, true);
      if (state.auth && state.auth.can_manage) {
        $("chatMeta").textContent = 'Saved turns: ' + payload.turns + ' · Memory file: ' + payload.memory_file;
      } else {
        $("chatMeta").textContent = 'Saved turns: 0 · Your chat is logged for future training.';
      }
      setStatus($("chatStatus"), 'Memory cleared', '');
    }

    async function triggerTask(path, button) {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      button.disabled = true;
      try {
        const payload = await getJson(path, {method: 'POST'});
        updateTask(payload.task);
      } catch (error) {
        updateTask({name: 'task', status: 'failed', log: error.message});
      } finally {
        button.disabled = false;
      }
    }

    async function saveUser() {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      const username = $("managedUsername").value.trim();
      const password = $("managedPassword").value;
      const role = $("managedRole").value || 'user';
      if (!username) {
        setUserManagerStatus('Username is required.', true);
        return;
      }
      try {
        const payload = await getJson('/api/users/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username, password, role}),
        });
        renderUsers(payload.users || []);
        setUserManagerStatus(payload.message || 'User saved.');
        $("managedPassword").value = '';
      } catch (error) {
        setUserManagerStatus(error.message, true);
      }
    }

    async function deleteUser(username) {
      if (!(state.auth && state.auth.can_manage)) {
        return;
      }
      if (!confirm('Delete user ' + username + '?')) {
        return;
      }
      try {
        const payload = await getJson('/api/users/delete', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username}),
        });
        renderUsers(payload.users || []);
        if ($("managedUsername").value.trim() === username) {
          resetUserForm();
        }
        setUserManagerStatus(payload.message || 'User deleted.');
      } catch (error) {
        setUserManagerStatus(error.message, true);
      }
    }

    function bindEvents() {
      applyFieldTooltips();
      $("saveSettingsButton").addEventListener('click', saveSettings);
      $("sendChatButton").addEventListener('click', sendChat);
      $("clearChatButton").addEventListener('click', clearChat);
      $("chatAttachFilesButton").addEventListener('click', () => $("chatFileInput").click());
      $("chatAttachFolderButton").addEventListener('click', () => $("chatFolderInput").click());
      $("chatClearFilesButton").addEventListener('click', clearChatAttachments);
      $("chatFileInput").addEventListener('change', async (event) => {
        await addChatFiles(event.target.files);
        event.target.value = '';
      });
      $("chatFolderInput").addEventListener('change', async (event) => {
        await addChatFiles(event.target.files);
        event.target.value = '';
      });
      $("rebuildDatasetButton").addEventListener('click', () => triggerTask('/api/dataset/rebuild', $("rebuildDatasetButton")));
      $("startTrainingButton").addEventListener('click', () => triggerTask('/api/training/start', $("startTrainingButton")));
      $("loginButton").addEventListener('click', login);
      $("logoutButton").addEventListener('click', logout);
      $("adminUsersButton").addEventListener('click', () => toggleUserManager());
      $("saveUserButton").addEventListener('click', saveUser);
      $("resetUserFormButton").addEventListener('click', resetUserForm);
      document.querySelectorAll('#settingsPanel input, #settingsPanel select, #settingsPanel textarea').forEach((element) => {
        element.addEventListener('input', markSettingsDirty);
        element.addEventListener('change', markSettingsDirty);
      });
      $("trainingPlatform").addEventListener('change', updateTrainingPlatformUI);
      $("chatInput").addEventListener('keydown', (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
          sendChat();
        }
      });
      ["loginUsername", "loginPassword"].forEach((id) => {
        $(id).addEventListener('keydown', (event) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            login();
          }
        });
      });
      renderChatAttachments();
      updateSaveButton();
    }

    async function bootstrap() {
      bindEvents();
      await refreshState({hydrateSettings: true, hydrateOptions: true, reloadPrompts: true, forceChat: true});
      clearInterval(state.pollTimer);
      state.pollTimer = setInterval(refreshState, currentPollInterval());
    }

    bootstrap().catch((error) => {
      updateTask({name: 'startup', status: 'failed', log: error.stack || error.message});
    });
  </script>
</body>
</html>
"""


def merge_dicts(base, override):
    for key, value in override.items():
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def is_local_request(host):
    raw_host = str(host or "").strip()
    if not raw_host:
        return False
    if raw_host.startswith("::ffff:"):
        raw_host = raw_host.split("::ffff:", 1)[1]
    try:
        return ipaddress.ip_address(raw_host).is_loopback
    except ValueError:
        return raw_host in {"localhost"}


def load_user_store():
    if not USERS_PATH.exists():
        return {"users": []}

    payload = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    users = []
    for entry in payload.get("users", []):
        username = str(entry.get("username") or "").strip()
        role = "admin" if entry.get("role") == "admin" else "user"
        salt = str(entry.get("salt") or "").strip()
        password_hash = str(entry.get("password_hash") or "").strip()
        if username and salt and password_hash:
            users.append({
                "username": username,
                "role": role,
                "salt": salt,
                "password_hash": password_hash,
            })
    return {"users": users}


def save_user_store(store):
    USERS_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_username(username):
    cleaned = str(username or "").strip()
    if not USERNAME_PATTERN.fullmatch(cleaned):
        raise ValueError("Username must be 3-64 characters using letters, numbers, dots, underscores, or dashes.")
    return cleaned


def hash_password(password, salt_hex=None):
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return salt.hex(), derived.hex()


def verify_password(password, salt_hex, password_hash):
    _, computed_hash = hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(computed_hash, password_hash)


def empty_chat_state():
    return {
        "history": [],
        "turns": 0,
        "memory_file": "",
    "reframed_memory": "",
    "has_reframed_memory": False,
    }


def normalize_chat_history_attachment(item):
  if not isinstance(item, dict):
    return None
  path = str(item.get("path") or "").strip()
  name = str(item.get("name") or "").strip() or Path(path).name or path
  if not name and not path:
    return None
  normalized = {
    "name": name,
    "path": path or name,
  }
  for key in ["type", "size", "char_count", "line_count"]:
    value = item.get(key)
    if value not in {None, ""}:
      normalized[key] = value
  return normalized


def serialize_chat_history_for_ui(history):
  serialized = []
  if not isinstance(history, list):
    return serialized
  for item in history:
    if not isinstance(item, dict):
      continue
    role = str(item.get("role") or "").strip().lower()
    if role not in {"user", "assistant"}:
      continue
    entry = {
      "role": role,
      "content": str(item.get("content") or ""),
    }
    raw_attachments = item.get("attachments") if role == "user" else []
    attachments = []
    if isinstance(raw_attachments, list):
      for attachment in raw_attachments:
        normalized = normalize_chat_history_attachment(attachment)
        if normalized:
          attachments.append(normalized)
    if attachments:
      entry["attachments"] = attachments
    serialized.append(entry)
  return serialized


def empty_task_state():
    return {
        "name": None,
        "status": "idle",
        "log": "",
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
    }


def empty_chat_queue_state():
    return {
        "running": False,
        "pending": 0,
        "active_job": None,
        "last_completed": None,
    }


def empty_model_load_state(model=""):
  return {
    "model": str(model or "").strip(),
    "status": "idle",
    "message": "No model load requested yet.",
    "done_reason": "",
    "updated_at": None,
  }


def warm_ollama_model(model, timeout=MODEL_LOAD_TIMEOUT):
  cleaned_model = str(model or "").strip()
  if not cleaned_model:
    raise ValueError("Model name cannot be empty.")

  if is_portable_bundle_model_name(cleaned_model):
    bundle_path = resolve_portable_bundle_model_path(cleaned_model)
    command = [
      resolve_repo_python_executable(),
      str(SCRIPTS_DIR / "run_portable_bundle_chat.py"),
      "--bundle-dir",
      str(bundle_path),
      "--warmup",
      "--temperature",
      "0",
      "--seed",
      "0",
      "--num-predict",
      "1",
    ]
    try:
      result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
      )
    except subprocess.TimeoutExpired as exc:
      raise RuntimeError(f"Timed out while loading portable model {cleaned_model}.") from exc
    except OSError as exc:
      raise RuntimeError(f"Could not run the portable model loader: {exc}") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
      raise RuntimeError(stderr or stdout or f"Portable model load failed with exit code {result.returncode}.")
    try:
      return json.loads(stdout) if stdout else {"done_reason": "load"}
    except json.JSONDecodeError as exc:
      raise RuntimeError(f"Portable model loader returned invalid JSON. {stdout}") from exc

  payload = json.dumps(
    {
      "model": cleaned_model,
      "prompt": "",
      "stream": False,
      "keep_alive": "15m",
      "options": {"num_predict": 0},
    }
  ).encode("utf-8")
  request = Request(
    OLLAMA_GENERATE_API_URL,
    data=payload,
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
      raise RuntimeError(f"Model load failed with HTTP {exc.code}: {detail}") from exc
    raise RuntimeError(f"Model load failed with HTTP {exc.code}.") from exc
  except URLError as exc:
    raise RuntimeError(
      "Could not reach Ollama at http://127.0.0.1:11434 while loading the model. Start Ollama and try again."
    ) from exc
  except (TimeoutError, socket.timeout) as exc:
    raise RuntimeError(f"Timed out while loading model {cleaned_model} into Ollama.") from exc


def load_settings():
  settings = deepcopy(DEFAULT_SETTINGS)
  if SETTINGS_PATH.exists():
    loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    merge_dicts(settings, loaded)
  else:
    save_settings(settings)
  return normalize_settings(settings)


def save_settings(settings):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_extra_args(raw_text):
    if not raw_text.strip():
        return []
    return shlex.split(raw_text, posix=(os.name != "nt"))


def resolve_repo_path(path_value):
  cleaned = str(path_value or "").strip()
  if not cleaned:
    return None
  return Path(cleaned) if os.path.isabs(cleaned) else REPO_ROOT / cleaned


def path_to_repo_relative_or_absolute(path):
  try:
    return str(path.relative_to(REPO_ROOT))
  except ValueError:
    return str(path)


def build_portable_bundle_model_name(bundle_dir):
  return f"{PORTABLE_BUNDLE_MODEL_PREFIX}{path_to_repo_relative_or_absolute(bundle_dir)}"


def is_portable_bundle_model_name(model_name):
  return str(model_name or "").strip().startswith(PORTABLE_BUNDLE_MODEL_PREFIX)


def resolve_portable_bundle_model_path(model_name):
  cleaned_model = str(model_name or "").strip()
  if not is_portable_bundle_model_name(cleaned_model):
    return None
  bundle_path = cleaned_model[len(PORTABLE_BUNDLE_MODEL_PREFIX):].strip()
  if not bundle_path:
    raise ValueError("Portable bundle model path is required.")
  resolved_path = resolve_repo_path(bundle_path)
  if resolved_path is None or not resolved_path.exists() or not resolved_path.is_dir():
    raise ValueError(f"Portable bundle not found: {bundle_path}")
  return resolved_path


def detect_local_model_bundle_backend(bundle_dir, metadata=None):
  metadata = metadata or {}
  trained_model_file = str(metadata.get("trained_model_file") or "").strip()
  if trained_model_file:
    trained_model_path = bundle_dir / trained_model_file
    if trained_model_path.is_dir():
      return "mlx"
    if trained_model_path.exists() and trained_model_path.suffix.lower() == ".gguf":
      return "gguf"
  if (bundle_dir / "fused_model").is_dir():
    return "mlx"
  for child in bundle_dir.iterdir():
    if child.is_file() and child.suffix.lower() == ".gguf":
      return "gguf"
  return "unknown"


def format_chat_model_label(model_name):
  cleaned_model = str(model_name or "").strip()
  if not is_portable_bundle_model_name(cleaned_model):
    return cleaned_model
  try:
    bundle_path = resolve_portable_bundle_model_path(cleaned_model)
  except ValueError:
    return cleaned_model
  return f"Portable: {bundle_path.name}"


def read_jsonl_rows(path):
  path = Path(path)
  if not path.exists():
    return []

  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      stripped = line.strip()
      if not stripped:
        continue
      try:
        row = json.loads(stripped)
      except json.JSONDecodeError:
        continue
      if isinstance(row, dict):
        rows.append(row)
  return rows


def write_jsonl_rows(path, rows):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deduplicate_training_example_rows(rows):
  unique_rows = []
  seen = set()
  for row in rows:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer:
      continue
    key = (question, answer)
    if key in seen:
      continue
    seen.add(key)
    normalized = dict(row)
    normalized["question"] = question
    normalized["answer"] = answer
    unique_rows.append(normalized)
  return unique_rows


def is_supported_training_upload(attachment):
  attachment_path = str(attachment.get("path") or attachment.get("name") or "").strip().lower()
  attachment_type = str(attachment.get("type") or "").strip().lower()
  content = str(attachment.get("content") or "")
  if "\x00" in content:
    return False

  extension = Path(attachment_path).suffix.lower()
  if extension in TRAINING_UPLOAD_ALLOWED_EXTENSIONS:
    return True
  if not extension and (attachment_type.startswith("text/") or attachment_type in TEXT_UPLOAD_MIME_TYPES):
    return True
  return False


def classify_training_upload_focus_tags(relative_path, content):
  path_text = str(relative_path or "").replace("\\", "/").lower()
  content_text = str(content or "").lower()
  tags = {"addon"}

  if "/weapons/" in path_text or "weapon_" in Path(path_text).name or "swep." in content_text:
    tags.add("swep")
  if "/entities/" in path_text or "ent." in content_text or "entity:" in content_text or "scripted_ents.register" in content_text:
    tags.add("entity")
  if "hook.add" in content_text or "gm:" in content_text or "gamemode:" in content_text or "/autorun/" in path_text:
    tags.add("hook")
  return sorted(tags)


def build_training_upload_question(relative_path, focus_tags):
  label = str(relative_path or "uploaded addon file").strip() or "uploaded addon file"
  if "swep" in focus_tags:
    return f'Write the Garry\'s Mod SWEP file "{label}".'
  if "entity" in focus_tags:
    return f'Write the Garry\'s Mod scripted entity file "{label}".'
  if "hook" in focus_tags:
    return f'Write the Garry\'s Mod addon hook file "{label}".'
  return f'Write the Garry\'s Mod addon file "{label}".'


def build_training_upload_row(attachment):
  relative_path = str(attachment.get("path") or attachment.get("name") or "attachment.txt").strip().replace("\\", "/").lstrip("/")
  content = str(attachment.get("content") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
  if not relative_path or not content:
    return None

  focus_tags = classify_training_upload_focus_tags(relative_path, content)
  question = build_training_upload_question(relative_path, focus_tags)
  score = 320
  if "swep" in focus_tags:
    score += 40
  if "entity" in focus_tags:
    score += 32
  if "hook" in focus_tags:
    score += 28

  return {
    "question": question,
    "answer": content,
    "prompt": question,
    "completion": content,
    "source": f"upload:{relative_path}",
    "dataset": "upload",
    "title": Path(relative_path).name or relative_path,
    "kind": "user-upload-example",
    "focus_tags": ",".join(focus_tags),
    "score": score,
    "page_score": score,
    "path": relative_path,
    "size": int(attachment.get("size") or len(content.encode("utf-8"))),
  }


def build_training_upload_chat_row(user_turn, attachments, assistant_text):
  cleaned_answer = str(assistant_text or "").strip()
  if not isinstance(user_turn, dict) or not cleaned_answer:
    return None

  attachment_labels = []
  focus_tags = {"addon"}
  for attachment in attachments or []:
    relative_path = str(attachment.get("path") or attachment.get("name") or "attachment.txt").strip().replace("\\", "/").lstrip("/")
    if not relative_path:
      continue
    attachment_labels.append(relative_path)
    for tag in classify_training_upload_focus_tags(relative_path, attachment.get("content") or ""):
      focus_tags.add(tag)

  if not attachment_labels:
    return None

  prompt_content = str(user_turn.get("prompt_content") or "").strip()
  if not prompt_content:
    return None

  source_seed = "\n".join([prompt_content, cleaned_answer] + attachment_labels)
  source_hash = hashlib.sha1(source_seed.encode("utf-8")).hexdigest()[:20]
  title = attachment_labels[0] if len(attachment_labels) == 1 else f"{len(attachment_labels)} attached files"
  score = 360
  if "swep" in focus_tags:
    score += 28
  if "entity" in focus_tags:
    score += 24
  if "hook" in focus_tags:
    score += 20

  return {
    "question": prompt_content,
    "answer": cleaned_answer,
    "prompt": prompt_content,
    "completion": cleaned_answer,
    "source": f"upload-chat:{source_hash}",
    "dataset": "upload-chat",
    "title": title,
    "kind": "user-upload-chat-example",
    "focus_tags": ",".join(sorted(focus_tags)),
    "score": score,
    "page_score": score,
    "attachment_paths": attachment_labels,
    "attachment_count": len(attachment_labels),
    "user_message": str(user_turn.get("display_content") or "").strip(),
  }


def upsert_uploaded_training_rows(rows):
  existing_rows = read_jsonl_rows(UPLOADED_FILES_DATASET_PATH)
  rows_by_source = {}
  ordered_sources = []

  for row in existing_rows:
    source = str(row.get("source") or "").strip()
    if not source:
      continue
    if source not in rows_by_source:
      ordered_sources.append(source)
    rows_by_source[source] = row

  for row in rows:
    source = str(row.get("source") or "").strip()
    if not source:
      continue
    if source not in rows_by_source:
      ordered_sources.append(source)
    rows_by_source[source] = row

  final_rows = [rows_by_source[source] for source in ordered_sources]
  write_jsonl_rows(UPLOADED_FILES_DATASET_PATH, final_rows)
  return len(final_rows)


def resolve_training_memory_dataset_file(training, materialize=False):
  configured_value = str(training.get("memory_dataset_file") or "").strip()
  configured_path = resolve_repo_path(configured_value) if configured_value else None
  upload_path = UPLOADED_FILES_DATASET_PATH if UPLOADED_FILES_DATASET_PATH.exists() else None

  if not upload_path:
    return configured_value
  if configured_path and configured_path.exists():
    try:
      same_path = configured_path.resolve() == upload_path.resolve()
    except FileNotFoundError:
      same_path = configured_path == upload_path
    if same_path:
      return path_to_repo_relative_or_absolute(upload_path)
    if materialize:
      merged_rows = deduplicate_training_example_rows(
        read_jsonl_rows(configured_path) + read_jsonl_rows(upload_path)
      )
      write_jsonl_rows(MERGED_MEMORY_DATASET_PATH, merged_rows)
    return path_to_repo_relative_or_absolute(MERGED_MEMORY_DATASET_PATH)
  return path_to_repo_relative_or_absolute(upload_path)


def migrate_dataset_artifact_path(path):
  normalized = str(path or "").strip().replace("\\", "/")
  if not normalized:
    return normalized
  normalized = normalized.replace("datasets/gmod_lua_unsloth_", "datasets/gmod_lua_")
  normalized = normalized.replace("datasets/gmod_reference_unsloth", "datasets/gmod_reference")
  normalized = normalized.replace("datasets/lua51_reference_unsloth", "datasets/lua51_reference")
  return normalized


def normalize_dataset_script_path(path):
  normalized = str(path or "").strip().replace("\\", "/")
  if not normalized or normalized in {
    "prepare_unsloth_training_data.py",
    "scripts/prepare_unsloth_training_data.py",
    "prepare_training_data.py",
    DATASET_SCRIPT_PATH,
  }:
    return DATASET_SCRIPT_PATH
  return normalized


def normalize_training_script_path(platform, path):
  normalized = str(path or "").strip().replace("\\", "/")
  legacy_or_default = {
    "train_mlx_gmod_macos.py",
    "scripts/train_mlx_gmod_macos.py",
    "train_llamacpp_ollama_macos.py",
    "scripts/train_llamacpp_ollama_macos.py",
    "train_llamacpp_ollama_windows.py",
    "scripts/train_llamacpp_ollama_windows.py",
  }
  if not normalized or normalized in legacy_or_default:
    return WINDOWS_TRAINING_SCRIPT_PATH if platform == "windows" else MAC_TRAINING_SCRIPT_PATH
  return normalized


def get_selected_training_dataset_files(training):
  raw_values = training.get("dataset_files")
  if isinstance(raw_values, str):
    raw_values = [raw_values]
  if not isinstance(raw_values, list):
    raw_values = []

  normalized = []
  seen = set()
  for value in raw_values:
    migrated = migrate_dataset_artifact_path(value)
    if not migrated or migrated in seen:
      continue
    normalized.append(migrated)
    seen.add(migrated)

  fallback = migrate_dataset_artifact_path(training.get("dataset_file"))
  if fallback and fallback not in seen:
    normalized.append(fallback)

  return normalized


def summarize_training_datasets(training):
  dataset_files = get_selected_training_dataset_files(training)
  if not dataset_files:
    return ""
  if len(dataset_files) == 1:
    return dataset_files[0]
  return f"{len(dataset_files)} datasets selected"


def get_prepared_training_dataset_path(training):
  dataset_files = get_selected_training_dataset_files(training)
  if len(dataset_files) <= 1:
    return dataset_files[0] if dataset_files else ""

  digest = hashlib.sha1("\n".join(dataset_files).encode("utf-8")).hexdigest()[:12]
  return f"training_runs/prepared_datasets/combined_train_{digest}.jsonl"


def materialize_training_dataset(training):
  dataset_files = get_selected_training_dataset_files(training)
  if not dataset_files:
    raise RuntimeError("Select at least one training dataset before starting training.")
  if len(dataset_files) == 1:
    dataset_path = REPO_ROOT / dataset_files[0]
    if not dataset_path.exists():
      raise RuntimeError(f"Training dataset not found: {dataset_files[0]}")
    return dataset_files[0]

  output_relative = get_prepared_training_dataset_path(training)
  output_path = REPO_ROOT / output_relative
  output_path.parent.mkdir(parents=True, exist_ok=True)

  seen_rows = set()
  total_rows = 0
  with output_path.open("w", encoding="utf-8") as output_handle:
    for relative_path in dataset_files:
      input_path = REPO_ROOT / relative_path
      if not input_path.exists():
        raise RuntimeError(f"Training dataset not found: {relative_path}")
      with input_path.open("r", encoding="utf-8") as input_handle:
        for line in input_handle:
          stripped = line.strip()
          if not stripped or stripped in seen_rows:
            continue
          seen_rows.add(stripped)
          output_handle.write(stripped + "\n")
          total_rows += 1

  if total_rows == 0:
    raise RuntimeError("The selected training datasets did not contain any usable rows.")
  return output_relative


def infer_eval_dataset_file(train_dataset_file):
  if not train_dataset_file.endswith("_train.jsonl"):
    return ""
  candidate = train_dataset_file[:-len("_train.jsonl")] + "_eval.jsonl"
  if (REPO_ROOT / candidate).exists():
    return candidate
  return ""


def list_dataset_files(kind):
  if not DATASETS_DIR.exists():
    return []

  files = []
  for path in sorted(DATASETS_DIR.glob("*.jsonl")):
    name = path.name
    if name.endswith("_messages.jsonl"):
      continue
    if kind == "eval" and not name.endswith("_eval.jsonl"):
      continue
    files.append(str(path.relative_to(REPO_ROOT)))
  return files


def list_ollama_models():
  try:
    result = subprocess.run(
      ["ollama", "list"],
      check=True,
      capture_output=True,
      text=True,
      timeout=8,
    )
  except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
    return []

  models = []
  for line in result.stdout.splitlines()[1:]:
    line = line.strip()
    if not line:
      continue
    name = line.split()[0]
    if name and name not in models:
      models.append(name)
  return models


def ollama_model_name_variants(model_name):
  cleaned = str(model_name or "").strip()
  if not cleaned:
    return []
  variants = [cleaned]
  if ":" not in cleaned:
    variants.append(f"{cleaned}:latest")
  elif cleaned.endswith(":latest"):
    variants.append(cleaned.rsplit(":", 1)[0])
  return variants


def resolve_installed_ollama_model_name(model_name, installed_models=None):
  installed = list(installed_models or list_ollama_models())
  installed_set = set(installed)
  for candidate in ollama_model_name_variants(model_name):
    if candidate in installed_set:
      return candidate
  return ""


def ollama_model_names_match(left, right):
  return bool(set(ollama_model_name_variants(left)) & set(ollama_model_name_variants(right)))


def resolve_export_model_dir_path(training):
  configured = training.get("export_model_dir") or DEFAULT_SETTINGS["training"]["export_model_dir"]
  return resolve_repo_path(configured)


def delete_local_model_bundle(settings, bundle_name):
  bundle = resolve_local_model_bundle(settings, bundle_name)
  bundle_dir = bundle["bundle_dir"]
  ollama_model_name = bundle["ollama_model_name"]

  # 1. Delete from Ollama if it exists
  if ollama_model_name:
    try:
      subprocess.run(["ollama", "rm", ollama_model_name], capture_output=True, text=True, check=False)
    except Exception:
      pass

  # 2. Delete the bundle folder from disk
  if bundle_dir.exists() and bundle_dir.is_dir():
    import shutil
    shutil.rmtree(bundle_dir)

  # 3. If this was the current chat model, reset it to a default or empty
  current_settings = STUDIO.get_settings()
  if current_settings["chat"]["model"] == bundle.get("portable_model_name") or current_settings["chat"]["model"] == ollama_model_name:
    current_settings["chat"]["model"] = ""
    STUDIO.set_settings(current_settings)


def load_local_model_bundle_payload(bundle_dir, settings, installed_models=None):
  modelfile_path = bundle_dir / "Modelfile"
  if not modelfile_path.exists():
    return None

  metadata = {}
  bundle_json_path = bundle_dir / "bundle.json"
  if bundle_json_path.exists():
    try:
      metadata = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      metadata = {}

  model_name = str(metadata.get("ollama_model_name") or bundle_dir.name).strip() or bundle_dir.name
  resolved_installed_name = resolve_installed_ollama_model_name(model_name, installed_models)
  portable_model_name = build_portable_bundle_model_name(bundle_dir)
  portable_backend = detect_local_model_bundle_backend(bundle_dir, metadata)
  return {
    "bundle_name": bundle_dir.name,
    "bundle_path": path_to_repo_relative_or_absolute(bundle_dir),
    "ollama_model_name": model_name,
    "portable_model_name": portable_model_name,
    "portable_backend": portable_backend,
    "installed_model_name": resolved_installed_name,
    "source_output_dir": str(metadata.get("source_output_dir") or "").strip(),
    "trained_model_file": str(metadata.get("trained_model_file") or "").strip(),
    "import_command": f"ollama create {model_name} -f {path_to_repo_relative_or_absolute(modelfile_path)} --experimental",
    "imported": bool(resolved_installed_name),
    "current_chat": settings["chat"]["model"] == portable_model_name,
    "current_training_base": ollama_model_names_match(settings["training"]["base_ollama_model"], model_name),
  }


def list_local_model_bundles(settings, installed_models=None):
  export_root = resolve_export_model_dir_path(settings["training"])
  if export_root is None or not export_root.exists() or not export_root.is_dir():
    return []

  installed = installed_models if installed_models is not None else list_ollama_models()
  bundles = []
  for bundle_dir in sorted(export_root.iterdir(), key=lambda path: path.name.lower()):
    if not bundle_dir.is_dir():
      continue
    payload = load_local_model_bundle_payload(bundle_dir, settings, installed_models=installed)
    if payload is not None:
      bundles.append(payload)
  return bundles


def resolve_local_model_bundle(settings, bundle_name):
  cleaned_bundle_name = str(bundle_name or "").strip()
  if not cleaned_bundle_name:
    raise ValueError("Bundle name is required.")

  export_root = resolve_export_model_dir_path(settings["training"])
  if export_root is None or not export_root.exists() or not export_root.is_dir():
    raise ValueError("The configured models folder does not exist.")

  bundle_dir = export_root / cleaned_bundle_name
  if not bundle_dir.exists() or not bundle_dir.is_dir():
    raise ValueError(f"Saved model bundle not found: {cleaned_bundle_name}")

  payload = load_local_model_bundle_payload(bundle_dir, settings, installed_models=list_ollama_models())
  if payload is None:
    raise ValueError(f"Saved model bundle is missing a Modelfile: {cleaned_bundle_name}")

  payload["bundle_dir"] = bundle_dir
  payload["modelfile_path"] = bundle_dir / "Modelfile"
  return payload


def build_model_import_command(settings, bundle_name):
  bundle = resolve_local_model_bundle(settings, bundle_name)
  return [
    "ollama",
    "create",
    bundle["ollama_model_name"],
    "-f",
    str(bundle["modelfile_path"]),
    "--experimental",
  ]


def build_studio_options(settings):
  ollama_models = list_ollama_models()
  for current_model in [settings["chat"]["model"], settings["training"]["base_ollama_model"]]:
    if current_model and not is_portable_bundle_model_name(current_model) and current_model not in ollama_models:
      ollama_models.insert(0, current_model)

  train_datasets = list_dataset_files("train")
  for dataset_file in get_selected_training_dataset_files(settings["training"]):
    if dataset_file and dataset_file not in train_datasets:
      train_datasets.insert(0, dataset_file)

  eval_datasets = list_dataset_files("eval")
  if settings["training"]["eval_file"] and settings["training"]["eval_file"] not in eval_datasets:
    eval_datasets.insert(0, settings["training"]["eval_file"])

  local_model_bundles = list_local_model_bundles(settings, installed_models=ollama_models)

  return {
    "ollama_models": ollama_models,
    "train_datasets": train_datasets,
    "eval_datasets": eval_datasets,
    "local_model_bundles": local_model_bundles,
    "training_platforms": TRAINING_PLATFORM_OPTIONS,
    "windows_acceleration_options": WINDOWS_ACCELERATION_OPTIONS,
    "mac_acceleration_options": MAC_ACCELERATION_OPTIONS,
    "chat_device_options": CHAT_DEVICE_OPTIONS,
    "user_roles": USER_ROLE_OPTIONS,
  }


def build_dataset_command(settings):
    python_executable = resolve_repo_python_executable()
    dataset = settings["dataset"]
    command = [python_executable, str(REPO_ROOT / dataset["script"])]
    command.extend(parse_extra_args(dataset.get("extra_args", "")))
    return command


def _is_real_win32_binary(path):
  """Return True if `path` looks like a real Win32 .exe (not a Microsoft Store
  shim that fails with WinError 193 when launched by subprocess.Popen)."""
  if not path or not os.path.isfile(path):
    return False
  normalized = os.path.normcase(os.path.abspath(path))
  if "windowsapps" in normalized:
    return False
  if not path.lower().endswith(".exe"):
    return False
  # Cheap PE-header check: real Win32 binaries start with the "MZ" magic.
  try:
    with open(path, "rb") as fh:
      return fh.read(2) == b"MZ"
  except OSError:
    return False


def resolve_repo_python_executable():
  """Return the path to a Python interpreter that is safe to hand to
  subprocess.Popen on the current platform.

  We have to be careful on Windows: the Microsoft Store / "pythoncore"
  Python stub reports sys.executable as a non-.exe shim path and also drops
  python.exe / python3.exe / py.exe shims into the WindowsApps directory on
  PATH. Launching any of those via subprocess.Popen fails with WinError 193
  ("%1 is not a valid Win32 application"), so we explicitly skip them and
  verify candidates are real Win32 binaries.
  """
  # 1. Prefer a repo-local virtualenv's interpreter when one exists. On
  #    Windows a venv ships a real .exe in .venv\Scripts\python.exe; the
  #    .venv\bin\python file is a *shell script* (POSIX-style launcher) that
  #    subprocess.Popen on Windows refuses to run, triggering WinError 193.
  #    So on Windows we ONLY consider the Scripts\python.exe layout, and we
  #    require the candidate to pass the Win32 binary check. On POSIX we use
  #    the bin/python layout with a normal executable bit check.
  if os.name == "nt":
    for parts in (
      [".venv", "Scripts", "python.exe"],
      [".venv", "Scripts", "python3.exe"],
    ):
      candidate = REPO_ROOT.joinpath(*parts)
      if _is_real_win32_binary(str(candidate)):
        return str(candidate)
  else:
    for parts in (
      [".venv", "bin", "python"],
      [".venv", "bin", "python3"],
    ):
      candidate = REPO_ROOT.joinpath(*parts)
      if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)

  if os.name == "nt":
    import shutil
    # 2. Prefer the official Python launcher (`py.exe`) -- always a real
    #    Win32 binary when installed by the python.org / winget installer.
    for launcher in ("py.exe", "py"):
      found = shutil.which(launcher)
      if found and _is_real_win32_binary(found):
        return found
    # 3. Otherwise look for python.exe / python3.exe on PATH, skipping the
    #    WindowsApps Store shims and verifying PE magic.
    for launcher in ("python.exe", "python3.exe", "python", "python3"):
      found = shutil.which(launcher)
      if _is_real_win32_binary(found):
        return found
    # 4. As a final Windows-specific fallback, look for the winget / official
    #    installer layout under %LocalAppData%\Programs\Python\Python3x\.
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
      programs_dir = os.path.join(local_appdata, "Programs", "Python")
      if os.path.isdir(programs_dir):
        try:
          for entry in sorted(os.listdir(programs_dir), reverse=True):
            candidate = os.path.join(programs_dir, entry, "python.exe")
            if _is_real_win32_binary(candidate):
              return candidate
        except OSError:
          pass
    raise RuntimeError(
      "Could not find a real Win32 Python interpreter to launch the training "
      "subprocess. The Microsoft Store Python stub (pythoncore) and the "
      "WindowsApps shims cannot be used because they fail with WinError 193. "
      "Run setup_windows.ps1 to install Python 3.12 and create a .venv, then "
      "launch the Studio with `.\.venv\Scripts\python.exe .\scripts\gmod_ai_studio.py`."
    )

  # POSIX (macOS/Linux) fallback: use the interpreter running the Studio.
  return sys.executable


def build_training_command(settings, materialize=False):
  python_executable = resolve_repo_python_executable()
  training = settings["training"]
  training_script = normalize_training_script_path(
    training.get("platform") or DEFAULT_SETTINGS["training"]["platform"],
    training.get("script") or DEFAULT_SETTINGS["training"]["script"],
  )
  dataset_file = materialize_training_dataset(training) if materialize else get_prepared_training_dataset_path(training)
  if not dataset_file:
    raise RuntimeError("Select at least one training dataset before starting training.")
  memory_dataset_file = resolve_training_memory_dataset_file(training, materialize=materialize)

  if training.get("platform") == "mac":
    command = [
      python_executable,
      str(REPO_ROOT / training_script),
      "--dataset-file",
      dataset_file,
      "--eval-file",
      training["eval_file"],
      "--mlx-model",
      training.get("mlx_model") or DEFAULT_MAC_MLX_MODEL,
      "--base-ollama-model",
      training["base_ollama_model"],
      "--output-dir",
      training["output_dir"],
      "--memory-dataset-file",
      memory_dataset_file,
      "--limit",
      str(training["limit"]),
      "--offset",
      str(training["offset"]),
      "--context-length",
      str(training["context_length"]),
      "--batch-size",
      str(training["batch_size"]),
      "--iters",
      str(training["iters"]),
      "--grad-accumulation-steps",
      str(training["grad_accumulation_steps"]),
      "--num-layers",
      str(training["num_layers"]),
      "--steps-per-report",
      str(training["steps_per_report"]),
      "--steps-per-eval",
      str(training["steps_per_eval"]),
      "--val-batches",
      str(training["val_batches"]),
      "--save-every",
      str(training["save_every"]),
      "--seed",
      str(training["seed"]),
      "--validation-split",
      str(training["validation_split"]),
      "--ollama-model-name",
      training["ollama_model_name"],
    ]
    if training.get("skip_memory_dataset"):
      command.append("--skip-memory-dataset")
    if training.get("prepare_only"):
      command.append("--prepare-only")
    if not training.get("create_ollama_model", True):
      command.append("--skip-ollama-create")
    if training.get("export_model_bundle"):
      export_model_dir = training.get("export_model_dir") or DEFAULT_SETTINGS["training"]["export_model_dir"]
      if export_model_dir and not os.path.isabs(export_model_dir):
        export_model_dir = str(REPO_ROOT / export_model_dir)
      command.extend(["--export-model-dir", export_model_dir])
    if training.get("grad_checkpoint"):
      command.append("--grad-checkpoint")
    else:
      command.append("--no-grad-checkpoint")
    if training.get("mask_prompt"):
      command.append("--mask-prompt")
    else:
      command.append("--no-mask-prompt")
    if training.get("dry_run"):
      command.append("--dry-run")
    mac_device = str(training.get("mac_acceleration") or "auto").strip().lower()
    if training.get("platform") == "mac" and mac_device and mac_device != "auto":
      command.extend(["--device", mac_device])
    command.extend(parse_extra_args(training.get("extra_args", "")))
    return command

  gpu_layers = training["gpu_layers"]
  if training.get("platform") == "windows" and str(training.get("windows_acceleration") or "auto").lower() == "cpu":
    gpu_layers = 0

  command = [
    python_executable,
    str(REPO_ROOT / training_script),
    "--dataset-file",
    dataset_file,
    "--base-ollama-model",
    training["base_ollama_model"],
    "--output-dir",
    training["output_dir"],
    "--memory-dataset-file",
    memory_dataset_file,
    "--limit",
    str(training["limit"]),
    "--offset",
    str(training["offset"]),
    "--context-length",
    str(training["context_length"]),
    "--batch-size",
    str(training["batch_size"]),
    "--ubatch-size",
    str(training["ubatch_size"]),
    "--epochs",
    str(training["epochs"]),
    "--gpu-layers",
    str(gpu_layers),
    "--threads",
    str(training["threads"]),
    "--seed",
    str(training["seed"]),
    "--validation-split",
    str(training["validation_split"]),
    "--ollama-model-name",
    training["ollama_model_name"],
  ]
  if training.get("base_model_gguf"):
    command.extend(["--base-model-gguf", training["base_model_gguf"]])
  if training.get("llama_bin_dir"):
    command.extend(["--llama-bin-dir", training["llama_bin_dir"]])
  if training.get("skip_memory_dataset"):
    command.append("--skip-memory-dataset")
  if training.get("prepare_only"):
    command.append("--prepare-only")
  if not training.get("create_ollama_model", True):
    command.append("--skip-ollama-create")
  if training.get("export_model_bundle"):
    export_model_dir = training.get("export_model_dir") or DEFAULT_SETTINGS["training"]["export_model_dir"]
    if export_model_dir and not os.path.isabs(export_model_dir):
      export_model_dir = str(REPO_ROOT / export_model_dir)
    command.extend(["--export-model-dir", export_model_dir])
  if training.get("dry_run"):
    command.append("--dry-run")
  command.extend(parse_extra_args(training.get("extra_args", "")))
  return command
def command_to_string(command):
    return " ".join(shlex.quote(part) for part in command)


def normalize_mac_training_settings(training):
  if not str(training.get("mlx_model") or "").strip():
    training["mlx_model"] = DEFAULT_MAC_MLX_MODEL
  training["context_length"] = max(128, min(training["context_length"], MLX_SAFE_MAX_SEQ_LENGTH))
  safe_batch_size = (
    MLX_SAFE_MAX_BATCH_SIZE_SHORT_CONTEXT
    if training["context_length"] <= 512
    else MLX_SAFE_MAX_BATCH_SIZE_LONG_CONTEXT
  )
  training["batch_size"] = max(1, min(training["batch_size"], safe_batch_size))
  training["val_batches"] = max(1, min(training["val_batches"], MLX_SAFE_MAX_VAL_BATCHES))
  if training["context_length"] > 512:
    training["grad_checkpoint"] = True


def normalize_settings(raw_settings):
    settings = deepcopy(DEFAULT_SETTINGS)
    merge_dicts(settings, raw_settings)
    if settings["training"].get("platform") not in {"mac", "windows"}:
        settings["training"]["platform"] = DEFAULT_SETTINGS["training"]["platform"]
    settings["dataset"]["script"] = normalize_dataset_script_path(settings["dataset"].get("script"))
    settings["training"]["script"] = normalize_training_script_path(
        settings["training"]["platform"],
        settings["training"].get("script"),
    )
    settings["training"]["dataset_file"] = migrate_dataset_artifact_path(settings["training"].get("dataset_file"))
    settings["training"]["eval_file"] = migrate_dataset_artifact_path(settings["training"].get("eval_file"))
    settings["training"]["dataset_files"] = get_selected_training_dataset_files(settings["training"])
    settings["training"]["dataset_file"] = settings["training"]["dataset_files"][0] if settings["training"]["dataset_files"] else ""
    acceleration = str(settings["training"].get("windows_acceleration") or "auto").strip().lower()
    if acceleration not in {"auto", "cpu", "cuda"}:
        acceleration = DEFAULT_SETTINGS["training"]["windows_acceleration"]
    settings["training"]["windows_acceleration"] = acceleration
    mac_acceleration = str(settings["training"].get("mac_acceleration") or "auto").strip().lower()
    if mac_acceleration not in {"auto", "cpu", "metal"}:
        mac_acceleration = DEFAULT_SETTINGS["training"]["mac_acceleration"]
    settings["training"]["mac_acceleration"] = mac_acceleration
    chat_device = str(settings["chat"].get("device") or "auto").strip().lower()
    if chat_device not in {"auto", "cpu", "metal", "cuda"}:
        chat_device = DEFAULT_SETTINGS["chat"]["device"]
    settings["chat"]["device"] = chat_device
    settings["app"]["port"] = int(settings["app"]["port"])
    settings["app"]["poll_interval_ms"] = int(settings["app"]["poll_interval_ms"])
    settings["chat"]["max_history_messages"] = int(settings["chat"]["max_history_messages"])
    settings["chat"]["seed"] = int(settings["chat"]["seed"])
    settings["chat"]["num_predict"] = int(settings["chat"]["num_predict"])
    settings["chat"]["timeout"] = float(settings["chat"]["timeout"])
    settings["chat"]["temperature"] = float(settings["chat"]["temperature"])
    for key in [
        "limit", "offset", "context_length", "batch_size", "ubatch_size", "epochs",
        "iters", "grad_accumulation_steps", "num_layers", "steps_per_report",
        "steps_per_eval", "val_batches", "save_every", "gpu_layers", "threads", "seed",
    ]:
        settings["training"][key] = int(settings["training"][key])
    settings["training"]["validation_split"] = float(settings["training"]["validation_split"])
    settings["training"]["export_model_bundle"] = bool(settings["training"].get("export_model_bundle"))
    settings["training"]["export_model_dir"] = str(
      settings["training"].get("export_model_dir") or DEFAULT_SETTINGS["training"]["export_model_dir"]
    ).strip()
    if settings["training"].get("platform") == "mac":
      normalize_mac_training_settings(settings["training"])
    if not settings["training"].get("eval_file"):
        settings["training"]["eval_file"] = infer_eval_dataset_file(settings["training"]["dataset_file"])
    return settings


class TaskRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self.current = {
            "name": None,
            "status": "idle",
            "log": "",
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
        }

    def snapshot(self):
        with self._lock:
            return deepcopy(self.current)

    def start(self, name, command):
        with self._lock:
            if self.current["status"] == "running":
                raise RuntimeError("A task is already running. Wait for it to finish before starting another one.")
            self.current = {
                "name": name,
                "status": "running",
                "log": "$ " + command_to_string(command) + "\n",
                "started_at": time.time(),
                "finished_at": None,
                "exit_code": None,
            }
        thread = threading.Thread(target=self._run, args=(command,), daemon=True)
        thread.start()
        return self.snapshot()

    def _append_log(self, text):
        with self._lock:
            self.current["log"] += text

    def _finish(self, status, exit_code=None):
        with self._lock:
            self.current["status"] = status
            self.current["finished_at"] = time.time()
            self.current["exit_code"] = exit_code

    def _run(self, command):
        try:
            executable = command[0] if command else ""
            if os.name == "nt" and executable and not os.path.isfile(executable):
                self._append_log(
                    f"\nCannot start training: the resolved Python executable "
                    f"does not exist on disk:\n  {executable}\n"
                    f"Re-run setup_windows.ps1, then launch the Studio from "
                    f".venv\\Scripts\\python.exe.\n"
                )
                self._finish("failed", exit_code=-1)
                return
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(line)
            exit_code = process.wait()
            if exit_code == 0:
                self._finish("completed", exit_code=exit_code)
            else:
                self._finish("failed", exit_code=exit_code)
        except OSError as exc:
            # WinError 193 here means the resolved "executable" is not a real
            # Win32 binary (most often the Microsoft Store Python shim).
            extra = ""
            if os.name == "nt" and exc.winerror == 193 and command:
                extra = (
                    f"\nThe command's first argument is not a valid Win32 "
                    f"executable:\n  {command[0]!r}\n"
                    f"This is almost always the Microsoft Store Python stub. "
                    f"Re-run setup_windows.ps1 to install Python 3.12 and "
                    f"create a .venv, then launch the Studio with "
                    f"`.\\.venv\\Scripts\\python.exe .\\scripts\\gmod_ai_studio.py`.\n"
                )
            self._append_log("\n" + traceback.format_exc() + extra)
            self._finish("failed", exit_code=-1)
        except Exception:
            self._append_log("\n" + traceback.format_exc())
            self._finish("failed", exit_code=-1)


class ChatQueueRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._job_counter = 0
        self.current = empty_chat_queue_state()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def snapshot(self):
        with self._lock:
            snapshot = deepcopy(self.current)
            snapshot["pending"] = self._queue.qsize()
            return snapshot

    def submit(self, username, message, attachment_count, callback):
        event = threading.Event()
        normalized_username = str(username or "anonymous").strip() or "anonymous"
        preview = " ".join(str(message or "").split())[:120]
        enqueued_at = time.time()

        with self._lock:
            self._job_counter += 1
            job_id = self._job_counter
            ahead_at_enqueue = self._queue.qsize() + (1 if self.current["running"] else 0)

        job = {
            "id": job_id,
            "username": normalized_username,
            "preview": preview,
            "attachment_count": int(attachment_count or 0),
            "callback": callback,
            "event": event,
            "result": None,
            "error": None,
            "enqueued_at": enqueued_at,
            "started_at": None,
            "finished_at": None,
            "ahead_at_enqueue": ahead_at_enqueue,
        }
        self._queue.put(job)
        event.wait()
        if job["error"] is not None:
            raise job["error"]

        wait_seconds = 0.0
        if job["started_at"] is not None:
            wait_seconds = max(0.0, job["started_at"] - job["enqueued_at"])
        run_seconds = 0.0
        if job["started_at"] is not None and job["finished_at"] is not None:
            run_seconds = max(0.0, job["finished_at"] - job["started_at"])

        return job["result"], {
            "job_id": job_id,
            "ahead_at_enqueue": ahead_at_enqueue,
            "wait_seconds": round(wait_seconds, 3),
            "run_seconds": round(run_seconds, 3),
        }

    def _run(self):
        while True:
            job = self._queue.get()
            started_at = time.time()
            job["started_at"] = started_at

            with self._lock:
                self.current = {
                    "running": True,
                    "pending": self._queue.qsize(),
                    "active_job": {
                        "id": job["id"],
                        "username": job["username"],
                        "preview": job["preview"],
                        "attachment_count": job["attachment_count"],
                        "enqueued_at": job["enqueued_at"],
                        "started_at": started_at,
                        "ahead_at_enqueue": job["ahead_at_enqueue"],
                    },
                    "last_completed": self.current.get("last_completed"),
                }

            try:
                job["result"] = job["callback"]()
            except Exception as exc:
                job["error"] = exc
            finally:
                finished_at = time.time()
                job["finished_at"] = finished_at
                with self._lock:
                    self.current = {
                        "running": False,
                        "pending": self._queue.qsize(),
                        "active_job": None,
                        "last_completed": {
                            "id": job["id"],
                            "username": job["username"],
                            "preview": job["preview"],
                            "attachment_count": job["attachment_count"],
                            "finished_at": finished_at,
                            "duration_seconds": round(max(0.0, finished_at - started_at), 3),
                            "errored": job["error"] is not None,
                        },
                    }
                job["event"].set()
                self._queue.task_done()


class ModelLoadRunner:
  def __init__(self):
    self._lock = threading.Lock()
    self._request_id = 0
    self.current = empty_model_load_state()

  def snapshot(self):
    with self._lock:
      return deepcopy(self.current)

  def get_status(self):
    snapshot = self.snapshot()
    return snapshot["status"], snapshot["message"]

  def ensure_loaded(self, model, force=False):
    cleaned_model = str(model or "").strip()
    if not cleaned_model:
      with self._lock:
        self.current = empty_model_load_state()
      return self.snapshot()

    with self._lock:
      if (
        not force
        and self.current["model"] == cleaned_model
        and self.current["status"] in {"loading", "loaded"}
      ):
        return deepcopy(self.current)

      self._request_id += 1
      request_id = self._request_id
      load_target = "models folder" if is_portable_bundle_model_name(cleaned_model) else "Ollama"
      self.current = {
        "model": cleaned_model,
        "status": "loading",
        "message": f"Loading {cleaned_model} from {load_target}...",
        "done_reason": "",
        "updated_at": time.time(),
      }

    thread = threading.Thread(target=self._run, args=(cleaned_model, request_id), daemon=True)
    thread.start()
    return self.snapshot()

  def _finish(self, model, request_id, status, message, done_reason=""):
    with self._lock:
      if request_id != self._request_id:
        return

      self.current = {
        "model": model,
        "status": status,
        "message": message,
        "done_reason": done_reason,
        "updated_at": time.time(),
      }

  def _run(self, model, request_id):
    started_at = time.time()
    try:
      result = warm_ollama_model(model)
      elapsed_seconds = round(time.time() - started_at, 2)
      done_reason = str(result.get("done_reason") or "load")
      self._finish(
        model,
        request_id,
        "loaded",
        f"Loaded {model} in {elapsed_seconds}s",
        done_reason=done_reason,
      )
    except Exception as exc:
      self._finish(model, request_id, "error", str(exc))


class StudioState:
  def __init__(self):
    self._lock = threading.Lock()
    self._upload_store_lock = threading.Lock()
    self.settings = load_settings()
    self.user_store = load_user_store()
    self.sessions = {}
    self.tasks = TaskRunner()
    self.chat_queue = ChatQueueRunner()
    self.model_loader = ModelLoadRunner()
    self.model_loader.ensure_loaded(self.settings["chat"]["model"], force=True)

  def get_settings(self):
    with self._lock:
      return deepcopy(self.settings)

  def _guest_auth(self):
    return {
      "authenticated": False,
      "username": "",
      "role": "guest",
      "can_manage": False,
      "is_local_admin": False,
    }

  def _auth_payload(self, username, role, is_local_admin=False):
    normalized_role = "admin" if role == "admin" else "user"
    return {
      "authenticated": True,
      "username": username,
      "role": normalized_role,
      "can_manage": normalized_role == "admin",
      "is_local_admin": bool(is_local_admin),
    }

  def _find_user_locked(self, username):
    for user in self.user_store["users"]:
      if user["username"] == username:
        return user
    return None

  def resolve_auth(self, client_host, session_token=None):
    if is_local_request(client_host):
      return self._auth_payload("localhost", "admin", is_local_admin=True)
    if not session_token:
      return self._guest_auth()
    with self._lock:
      session = self.sessions.get(session_token)
      if not session:
        return self._guest_auth()
      user = self._find_user_locked(session["username"])
      if not user:
        self.sessions.pop(session_token, None)
        return self._guest_auth()
      return self._auth_payload(user["username"], user["role"])

  def login_user(self, username, password):
    cleaned_username = validate_username(username)
    with self._lock:
      user = self._find_user_locked(cleaned_username)
      if not user or not verify_password(password, user["salt"], user["password_hash"]):
        raise PermissionError("Invalid username or password.")
      token = secrets.token_urlsafe(32)
      self.sessions[token] = {
        "username": cleaned_username,
        "created_at": time.time(),
      }
      auth = self._auth_payload(user["username"], user["role"])
    return token, auth

  def logout_session(self, session_token):
    if not session_token:
      return
    with self._lock:
      self.sessions.pop(session_token, None)

  def list_users(self):
    with self._lock:
      users = [
        {"username": user["username"], "role": user["role"]}
        for user in sorted(self.user_store["users"], key=lambda item: item["username"].lower())
      ]
    return users

  def upsert_user(self, username, password, role):
    cleaned_username = validate_username(username)
    normalized_role = "admin" if role == "admin" else "user"
    raw_password = str(password or "")
    with self._lock:
      user = self._find_user_locked(cleaned_username)
      if user is None:
        if not raw_password:
          raise ValueError("Password is required when creating a new user.")
        salt, password_hash = hash_password(raw_password)
        self.user_store["users"].append({
          "username": cleaned_username,
          "role": normalized_role,
          "salt": salt,
          "password_hash": password_hash,
        })
      else:
        user["role"] = normalized_role
        if raw_password:
          salt, password_hash = hash_password(raw_password)
          user["salt"] = salt
          user["password_hash"] = password_hash
      self.user_store["users"] = sorted(
        self.user_store["users"],
        key=lambda item: item["username"].lower(),
      )
      save_user_store(self.user_store)
    return self.list_users()

  def delete_user(self, username):
    cleaned_username = validate_username(username)
    with self._lock:
      remaining_users = [user for user in self.user_store["users"] if user["username"] != cleaned_username]
      if len(remaining_users) == len(self.user_store["users"]):
        raise ValueError("User not found.")
      self.user_store["users"] = remaining_users
      expired_tokens = [
        token for token, session in self.sessions.items()
        if session.get("username") == cleaned_username
      ]
      for token in expired_tokens:
        self.sessions.pop(token, None)
      save_user_store(self.user_store)
    return self.list_users()

  def _resolve_chat_memory(self, auth, settings):
    if not auth.get("authenticated"):
      return None, ""
    if auth.get("is_local_admin"):
      relative_path = settings["chat"]["memory_file"]
      return REPO_ROOT / relative_path, relative_path
    username = validate_username(auth["username"])
    memory_path = CHAT_USERS_DIR / f"{username}.json"
    return memory_path, str(memory_path.relative_to(REPO_ROOT))

  def set_settings(self, settings):
    normalized = normalize_settings(settings)
    previous_model = ""
    with self._lock:
      previous_model = self.settings["chat"]["model"]
      self.settings = normalized
      save_settings(normalized)
    if normalized["chat"]["model"] != previous_model:
      self.model_loader.ensure_loaded(normalized["chat"]["model"], force=True)
    return self.get_settings()

  def use_chat_model(self, model_name):
    cleaned_model = str(model_name or "").strip()
    if not cleaned_model:
      raise ValueError("Model name is required.")

    settings = self.get_settings()
    if is_portable_bundle_model_name(cleaned_model):
      resolve_portable_bundle_model_path(cleaned_model)
      settings["chat"]["model"] = cleaned_model
    else:
      installed_models = list_ollama_models()
      resolved_model = resolve_installed_ollama_model_name(cleaned_model, installed_models)
      settings["chat"]["model"] = resolved_model or cleaned_model
    return self.set_settings(settings)

  def start_model_import(self, bundle_name):
    settings = self.get_settings()
    bundle = resolve_local_model_bundle(settings, bundle_name)
    return self.use_chat_model(bundle["portable_model_name"])

  def chat_state(self, auth):
    if not auth.get("authenticated"):
      return empty_chat_state()
    settings = self.get_settings()
    memory_path, memory_file = self._resolve_chat_memory(auth, settings)
    memory_payload = load_memory_payload(memory_path)
    history = memory_payload["messages"]
    reframed_memory = str(memory_payload.get("reframed_memory") or "")
    return {
      "history": serialize_chat_history_for_ui(history),
      "turns": len(history) // 2,
      "memory_file": memory_file,
      "reframed_memory": reframed_memory,
      "has_reframed_memory": bool(reframed_memory),
    }

  def clear_chat(self, auth):
    if not auth.get("authenticated"):
      raise PermissionError("Login required.")
    settings = self.get_settings()
    memory_path, _ = self._resolve_chat_memory(auth, settings)
    save_memory(memory_path, settings["chat"]["model"], [])
    return self.chat_state(auth)

  def store_uploads(self, auth, attachments):
    if not auth.get("authenticated"):
      raise PermissionError("Login required.")

    normalized_attachments = normalize_attachments(attachments)
    rows = []
    skipped = 0
    for attachment in normalized_attachments:
      if not is_supported_training_upload(attachment):
        skipped += 1
        continue
      row = build_training_upload_row(attachment)
      if not row:
        skipped += 1
        continue
      rows.append(row)

    with self._upload_store_lock:
      stored_total = len(read_jsonl_rows(UPLOADED_FILES_DATASET_PATH))
      if rows:
        stored_total = upsert_uploaded_training_rows(rows)

    return {
      "stored_count": len(rows),
      "skipped_count": skipped,
      "total_rows": stored_total,
      "dataset_file": path_to_repo_relative_or_absolute(UPLOADED_FILES_DATASET_PATH),
    }

  def _run_chat_turn_now(self, auth, message, attachments=None):
    settings = self.get_settings()
    chat = settings["chat"]
    memory_path, _ = self._resolve_chat_memory(auth, settings)
    history = load_memory(memory_path)
    args = SimpleNamespace(
      model=chat["model"],
      memory_file=str(memory_path),
      training_log_file=str(REPO_ROOT / chat["training_log_file"]),
      messages_log_file=str(REPO_ROOT / chat["messages_log_file"]),
      max_history_messages=chat["max_history_messages"],
      temperature=chat["temperature"],
      seed=chat["seed"],
      timeout=chat["timeout"],
      num_predict=chat["num_predict"],
      device=chat.get("device", "auto"),
      enable_thinking=chat["enable_thinking"],
    )
    assistant_text, elapsed_seconds, reply_info = run_turn(args, history, message, attachments=attachments)
    supported_attachments = [
      attachment
      for attachment in normalize_attachments(attachments)
      if is_supported_training_upload(attachment)
    ]
    if supported_attachments:
      upload_user_turn = build_user_turn(message, supported_attachments)
      upload_chat_row = build_training_upload_chat_row(upload_user_turn, supported_attachments, assistant_text)
      if upload_chat_row:
        with self._upload_store_lock:
          upsert_uploaded_training_rows([upload_chat_row])
    state = self.chat_state(auth)
    state["assistant_text"] = assistant_text
    state["elapsed_seconds"] = elapsed_seconds
    state["reply_info"] = reply_info
    return state

  def run_chat_turn(self, auth, message, attachments=None):
    if not auth.get("authenticated"):
      raise PermissionError("Login required.")

    normalized_attachments = normalize_attachments(attachments)
    username = "localhost" if auth.get("is_local_admin") else str(auth.get("username") or "anonymous")
    state, queue_info = self.chat_queue.submit(
      username=username,
      message=message,
      attachment_count=len(normalized_attachments),
      callback=lambda: self._run_chat_turn_now(auth, message, attachments=normalized_attachments),
    )
    state["queue"] = queue_info
    return state


STUDIO = StudioState()


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "GModAIStudio/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/gmodai.html"}:
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(self._build_state_payload())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/login":
                try:
                    token, auth = STUDIO.login_user(payload.get("username"), payload.get("password"))
                except PermissionError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.UNAUTHORIZED)
                    return
                self._send_json(
                    self._build_state_payload(auth=auth),
                    headers={"Set-Cookie": self._build_session_cookie(token)},
                )
                return

            if parsed.path == "/api/logout":
                STUDIO.logout_session(self._get_session_token())
                self._send_json(
                    self._build_state_payload(auth=STUDIO.resolve_auth(self.client_address[0], None)),
                    headers={"Set-Cookie": self._clear_session_cookie()},
                )
                return

            auth = self._auth_context()
            if parsed.path == "/api/settings":
                self._require_admin(auth)
                settings = STUDIO.set_settings(payload)
                self._send_json(self._build_state_payload(auth=auth, settings=settings))
                return
            if parsed.path == "/api/chat":
                message = str(payload.get("message") or "").strip()
                attachments = payload.get("attachments")
                if not message and not attachments:
                    self._send_json({"error": "Message cannot be empty."}, status=HTTPStatus.BAD_REQUEST)
                    return
                state = STUDIO.run_chat_turn(auth, message, attachments=attachments)
                self._send_json(state)
                return
            if parsed.path == "/api/uploads/store":
              attachments = payload.get("attachments")
              self._send_json(STUDIO.store_uploads(auth, attachments))
              return
            if parsed.path == "/api/chat/clear":
                self._send_json(STUDIO.clear_chat(auth))
                return
            if parsed.path == "/api/dataset/rebuild":
                self._require_admin(auth)
                task = STUDIO.tasks.start("dataset-rebuild", build_dataset_command(STUDIO.get_settings()))
                self._send_json({"task": task})
                return
            if parsed.path == "/api/training/start":
                self._require_admin(auth)
                task = STUDIO.tasks.start("training", build_training_command(STUDIO.get_settings(), materialize=True))
                self._send_json({"task": task})
                return
            if parsed.path == "/api/models/load":
              self._require_admin(auth)
              STUDIO.start_model_import(payload.get("bundle_name"))
              self._send_json(self._build_state_payload(auth=auth))
              return
            if parsed.path == "/api/models/delete":
              self._require_admin(auth)
              try:
                delete_local_model_bundle(STUDIO.get_settings(), payload.get("bundle_name"))
                self._send_json(self._build_state_payload(auth=auth))
              except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
              return
            if parsed.path == "/api/models/use":
              self._require_admin(auth)
              settings = STUDIO.use_chat_model(payload.get("model_name"))
              self._send_json(self._build_state_payload(auth=auth, settings=settings))
              return
            if parsed.path == "/api/users/save":
                self._require_admin(auth)
                users = STUDIO.upsert_user(
                    payload.get("username"),
                    payload.get("password"),
                    payload.get("role"),
                )
                self._send_json({"users": users, "message": "User saved."})
                return
            if parsed.path == "/api/users/delete":
                self._require_admin(auth)
                users = STUDIO.delete_user(payload.get("username"))
                self._send_json({"users": users, "message": "User deleted."})
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format, *args):
        return

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_html(self, body):
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload, status=HTTPStatus.OK, headers=None):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _get_session_token(self):
        cookie_header = self.headers.get("Cookie") or ""
        if not cookie_header:
            return ""
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        morsel = cookie.get(SESSION_COOKIE_NAME)
        return morsel.value if morsel else ""

    def _build_session_cookie(self, session_token):
        cookie = SimpleCookie()
        cookie[SESSION_COOKIE_NAME] = session_token
        morsel = cookie[SESSION_COOKIE_NAME]
        morsel["path"] = "/"
        morsel["httponly"] = True
        morsel["samesite"] = "Lax"
        return morsel.OutputString()

    def _clear_session_cookie(self):
        cookie = SimpleCookie()
        cookie[SESSION_COOKIE_NAME] = ""
        morsel = cookie[SESSION_COOKIE_NAME]
        morsel["path"] = "/"
        morsel["httponly"] = True
        morsel["samesite"] = "Lax"
        morsel["expires"] = "Thu, 01 Jan 1970 00:00:00 GMT"
        morsel["max-age"] = 0
        return morsel.OutputString()

    def _auth_context(self):
        return STUDIO.resolve_auth(self.client_address[0], self._get_session_token())

    def _require_admin(self, auth):
        if not auth.get("can_manage"):
            raise PermissionError("Admin access required.")

    def _build_commands(self, settings):
        return {
            "dataset": command_to_string(build_dataset_command(settings)),
            "training": command_to_string(build_training_command(settings)),
        }

    def _build_options(self, settings):
        return build_studio_options(settings)

    def _build_state_payload(self, auth=None, settings=None):
        resolved_auth = auth or self._auth_context()
        current_settings = settings or STUDIO.get_settings()
        payload = {
            "auth": resolved_auth,
            "public": {
            "model": format_chat_model_label(current_settings["chat"]["model"]),
            "dataset": summarize_training_datasets(current_settings["training"]),
            },
          "model_load": STUDIO.model_loader.snapshot(),
            "chat": STUDIO.chat_state(resolved_auth),
            "task": STUDIO.tasks.snapshot() if resolved_auth.get("can_manage") else empty_task_state(),
            "users": STUDIO.list_users() if resolved_auth.get("can_manage") else [],
            "prompts": (
                PROMPTS_PATH.read_text(encoding="utf-8")
                if resolved_auth.get("can_manage") and PROMPTS_PATH.exists()
                else ""
            ),
        }
        if resolved_auth.get("can_manage"):
            payload["settings"] = current_settings
            payload["options"] = self._build_options(current_settings)
            payload["commands"] = self._build_commands(current_settings)
        return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local GMod AI Studio web panel.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = STUDIO.get_settings()
    host = args.host or settings["app"]["host"]
    port = args.port or int(settings["app"]["port"])
    httpd = ThreadingHTTPServer((host, port), StudioHandler)
    url = f"http://{host}:{port}"
    print(f"GMod AI Studio running at {url}")
    print(f"Settings file: {SETTINGS_PATH}")
    if settings["app"].get("auto_open_browser", True) and not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GMod AI Studio.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()