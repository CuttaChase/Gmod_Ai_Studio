$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RecommendedChatModel = 'qwen2.5-coder:1.5b'
$RecommendedLlamaCppModelFile = 'Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf'
$RecommendedLlamaCppModelUrl = 'https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf?download=true'
$RecommendedLlamaCppModelDir = Join-Path $RepoRoot 'models\base'
$RecommendedLlamaCppModelPath = Join-Path $RecommendedLlamaCppModelDir $RecommendedLlamaCppModelFile
Set-Location $RepoRoot

function Ensure-WingetPackage {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string]$Override = ''
    )

    $args = @(
        'install',
        '--id', $Id,
        '-e',
        '--accept-package-agreements',
        '--accept-source-agreements'
    )

    if ($Override) {
        $args += @('--override', $Override)
    }

    & winget @args | Out-Host
}

function Resolve-ToolPath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Fallbacks = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    foreach ($candidate in $Fallbacks) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Download-FileIfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $Parent = Split-Path -Parent $Destination
    if ($Parent -and -not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    if (Test-Path $Destination) {
        return
    }

    Write-Host "Downloading GGUF model: $(Split-Path -Leaf $Destination)"
    Invoke-WebRequest -Uri $Uri -OutFile $Destination
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget is required for setup_windows.ps1. Install App Installer from Microsoft Store, then rerun this script.'
}

Ensure-WingetPackage -Id 'Python.Python.3.12'
Ensure-WingetPackage -Id 'Ollama.Ollama'
Ensure-WingetPackage -Id 'Git.Git'
Ensure-WingetPackage -Id 'Kitware.CMake'
Ensure-WingetPackage -Id 'Microsoft.VisualStudio.2022.BuildTools' -Override '--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'

$OllamaExe = Resolve-ToolPath -Name 'ollama.exe' -Fallbacks @(
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    "$env:ProgramFiles\Ollama\ollama.exe"
)
if (-not $OllamaExe) {
    throw 'ollama.exe was not found after installation. Open a new PowerShell window or launch Ollama once, then rerun setup_windows.ps1.'
}

$PythonBootstrap = Resolve-ToolPath -Name 'py.exe'
if ($PythonBootstrap) {
    if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
        & $PythonBootstrap -3.12 -m venv .venv
    }
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
        & python -m venv .venv
    }
}
else {
    throw 'Python was not found after installation. Open a new PowerShell window and rerun setup_windows.ps1.'
}

$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
    throw 'The virtual environment Python executable was not created. Rerun setup_windows.ps1.'
}

& $PythonExe -m pip install -U pip setuptools wheel

@"
import json
from pathlib import Path

repo = Path.cwd()
settings_path = repo / 'gmod_ai_studio_settings.json'
recommended_chat_model = '$RecommendedChatModel'
legacy_chat_models = {'', 'ministral-3:3b', 'llama3.2:1b', 'qwen3.5:4b'}
legacy_output_dirs = {'', 'training_runs/gmod-studio-windows', 'training_runs/llamacpp-llama32-1b-windows'}
legacy_ollama_model_names = {'', 'gmod_lua', 'gmod-llama32-1b-llamacpp'}

if settings_path.exists():
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
else:
    settings = {}


def replace_if_missing_or_legacy(mapping, key, value, legacy_values):
    current = str(mapping.get(key) or '').strip()
    if current in legacy_values:
        mapping[key] = value

app = settings.setdefault('app', {})
chat = settings.setdefault('chat', {})
dataset = settings.setdefault('dataset', {})
training = settings.setdefault('training', {})

app.setdefault('host', '127.0.0.1')
app.setdefault('port', 8765)
app.setdefault('auto_open_browser', True)
app.setdefault('poll_interval_ms', 2500)

replace_if_missing_or_legacy(chat, 'model', recommended_chat_model, legacy_chat_models)
chat.setdefault('memory_file', 'training_runs/chat_memory/session_memory.json')
chat.setdefault('training_log_file', 'output/chat_memory_train.jsonl')
chat.setdefault('messages_log_file', 'output/chat_memory_messages.jsonl')
chat.setdefault('max_history_messages', 48)
chat.setdefault('temperature', 0.1)
chat.setdefault('seed', 3407)
chat.setdefault('timeout', 600.0)
chat.setdefault('num_predict', 768)
chat.setdefault('device', 'auto')
chat.setdefault('enable_thinking', False)

dataset['script'] = 'scripts/prepare_training_data.py'
dataset.setdefault('extra_args', '')

training['platform'] = 'windows'
training['script'] = 'scripts/train_llamacpp_ollama_windows.py'
training['dataset_file'] = 'datasets/gmod_lua_coding_train.jsonl'
training['dataset_files'] = ['datasets/gmod_lua_coding_train.jsonl']
training['eval_file'] = 'datasets/gmod_lua_coding_eval.jsonl'
training['mlx_model'] = ''
training.setdefault('base_model_gguf', '')
replace_if_missing_or_legacy(training, 'base_ollama_model', recommended_chat_model, legacy_chat_models)
replace_if_missing_or_legacy(training, 'output_dir', 'training_runs/llamacpp-qwen25-coder-1_5b-windows', legacy_output_dirs)
training.setdefault('export_model_bundle', True)
training.setdefault('export_model_dir', 'models')
training.setdefault('memory_dataset_file', 'output/chat_memory_train.jsonl')
training.setdefault('skip_memory_dataset', False)
training.setdefault('limit', 128)
training.setdefault('offset', 0)
training.setdefault('context_length', 512)
training.setdefault('batch_size', 64)
training.setdefault('ubatch_size', 16)
training.setdefault('epochs', 1)
training.setdefault('iters', 300)
training.setdefault('grad_accumulation_steps', 4)
training.setdefault('num_layers', 8)
training.setdefault('steps_per_report', 10)
training.setdefault('steps_per_eval', 50)
training.setdefault('val_batches', 2)
training.setdefault('save_every', 50)
training.setdefault('gpu_layers', 0)
training.setdefault('threads', 8)
training.setdefault('seed', 3407)
training.setdefault('validation_split', 0.1)
training.setdefault('windows_acceleration', 'auto')
training.setdefault('mac_acceleration', 'auto')
training.setdefault('llama_bin_dir', '')
training.setdefault('prepare_only', False)
training.setdefault('create_ollama_model', True)
replace_if_missing_or_legacy(training, 'ollama_model_name', 'gmod-qwen25-coder-1_5b-llamacpp', legacy_ollama_model_names)
training.setdefault('grad_checkpoint', True)
training.setdefault('mask_prompt', True)
training.setdefault('dry_run', False)
training.setdefault('extra_args', '')

(repo / 'datasets' / 'scraped').mkdir(parents=True, exist_ok=True)
(repo / 'output').mkdir(parents=True, exist_ok=True)
(repo / 'training_runs' / 'chat_memory' / 'users').mkdir(parents=True, exist_ok=True)

settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
"@ | & $PythonExe -

try {
    & $OllamaExe list *> $null
}
catch {
    Start-Process -FilePath $OllamaExe -ArgumentList 'serve' -WindowStyle Hidden
    $ready = $false
    foreach ($attempt in 1..30) {
        Start-Sleep -Seconds 1
        try {
            & $OllamaExe list *> $null
            $ready = $true
            break
        }
        catch {
        }
    }

    if (-not $ready) {
        throw 'Ollama did not become ready in time. Start it manually with `ollama serve` and rerun setup_windows.ps1.'
    }
}

$Models = @(
    @"
import json
from pathlib import Path

settings = json.loads(Path('gmod_ai_studio_settings.json').read_text(encoding='utf-8'))
models = []
for value in [settings['chat'].get('model'), settings['training'].get('base_ollama_model'), '$RecommendedChatModel']:
    cleaned = str(value or '').strip()
    if cleaned and cleaned not in models:
        models.append(cleaned)
for model in models:
    print(model)
"@ | & $PythonExe -
) | Where-Object { $_.Trim() -ne '' }

foreach ($Model in $Models) {
    Write-Host "Pulling Ollama model: $Model"
    & $OllamaExe pull $Model
}

Download-FileIfMissing -Uri $RecommendedLlamaCppModelUrl -Destination $RecommendedLlamaCppModelPath

$WindowsAcceleration = (@"
import json
from pathlib import Path

settings = json.loads(Path('gmod_ai_studio_settings.json').read_text(encoding='utf-8'))
print(str(settings.get('training', {}).get('windows_acceleration', 'auto')).strip().lower() or 'auto')
"@ | & $PythonExe -).Trim()

if ($WindowsAcceleration -notin @('auto', 'cpu', 'cuda')) {
    $WindowsAcceleration = 'auto'
}

$CudaAvailable = $false
if (Resolve-ToolPath -Name 'nvcc.exe' -Fallbacks @("$env:CUDA_PATH\bin\nvcc.exe")) {
    $CudaAvailable = $true
}

$ResolvedAcceleration = 'cpu'
if ($WindowsAcceleration -eq 'cuda') {
    if (-not $CudaAvailable) {
        throw 'Windows acceleration is set to cuda, but nvcc.exe was not found. Install the CUDA toolkit or switch the setting to auto or cpu.'
    }
    $ResolvedAcceleration = 'cuda'
}
elseif ($WindowsAcceleration -eq 'auto' -and $CudaAvailable) {
    $ResolvedAcceleration = 'cuda'
}

$LlamaFinetune = Resolve-ToolPath -Name 'llama-finetune.exe'

if (-not $LlamaFinetune -or $ResolvedAcceleration -eq 'cuda') {
    $GitExe = Resolve-ToolPath -Name 'git.exe' -Fallbacks @(
        "$env:ProgramFiles\Git\cmd\git.exe",
        "$env:ProgramFiles\Git\bin\git.exe"
    )
    $CmakeExe = Resolve-ToolPath -Name 'cmake.exe' -Fallbacks @(
        "$env:ProgramFiles\CMake\bin\cmake.exe"
    )

    if (-not $GitExe) {
        throw 'git.exe was not found after installation. Open a new PowerShell window and rerun setup_windows.ps1.'
    }
    if (-not $CmakeExe) {
        throw 'cmake.exe was not found after installation. Open a new PowerShell window and rerun setup_windows.ps1.'
    }

    $LlamaRoot = Join-Path $HOME 'llama.cpp'
    if (-not (Test-Path $LlamaRoot)) {
        & $GitExe clone https://github.com/ggml-org/llama.cpp.git $LlamaRoot
    }

    $CmakeConfigureArgs = @('-S', $LlamaRoot, '-B', (Join-Path $LlamaRoot 'build'))
    if ($ResolvedAcceleration -eq 'cuda') {
        $CmakeConfigureArgs += '-DGGML_CUDA=ON'
    }

    & $CmakeExe @CmakeConfigureArgs
    & $CmakeExe --build (Join-Path $LlamaRoot 'build') --config Release --target llama-finetune llama-perplexity llama-cli llama-server
}

$BinCandidates = @(
    (Join-Path $HOME 'llama.cpp\build\bin\Release'),
    (Join-Path $HOME 'llama.cpp\build\bin')
)

$LlamaBinDir = $BinCandidates | Where-Object { Test-Path (Join-Path $_ 'llama-finetune.exe') } | Select-Object -First 1
if (-not $LlamaBinDir) {
    $ResolvedLlamaFinetune = Resolve-ToolPath -Name 'llama-finetune.exe'
    if ($ResolvedLlamaFinetune) {
        $LlamaBinDir = Split-Path -Parent $ResolvedLlamaFinetune
    }
}

if (-not $LlamaBinDir) {
    throw 'llama-finetune.exe could not be found after building llama.cpp.'
}

@"
import json
from pathlib import Path

settings_path = Path('gmod_ai_studio_settings.json')
settings = json.loads(settings_path.read_text(encoding='utf-8'))
settings['training']['platform'] = 'windows'
settings['training']['llama_bin_dir'] = r'$LlamaBinDir'
if not str(settings['training'].get('base_model_gguf') or '').strip():
    settings['training']['base_model_gguf'] = r'$RecommendedLlamaCppModelPath'
settings['training']['windows_acceleration'] = '$WindowsAcceleration'
if '$ResolvedAcceleration' == 'cpu':
    settings['training']['gpu_layers'] = 0
elif int(settings['training'].get('gpu_layers', 0) or 0) <= 0:
    settings['training']['gpu_layers'] = 999
settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
"@ | & $PythonExe -

& $PythonExe .\scripts\prepare_training_data.py

Write-Host ''
Write-Host 'Windows setup complete.'
Write-Host 'Installed:'
Write-Host "  - Ollama chat model: $RecommendedChatModel"
Write-Host "  - llama.cpp GGUF: $RecommendedLlamaCppModelPath"
Write-Host 'Next:'
Write-Host "  cd `"$RepoRoot`""
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  python .\run_gmod_ai_studio.py'