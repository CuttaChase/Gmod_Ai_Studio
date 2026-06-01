@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "POWERSHELL_SCRIPT=%SCRIPT_DIR%setup_windows.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%POWERSHELL_SCRIPT%" (
    echo setup_windows.ps1 was not found next to this batch file.
    pause
    exit /b 1
)

if not exist "%POWERSHELL_EXE%" (
    set "POWERSHELL_EXE=powershell.exe"
)

echo Running Windows prerequisite installer...
echo This will install Python, Ollama, Git, CMake, Visual Studio Build Tools,
echo and the recommended small coding models for chat and llama.cpp.
echo.

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%POWERSHELL_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo setup_windows.ps1 failed with exit code %EXIT_CODE%.
    echo Review the error output above, then rerun this file or setup_windows.ps1 directly.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo Windows prerequisite install finished successfully.
exit /b 0