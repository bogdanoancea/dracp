@echo off
REM DRACP -- single entry point for every experiment (Windows).
REM
REM   run.bat                 tier "main": what the paper needs
REM   run.bat --tier quick    smoke test (minutes) -- run this first
REM   run.bat --tier full     everything, including hyperparameter sweeps
REM   run.bat montecarlo report   individual stages
REM
REM All results land in a single results\ directory.
setlocal
cd /d "%~dp0"

if not exist "experiments\common.py" (
  echo ERROR: run this from the repository root ^(experiments\common.py not found^).
  exit /b 1
)

if "%PYTHON_BIN%"=="" (
  where py >nul 2>&1 && (set "PYTHON_BIN=py -3") || (set "PYTHON_BIN=python")
)

%PYTHON_BIN% --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: no usable Python found. Install Python 3 or set PYTHON_BIN.
  exit /b 1
)

%PYTHON_BIN% run.py %*
exit /b %errorlevel%
