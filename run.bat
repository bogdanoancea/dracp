@echo off
REM Thin wrapper for Windows: the pipeline lives in run.py.
REM   run.bat                     all stages
REM   run.bat montecarlo report   selected stages
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (py -3 run.py %*) || (python run.py %*)
endlocal
