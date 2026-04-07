@echo off
rem Run genesis_triton_diff.py with MSVC Triton support.
rem
rem Prerequisites:
rem   - Set VCVARS to your vcvars64.bat path, e.g.:
rem       set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
rem   - Set GENESIS_ROOT to your Genesis v3 model directory.
rem   - Run from the prism repo root, or set PRISM_ROOT to override.

if not defined VCVARS (
    echo error: VCVARS is not set. Point it to your vcvars64.bat.
    exit /b 1
)
if not defined GENESIS_ROOT (
    echo error: GENESIS_ROOT is not set. Point it to your orchOSModel-genesis-v3 directory.
    exit /b 1
)

call "%VCVARS%" >NUL
if errorlevel 1 (
    echo error: vcvars64.bat failed.
    exit /b 1
)

if defined PRISM_ROOT cd /d "%PRISM_ROOT%"
python scripts\genesis_triton_diff.py
