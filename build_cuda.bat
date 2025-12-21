@echo off
REM Build script for cover_cuda (GPU-accelerated version)
REM Requires NVIDIA CUDA Toolkit

echo.
echo Building cover_cuda (GPU-accelerated version)
echo ==============================================
echo.

REM Check for NVCC
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: NVCC not found!
    echo.
    echo Please install NVIDIA CUDA Toolkit from:
    echo   https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, make sure nvcc is in your PATH.
    echo Typical location: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin
    echo.
    goto :end
)

echo Found NVCC - compiling CUDA version...
echo.

REM Compile with common GPU architectures
REM Adjust -arch based on your GPU:
REM   sm_60 = Pascal (GTX 1080, etc.)
REM   sm_70 = Volta (V100, Titan V)
REM   sm_75 = Turing (RTX 2080, etc.)
REM   sm_80 = Ampere (RTX 3080, A100)
REM   sm_86 = Ampere (RTX 3090)
REM   sm_89 = Ada Lovelace (RTX 4090)

nvcc -O3 -o cover_cuda.exe cover_cuda.cu -arch=sm_60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80

if %errorlevel% == 0 (
    echo.
    echo SUCCESS! Built cover_cuda.exe
    echo.
    echo Usage: cover_cuda.exe t=2 k=3 m=2 v=7 b=7 runs=4096
    echo.
    echo Options:
    echo   runs=N      - Number of parallel SA runs on GPU (default: 4096)
    echo   All other options same as original cover program
    echo.
) else (
    echo.
    echo COMPILATION FAILED
    echo Please check your CUDA installation and GPU compatibility.
)

:end
pause



