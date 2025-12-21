@echo off
REM Build script for cover_cuda_v2 - the properly parallelized GPU version

echo Building cover_cuda_v2.exe...
echo.

REM Find CUDA
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0"
) else (
    echo CUDA not found! Please install NVIDIA CUDA Toolkit.
    exit /b 1
)

echo Using CUDA at: %CUDA_PATH%
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM Detect GPU architecture
echo Detecting GPU...
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul

REM Compile for common architectures
echo.
echo Compiling...
nvcc -O3 -arch=sm_75 -o cover_cuda_v2.exe cover_cuda_v2.cu

if %errorlevel% neq 0 (
    echo.
    echo Build failed! Trying with sm_60...
    nvcc -O3 -arch=sm_60 -o cover_cuda_v2.exe cover_cuda_v2.cu
)

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo SUCCESS: cover_cuda_v2.exe built!
echo.
echo Usage:
echo   cover_cuda_v2.exe v=27 k=6 m=4 t=3 b=86 CF=0.999 IT=0.6 frozen=1000 EL=0 blocks=64 L=1000
echo.
echo   blocks = number of parallel SA processes on GPU (default 64)
echo   L = iterations per temperature level (default 1000)


