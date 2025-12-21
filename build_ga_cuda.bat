@echo off
echo Building CUDA Genetic Algorithm for Covering Designs...

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

REM CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

REM Check if nvcc exists
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: nvcc not found. Please install CUDA Toolkit.
    exit /b 1
)

echo Compiling cover_ga_cuda.cu...
nvcc -O3 -arch=sm_50 -o cover_ga_cuda.exe cover_ga_cuda.cu

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build successful! Created cover_ga_cuda.exe
    echo.
    echo Usage examples:
    echo   cover_ga_cuda.exe v=27 k=6 m=4 t=3 b=86 pop=8192 gen=5000
    echo   cover_ga_cuda.exe v=27 k=6 m=4 t=3 b=85 pop=16384 gen=10000 mut=0.15
    echo.
    echo Parameters:
    echo   v, k, m, t, b  - Covering design parameters
    echo   pop            - Population size (default: 8192)
    echo   gen            - Number of generations (default: 10000)
    echo   mut            - Mutation rate (default: 0.1)
    echo   tour           - Tournament size (default: 4)
    echo   elite          - Number of elite individuals (default: 10)
) else (
    echo.
    echo Build failed!
)

