@echo off
REM Build script for cover_big_cuda.cu
REM Optimized for large covering design problems like L(49,6,6,3)

echo Building CUDA Big Covering Design Solver...

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

REM CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

REM Optional overrides
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_86
if "%DELTA_THREADS%"=="" set DELTA_THREADS=256

REM Check if nvcc exists
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: nvcc not found. Please install CUDA Toolkit.
    exit /b 1
)

echo Compiling cover_big_cuda.cu...
nvcc -O3 -arch=%CUDA_ARCH% -use_fast_math -DDELTA_THREADS=%DELTA_THREADS% -o cover_big_cuda.exe cover_big_cuda.cu

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build successful! Created cover_big_cuda.exe
    echo.
    echo Usage examples:
    echo   cover_big_cuda.exe v=49 k=6 m=6 t=3 b=163
    echo   cover_big_cuda.exe v=27 k=6 m=4 t=3 b=85 runs=8192 iter=500000
    echo.
    echo Parameters:
    echo   v, k, m, t, b  - Covering design parameters
    echo   runs           - Number of parallel SA runs ^(default: 4096^)
    echo   iter           - Iterations per SA run ^(default: 100000^)
    echo   sample         - Sample size for cost estimation ^(default: 10000^)
    echo   temp           - Initial temperature ^(default: 1000.0^)
    echo   cool           - Cooling rate ^(default: 0.999^)
    echo   rounds         - Number of SA rounds ^(default: 100^)
    echo   output         - Output file ^(default: solution.txt^)
) else (
    echo.
    echo Build failed!
)
