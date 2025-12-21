@echo off
echo Building CUDA Merge-Diff Search...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

nvcc -O3 -arch=sm_50 -o cover_cuda_mergediff.exe cover_cuda_mergediff.cu

if %ERRORLEVEL% equ 0 (
    echo Build successful! Created cover_cuda_mergediff.exe
    echo.
    echo Usage: cover_cuda_mergediff.exe v=27 k=6 m=4 t=3 b=86 runs=256 iter=100000 rounds=100
) else (
    echo Build failed!
)

