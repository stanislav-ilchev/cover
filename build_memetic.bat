@echo off
echo Building CUDA Memetic Algorithm...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

nvcc -O3 -arch=sm_50 -o cover_memetic_cuda.exe cover_memetic_cuda.cu

if %ERRORLEVEL% equ 0 (
    echo Build successful! Created cover_memetic_cuda.exe
) else (
    echo Build failed with error %ERRORLEVEL%
)

