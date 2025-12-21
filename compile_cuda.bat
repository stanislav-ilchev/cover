@echo off
REM Set up Visual Studio 2022 environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Add CUDA to PATH
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Compile
cd /d "C:\Users\Stanislav Ilchev\Desktop\cover"
nvcc -O3 -o cover_cuda.exe cover_cuda.cu -arch=sm_75 -Xcompiler "/W0"

if %errorlevel% == 0 (
    echo.
    echo SUCCESS! Built cover_cuda.exe
) else (
    echo.
    echo Compilation failed.
)
pause


