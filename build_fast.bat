@echo off
REM Build script for bitmask-optimized cover programs
REM Requires GCC (MinGW) - download from https://www.msys2.org/

echo Building cover_fast.exe and cover_ultra.exe...
echo.

REM Try to find GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo GCC not found in PATH, trying MSYS2...
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

REM Build cover_fast with maximum optimization
echo Compiling cover_fast.c...
gcc -O3 -march=native -mtune=native -ffast-math -flto -DNDEBUG ^
    -o cover_fast.exe cover_fast.c -lm

if %errorlevel% neq 0 (
    echo.
    echo Failed to compile cover_fast.c
    echo Make sure GCC is installed. Install MSYS2 from https://www.msys2.org/
    echo Then run: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)

echo cover_fast.exe built successfully!

REM Build cover_ultra with maximum optimization
echo Compiling cover_ultra.c...
gcc -O3 -march=native -mtune=native -ffast-math -flto -DNDEBUG ^
    -o cover_ultra.exe cover_ultra.c -lm

if %errorlevel% neq 0 (
    echo.
    echo Failed to compile cover_ultra.c
    exit /b 1
)

echo cover_ultra.exe built successfully!
echo.
echo Usage examples:
echo   cover_fast.exe v=27 k=6 m=4 t=3 b=86 CF=0.999 frozen=1000000
echo   cover_ultra.exe v=27 k=6 m=4 t=3 b=85 CF=0.999 IT=0.5 frozen=10000000 EL=0
echo.
echo To search for world record (b=85 instead of 86):
echo   cover_ultra.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 frozen=100000000 EL=0 verbose=2

