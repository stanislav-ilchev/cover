@echo off
REM Build script for cover with ultra-optimized PRNG and sorting
REM Uses anneal_fast.c with xorshift64* PRNG and insertion sort

echo Building cover_ultra.exe (ultra-optimized version)...
echo.

REM Try to find GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo GCC not found in PATH, trying MSYS2...
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

REM Compile all object files with aggressive optimizations
echo Compiling with -O3 -march=native -flto...

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c cover.c -o cover.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c bincoef.c -o bincoef.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c tables.c -o tables.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c setoper.c -o setoper.o
if %errorlevel% neq 0 goto :error

REM Use the ultra-optimized anneal_fast.c
gcc -O3 -march=native -ffast-math -flto -funroll-loops -c anneal_fast.c -o anneal.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c solcheck.c -o solcheck.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c exp.c -o exp.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -c arg.c -o arg.o
if %errorlevel% neq 0 goto :error

REM Link with LTO
echo Linking with LTO...
gcc -O3 -march=native -flto -o cover_ultra.exe cover.o bincoef.o tables.o setoper.o anneal.o solcheck.o exp.o arg.o -lm
if %errorlevel% neq 0 goto :error

echo.
echo SUCCESS: cover_ultra.exe built!
echo.
echo Optimizations included:
echo   - Xorshift64* PRNG (faster than rand/random)
echo   - Insertion sort for small arrays (faster than qsort)
echo   - Link-time optimization (LTO)
echo   - Loop unrolling
echo   - Hardware popcount/ctz intrinsics
echo.
echo Usage:
echo   cover_ultra.exe v=27 k=6 m=4 t=3 b=86 CF=0.999 IT=0.6 frozen=1000000
echo.
goto :end

:error
echo.
echo BUILD FAILED!
echo Make sure GCC (MinGW) is installed.
echo Install MSYS2 from https://www.msys2.org/
echo Then run: pacman -S mingw-w64-x86_64-gcc

:end

