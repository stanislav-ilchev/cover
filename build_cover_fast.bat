@echo off
REM Build script for cover with bitmask-optimized covering computation
REM This version uses tables_fast.c instead of tables.c

echo Building cover_optimized.exe (bitmask-accelerated version)...
echo.

REM Try to find GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo GCC not found in PATH, trying MSYS2...
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

REM Compile all object files with high optimization
echo Compiling with -O3 -march=native...

gcc -O3 -march=native -ffast-math -c cover.c -o cover.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c bincoef.c -o bincoef.o
if %errorlevel% neq 0 goto :error

REM Use the optimized tables_fast.c instead of tables.c
gcc -O3 -march=native -ffast-math -c tables_fast.c -o tables.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c setoper.c -o setoper.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c anneal.c -o anneal.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c solcheck.c -o solcheck.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c exp.c -o exp.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -c arg.c -o arg.o
if %errorlevel% neq 0 goto :error

REM Link
echo Linking...
gcc -O3 -o cover_optimized.exe cover.o bincoef.o tables.o setoper.o anneal.o solcheck.o exp.o arg.o -lm
if %errorlevel% neq 0 goto :error

echo.
echo SUCCESS: cover_optimized.exe built!
echo.
echo The optimized version uses hardware popcount for faster covering computation.
echo Use with OntheFly=1 for best results:
echo.
echo   cover_optimized.exe v=27 k=6 m=4 t=3 b=86 CF=0.999 IT=0.6 frozen=1000000 OF=1
echo.
goto :end

:error
echo.
echo BUILD FAILED!
echo Make sure GCC (MinGW) is installed.
echo Install MSYS2 from https://www.msys2.org/
echo Then run: pacman -S mingw-w64-x86_64-gcc

:end

