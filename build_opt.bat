@echo off
REM Build script for cover with aggressive compiler optimizations
REM Uses standard anneal.c but with LTO and other optimizations

echo Building cover_opt.exe (aggressively optimized)...
echo.

REM Try to find GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo GCC not found in PATH, trying MSYS2...
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

REM Compile with aggressive optimizations
echo Compiling with -O3 -march=native -flto -funroll-loops...

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c cover.c -o cover.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c bincoef.c -o bincoef.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c tables.c -o tables.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c setoper.c -o setoper.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c anneal_opt.c -o anneal.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c solcheck.c -o solcheck.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c exp.c -o exp.o
if %errorlevel% neq 0 goto :error

gcc -O3 -march=native -ffast-math -flto -funroll-loops -fomit-frame-pointer -c arg.c -o arg.o
if %errorlevel% neq 0 goto :error

REM Link with LTO
echo Linking with LTO...
gcc -O3 -march=native -flto -fomit-frame-pointer -o cover_opt.exe cover.o bincoef.o tables.o setoper.o anneal.o solcheck.o exp.o arg.o -lm
if %errorlevel% neq 0 goto :error

echo.
echo SUCCESS: cover_opt.exe built!
echo.
echo Optimizations:
echo   - Link-time optimization (LTO)
echo   - Loop unrolling  
echo   - Fast math
echo   - Native CPU instructions
echo   - Omit frame pointer
echo.
goto :end

:error
echo.
echo BUILD FAILED!

:end

