@echo off
REM Simple parallel launcher - runs 8 cover.exe instances
REM Each with slightly different InitTemp to explore different regions

echo Starting 8 parallel cover searches for b=85 (world record attempt)...
echo.

start "Cover Job 1" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.40 frozen=100000000 EL=0 result=result1.res log=log1.txt
start "Cover Job 2" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.45 frozen=100000000 EL=0 result=result2.res log=log2.txt
start "Cover Job 3" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.50 frozen=100000000 EL=0 result=result3.res log=log3.txt
start "Cover Job 4" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.55 frozen=100000000 EL=0 result=result4.res log=log4.txt
start "Cover Job 5" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.60 frozen=100000000 EL=0 result=result5.res log=log5.txt
start "Cover Job 6" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.65 frozen=100000000 EL=0 result=result6.res log=log6.txt
start "Cover Job 7" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.70 frozen=100000000 EL=0 result=result7.res log=log7.txt
start "Cover Job 8" cover.exe v=27 k=6 m=4 t=3 b=85 CF=0.9999 IT=0.75 frozen=100000000 EL=0 result=result8.res log=log8.txt

echo All 8 jobs started!
echo.
echo Check result1.res through result8.res for solutions.
echo If any file has content, that's your solution!
echo.
echo To stop all jobs: taskkill /IM cover.exe /F

