@echo off
cd /d "%~dp0"
title Akilli Cerrahi Asistan - Sunucu

echo.
echo  ================================================
echo   Akilli Cerrahi Asistan  ^|  Yakin Dogu Uni.
echo  ================================================
echo.

echo [1/2] Flask kontrol ediliyor...
.venv\Scripts\python.exe -m pip install flask --quiet 2>nul
if errorlevel 1 (
    python -m pip install flask --quiet 2>nul
)

echo [2/2] Sunucu baslatiliyor...
echo.
echo  Tarayicinizda acin: http://localhost:5000
echo  Kapatmak icin bu pencerede CTRL+C
echo.

.venv\Scripts\python.exe server.py
if errorlevel 1 (
    python server.py
)

pause
