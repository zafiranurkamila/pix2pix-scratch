@echo off
rem ===========================================================================
rem  Jalankan semua eksperimen tuning berurutan: train -> evaluate tiap run,
rem  lalu gabungkan hasilnya (tabel, grid perbandingan, grafik).
rem
rem  Pemakaian (dari folder mana saja):
rem    run_experiments.bat          eksperimen asli, 200 epoch per run
rem    run_experiments.bat dummy    tes cepat pakai data dummy (4 epoch per run)
rem
rem  Aman dijalankan ulang: run yang sudah selesai langsung lewat (resume
rem  dari last.pt di epoch terakhir), run yang terputus dilanjutkan.
rem
rem  Log lengkap tiap run ada di outputs\_batch_logs\ (console hanya
rem  menampilkan status mulai/selesai).
rem ===========================================================================
setlocal
cd /d "%~dp0"

rem Tanpa ini Python menahan output print saat dialihkan ke file,
rem sehingga log di outputs\_batch_logs\ kosong sampai run selesai.
set PYTHONUNBUFFERED=1

rem Ganti kalau perlu, misal: set PY=.venv\Scripts\python.exe
set PY=py
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set PREFIX=
set EXTRA=
if /i "%~1"=="dummy" (
    set PREFIX=dummy_
    set EXTRA=--dummy --epochs 4 --decay_start_epoch 2 --save_every 2
)

set LOGDIR=outputs\_batch_logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set FAILED=0

echo [%date% %time%] Mulai semua eksperimen (prefix: "%PREFIX%")

rem          nama run     argumen yang diubah dari baseline
call :run    baseline     ""
call :run    l1_only      "--lambda_adv 0"
call :run    lambda10     "--lambda_l1 10"
call :run    lr1e-4       "--lr 1e-4"
call :run    bs4          "--batch_size 4"

rem Opsional: ukur variasi antar seed (hapus "rem" di depan untuk mengaktifkan)
rem call :run  baseline_seed1  "--seed 1"
rem call :run  baseline_seed2  "--seed 2"

echo [%date% %time%] Menggabungkan hasil...
%PY% -m src.compare_runs --runs %PREFIX%baseline %PREFIX%l1_only %PREFIX%lambda10 %PREFIX%lr1e-4 %PREFIX%bs4 --out outputs\%PREFIX%comparison > "%LOGDIR%\%PREFIX%compare.log" 2>&1
if errorlevel 1 (
    echo   GAGAL compare_runs, lihat %LOGDIR%\%PREFIX%compare.log
    set FAILED=1
)

echo [%date% %time%] Selesai.
if "%FAILED%"=="1" (
    echo Ada langkah yang GAGAL, cek log di %LOGDIR%
    exit /b 1
)
echo Hasil gabungan: outputs\%PREFIX%comparison
exit /b 0


rem ---------------------------------------------------------------------------
rem  :run <nama> "<argumen>"  -> train lalu evaluate satu run
rem ---------------------------------------------------------------------------
:run
set NAME=%PREFIX%%~1
echo [%date% %time%] Train    %NAME%  %~2
%PY% -m src.train --run_name %NAME% %~2 %EXTRA% > "%LOGDIR%\%NAME%_train.log" 2>&1
if errorlevel 1 (
    echo   GAGAL train %NAME%, lihat %LOGDIR%\%NAME%_train.log
    set FAILED=1
    goto :eof
)
echo [%date% %time%] Evaluate %NAME%
%PY% -m src.evaluate --run_name %NAME% > "%LOGDIR%\%NAME%_eval.log" 2>&1
if errorlevel 1 (
    echo   GAGAL evaluate %NAME%, lihat %LOGDIR%\%NAME%_eval.log
    set FAILED=1
    goto :eof
)
goto :eof
