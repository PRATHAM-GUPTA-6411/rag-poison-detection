@echo off
REM Usage: run_all.bat nq      (or msmarco). Ollama must be running with the model pulled for step 4.
set DS=%1
if "%DS%"=="" set DS=nq
python 01_prepare_data.py --dataset %DS% || exit /b 1
python 02_retrieve_signals.py --dataset %DS% || exit /b 1
python 03_detect.py --dataset %DS% || exit /b 1
python 04_generate_asr.py --dataset %DS% --no-llm || exit /b 1
python 04_generate_asr.py --dataset %DS% || exit /b 1
echo Done. Results in results\
