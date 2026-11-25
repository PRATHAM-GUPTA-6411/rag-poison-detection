@echo off
REM One-click run of the v4 experiments (both datasets, then Mistral-7B). Ollama must be running.
setlocal
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
for %%D in (nq msmarco) do (
  echo ===== %%D: generating adaptive attacks =====
  python 01b_generate_attacks.py --dataset %%D || goto :err
  python 01_prepare_data.py --dataset %%D --poison-only || goto :err
  echo ===== %%D: signals =====
  python 02_retrieve_signals.py --dataset %%D || goto :err
  echo ===== %%D: detection =====
  python 03_detect.py --dataset %%D || goto :err
  python 04_generate_asr.py --dataset %%D --no-llm || goto :err
  echo ===== %%D: Llama-3-8B =====
  python 04_generate_asr.py --dataset %%D || goto :err
)
echo ===== downloading Mistral-7B =====
"%OLLAMA_EXE%" pull mistral:7b || goto :err
set OLLAMA_MODEL=mistral:7b
for %%D in (nq msmarco) do (
  echo ===== %%D: Mistral-7B =====
  python 04_generate_asr.py --dataset %%D || goto :err
)
echo.
echo ALL DONE. Results are in the results folder.
goto :eof
:err
echo.
echo FAILED at the step above. Send a screenshot of this window.
exit /b 1
