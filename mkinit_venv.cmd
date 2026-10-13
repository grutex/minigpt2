@echo off
setlocal

pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\activate.bat" (
    python -m venv ".venv"
    if errorlevel 1 goto :erro
)

if not exist "requirements.txt" (
    type nul > "requirements.txt"
)

".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto :erro

cmd /k "call .venv\Scripts\activate.bat"

popd
endlocal
exit /b 0

:erro
echo Erro ao preparar o ambiente ou instalar as dependencias.
pause
popd
endlocal
exit /b 1