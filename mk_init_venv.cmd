@echo off
setlocal

pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\activate.bat" (
    python -m venv ".venv"
    if errorlevel 1 (
        echo Erro ao criar o ambiente virtual. Verifique a instalacao do Python.
        pause
        popd
        exit /b 1
    )
)

cmd /k "call .venv\Scripts\activate.bat"

popd
endlocal