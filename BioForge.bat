@echo off
REM ============================================================
REM  BioForge - lanzador de la app de escritorio (doble clic)
REM  Abre la ventana local. No conecta a ningun servidor.
REM ============================================================
chcp 65001 >nul
title BioForge
cd /d "%~dp0"
set PYTHONUTF8=1

echo(
echo   ====================================
echo    BioForge - iniciando la app local
echo   ====================================
echo(

REM --- elegir interprete de Python (py launcher o python) ---
set "PY="
where py  >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"

if not defined PY (
  echo   [X] No se encontro Python en este equipo.
  echo       Instalalo desde https://www.python.org/downloads/
  echo       y marca "Add Python to PATH" durante la instalacion.
  echo(
  pause
  exit /b 1
)

REM --- asegurar pywebview la primera vez ---
%PY% -c "import webview" 2>nul
if errorlevel 1 (
  echo   Instalando pywebview la primera vez, espera un momento...
  %PY% -m pip install pywebview
  echo(
)

REM --- abrir la app ---
echo   Abriendo la ventana...
%PY% "app\main.py"

if errorlevel 1 (
  echo(
  echo   *** Hubo un error al arrancar. ***
  echo   Copia el mensaje de arriba y pasamelo para arreglarlo.
  echo(
  pause
)
