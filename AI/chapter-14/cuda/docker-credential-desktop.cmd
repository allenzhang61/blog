@echo off
if "%1"=="list" (
  echo {}
  exit /b 0
)
if "%1"=="get" (
  echo {"ServerURL":"","Username":"","Secret":""}
  exit /b 0
)
exit /b 0
