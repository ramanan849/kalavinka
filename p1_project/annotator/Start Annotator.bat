@echo off
title P1 Gold Annotator
cd /d "%~dp0annotator"
echo Starting the annotator... keep this window open while you work.
python server.py
if errorlevel 1 (
  echo.
  echo Could not start with "python". Trying "py"...
  py server.py
  pause
)
