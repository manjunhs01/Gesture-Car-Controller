@echo off
REM Presentation launcher: big always-on-top preview (camera, hands, rotating steering wheel, gas/brake bars), realistic steering feel.
REM Keys go to the focused window: after "calibrated", click your game / browser tab.
cd /d "%~dp0"
call run.bat --preview --presentation --preview-scale 0.55 --profile presentation %*
