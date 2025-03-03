@echo off
REM Windows command file to run the visualization tool
REM This simply calls the bash script with all arguments passed through

bash "%~dp0\run_visualization.sh" %* 