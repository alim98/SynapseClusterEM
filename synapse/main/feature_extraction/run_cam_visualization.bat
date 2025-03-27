@echo off
echo Generating animated GIF attention maps for segmentation types 10, 11, 12, and 13...
python visualize_segmentation_cam.py
echo GIF visualization complete.
pause 