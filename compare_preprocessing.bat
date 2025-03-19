@echo off
python main/compare_csvs.py ^
"C:\Users\alim9\Documents\codes\synapse2\results\run_2025-03-18_19-08-06\features_extraction_stage_specific_layer20_segNone_alphaNone\features_layer20_segNone_alphaNone.csv" ^
"C:\Users\alim9\Documents\codes\synapse2\results\run_2025-03-18_18-12-54\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv" ^
--output_dir "C:\Users\alim9\Documents\codes\synapse2\results\preprocessing_comparison" ^
--label1 "Normal Cropping" ^
--label2 "Intelligent Cropping w=7"

echo.
echo Comparison completed. Results saved to: C:\Users\alim9\Documents\codes\synapse2\results\preprocessing_comparison
echo. 