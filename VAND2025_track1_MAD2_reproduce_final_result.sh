# test on public to get threshold file 
python test_public.py --data_root /path/to/your/mvtec_ad_2 --results_dir ./results_dir

# test call
python test_private.py --data_root /path/to/your/mvtec_ad_2 --results_dir ./results_VAND2025_superad

# check and prepare data for upload
python check_and_prepare_data_for_upload.py --submission_path ./results_VAND2025_superad/submission_folder