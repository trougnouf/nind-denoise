echo 'Downloading Commons:Featured Pictures dataset'
python ../common/tools/wikidownloader.py --cat FP --delete_uncat
# you can run the above twice to ensure all files are downloaded
echo 'Filtering dataset by ISO values'
python tools/filter_dataset_by_iso.py --data_dpath ../../datasets/FeaturedPictures --maxISO 200
echo 'Cropping dataset to 1024 px'
cd ../common
python tools/crop_ds.py --cs 1024 --ds_dir filtered/ISO200/FeaturedPictures
echo 'Verifying cropped images'
python tools/verify_images.py ../../datasets/filtered/ISO200/resized/1024/FeaturedPictures --save_img
cd nind_denoise
