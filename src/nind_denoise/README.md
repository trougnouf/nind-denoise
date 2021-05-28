# mthesis-denoise

Master thesis on natural image noise removal using Convolutional Neural Networks. Works with the Natural Image Noise Dataset to apply to real photographs, using a UNet network architecture by default.

Much lighter version of https://github.com/trougnouf/mthesis-denoise

TODO: tensorflow implementation (has C API which would work with darktable),

## test

### denoise an image

Requirements: pytorch [, exiftool]

```
python3 denoise_image.py --cs <CS> --ucs <UCS> --model_path models/[model.pth] -i <input_image_path> [-o output_image_path] --network <architecture> [--device #]
```
where device is 0 for a CUDA-capable GPU, or -1 for CPU (slow)

eg (**recommended for most applications**):
`python3 denoise_image.py --cs 512 --ucs 384 --network UNet --model_path "../../models/2019-02-18T20:10_run_nn.py_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--skip_sizecheck_--lr_3e-4" -i <input_image_path> [-o output_image_path]`

or if you have a UtNet model (likely even better):
`python denoise_image.py --cs 552 --ucs 540 --network UtNet --model_path <PT_MODEL_PATH> --input <IMAGE_TO_DENOISE>`

Note that you should run this on images which have not been sharpened, and apply sharpening then.

### test a trained model on the test_reserve

The following script tests a model on data it has not seen during training (--test_reserve list),
and reports a final ms-ssim, ssim, and mse losses.

```
python denoise_dir.py --cs 552 --ucs 540 --network UtNet --model_path <PT_MODEL_PATH>
```
(add originally downloaded dataset information with --orig_data, eg ../../datasets/NIND if it is not
in its default location)

You can use "--device -1" to perform the test on CPU. Slow but useful if you are already using the
GPU for training.

## train

Requirements: python-pytorch, python-configargparse >= 1.3, python-opencv, python-piqa, torchvision, bash, imagemagick, python-opencv, libjpeg[-turbo] [, wget]

eg installation on (Arch Linux): 

```
sudo pacman -S python-pytorch-opt-cuda python-opencv imagemagick libjpeg-turbo wget
pacaur -Se python-torchvision-cuda  # I highly recommend removing the check() function because the tests take forever
pacaur -S python-pytorch-piqa python-configargparse-git  # configargparse is outdated in the arch repo and results in 'None' (string) values.
```
eg pip installation (note that imagemagick and libjpeg(<-turbo>) are not python packages and need to be installed for your distribution):

```
pip3 install --user torch torchvision piqa ConfigArgParse 
pip3 install --user opencv-python # recommended to install python-opencv from your distribution's package manager instead
echo 'Don't forget to install imagemagick and libjpeg'
```


### Dataset gathering

Note that cropping png files is currently extremely slow for non-JPEG images, due to the cropping process opening the whole file every time it makes a new crop.
TODO use opencv and open once save many for tiff/png images.

```bash
python3 tools/dl_ds_1.py --use_wget         # --use_wget is much less likely to result in half-downloaded files
python3 tools/crop_ds.py --cs 256 --stride 192  # this takes a long time.
python tools/pick_validation_set.py --num_crops 300
```

Optionally, run "python tools/make_dataset_crops_list.py" (with the same dataset-related options as above) to generate a list of crops with ms-ssim loss in datasets/<dsname>-msssim.csv. (Provided for NIND_256_192) This can be useful if you would like to modify the dataset to handle a simpler list and only take images above a given quality threshold.

Cropping can be accelerated greatly by temporarily placing the downloaded dataset (approximately 30.1 GB) on a SSD.

If you would like to train with a different crop size / ucs combination or test set, you must run tools/pick\_validation\_set.py with the appropriate parameters.

A valid dataset size is determined as follow:
- The minimum size for MS-SSIM is 161 (as determined in common.pt_losses). Account for 25% border with U-Net. MS-SSIM does the equivalent of /16 then conv 11; 176px appears to be an ideal size
- U-Net: conv: -2px, down: /2. -> ((((dim−4)÷2−4)÷2−4)÷2−4)÷2−4 should be whole-> 172, 188, 204, 220. UtNet (not fully tested): bottom: (((176÷2−4)÷2−4)÷2−4)÷2−2 (184 px works)
- JPEG lossless crop: must be divideable by /16
-> crop size of 256 px with 64 px offset; random crop of 220, central loss of 

In the end you should have ROOT/datasets/cropped/NIND_256_192/*/* which takes up approximately 46.4 GB. Consider putting that directory on a SSD for faster training.


### Train U-Net denoiser


```bash
# batch_size 20 is for an 8 GB GPU, adjust for available VRAM
# train a single U-Net generator:
python3 nn_train.py --config configs/train_conf_unet.yaml --batch_size 20 --train_data ../../datasets/train/NIND_256_192
```

### Train UtNet denoiser

UtNet uses transposed convolutions instead of convolutions in the decoder, resulting in a
symmetrical encoder/decoder and no padding necessary; fewer resources and better denoising.
This network is recommended.

```bash
# batch_size 30 is for an 8 GB GPU, adjust for available VRAM
# train a single UtNet generator:
python3 nn_train.py --config configs/train_conf_utnet_std.yaml --batch_size 30 --train_data ../../datasets/train/NIND_256_192
```

### Train with a discriminator (cGAN, highly experimental)
```bash
python3 nn_train.py --d_network Hulf112Disc --batch_size 10
```

note that run\_nn.py contains slightly more options (such as compression and artificial noise) but it only trains one network at a time. nn\_train.py can currently train one generator and two discriminators.
