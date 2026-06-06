#!/bin/bash
# Denoise the input image using the NIND method
# Usage: ./denoise.sh input_image

uv run denoise_image.py \
    --input "$1" \
    --output "denoised_$1.tiff" \
    --model_path ../../models/nind_denoise/2021-05-23T10_16_nn_train.py_--config_configs-train_conf_unet.yaml_--debug_options_output_val_images_keep_all_output_images_--test_interval_0_--epochs_1000_--reduce_lr_factor_0.95_--patience_3/generator_734.pt \
    --network UNet
    
