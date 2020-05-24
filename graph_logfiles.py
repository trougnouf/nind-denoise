import os
import argparse
import matplotlib.pyplot as plt
import numpy
from random import random
from math import floor
from dataset_torch_3 import sortISOs
import csv
import json
import graph_utils

# generate results from log file
# python graph_logfiles.py --experiment "Hul112Disc filters, final activation, and batch normalization" --yaxis "Average MSE loss" --uneven_graphs --pre Loss_D: --smoothing_factor 600
# python graph_logfiles.py --experiment "Hul(b)128Net" --yaxis "Average 1-SSIM loss" --smoothing_factor 20 --uneven_graphs
# python graph_logfiles.py --experiment "Label smoothing" --yaxis "Average MSE loss (discriminator)" --pre "Loss_D: " --post " (" --smoothing_factor 2500 --uneven_graphs
# python graph_logfiles.py --experiment "cGAN training performance" --yaxis "Average SSIM loss during training" --pre "Average SSIM loss: " --smoothing_factor 1 --uneven_graphs --xaxis Epoch
default_smoothing_factor = 50

# Params
parser = argparse.ArgumentParser(description='Grapher')
parser.add_argument('--xaxis', type=str)
parser.add_argument('--yaxis', type=str, default='Average MSE')
parser.add_argument('--experiment', type=str)
parser.add_argument('--list_experiments', action='store_true')
parser.add_argument('--smoothing_factor', type=int, default=default_smoothing_factor)
parser.add_argument('--uneven_graphs', action='store_true', help='Allow some graphs to display more data points')
parser.add_argument('--pre', type=str, default="What precedes the variable to graph")
parser.add_argument('--post', type=str, default="What follows the variable to graph")
args = parser.parse_args()

if args.xaxis is None:
    xaxis = "iterations * %u"%(args.smoothing_factor)
else:
    xaxis = args.xaxis

experiments = dict()
# these should be classes w/ all the parameters, oh well
experiments['Hul112Disc filters, final activation, and batch normalization'] = {
#    "None-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_No_activation",
#    "Sigmoid-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_Sigmoid",
#    "PReLU-bc": "results/train/2019-05-22-D_sanity_check_not_conditional",
#    "Final pooling-bc": "results/train/2019-05-22-D_sanity_check_finalpool",
    "PReLU FA, funit=16, final max pooling": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_finalpool",
    "Sigmoid FA, funit=16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_sigmoid",
    "None FA, funit=16": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_no_activation",
"PReLU FA, funit=16, final max pooling, no BN": "results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16_finalpool",
"PReLU FA, funit=16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16",
"PReLU FA, funit=32":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit32",
"PReLU FA, funit=16, no BN":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16",
"PReLU FA, funit=32, no BN":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit32",
#"LeakyReLU activations,  funit=24":"results/train/2019-05-26-D_sanity_check_Hull112Disc_fixed_funit24_LeakyReLU",
#"LeakyReLU FA, funit=24": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU",
#"LeakyReLU FA, funit=24, final max pooling": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU_finalpool"
}
experiments['Hul(b)128Net'] = {"BN, PReLU": "results/train/2019-05-23-Hulb128Net-BN",
"No BN, PReLU": "results/train/2019-05-23-Hulb128Net-NoBN",
"No BN, no final activation": "results/train/2019-05-23-Hulb128Net-NoBN-NoAct"}

experiments["Label smoothing"] = {
"Always noisy labels": "results/train/2019-05-28-Hulb128Disc-very_noisy_probabilities",
"Noisy positive labels": "results/train/2019-05-28-Hulb128Disc-std_noisy_probabilities"
}

experiments["cGAN training performance"] = {
"U-Net (no discriminator)": ["results/train/2019-02-18T20:10_run_nn.py_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--skip_sizecheck_--lr_3e-4-stdlog"],
"Hulb128Net (no discriminator)": ["results/train/2019-06-01T00:03_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--d_network_PatchGAN_--batch_size_18", "results/train/2019-06-03T07:47_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--d_network_PatchGAN_--batch_size_18_--start_epoch_32_--g_weights_dict_path_models-2019-06-01T00:03_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--d_network_PatchGAN_--batch_size_18-generat", "results/train/2019-06-06T13:15_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--batch_size_18_--start_epoch_64_--patience_2_--g_weights_dict_path_models-2019-06-03T07:47_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--d_network_PatchGAN_--batch_size_18_--start_epoch_3", "results/train/2019-06-07T07:17_nn_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--batch_size_18_--start_epoch_71_--g_lr_0.000225_--patience_2_--g_model_path_models-2019-06-06T13:15_gan_train.py_--weight_L1_0.2_--weight_SSIM_0.8_--batch_size_18_--start_epoch_64_--patience_"],
#"Hulf112Disc-2": ["results/train/2019-06-02T15:05_gan_train.py_--weight_SSIM_0.2_--weight_L1_0.05_--cuda_device_1_--batch_size_18_--d_funit_31_--d_network_Hulf112Disc"],
"Hul112Disc": ["results/train/2019-05-31T10:56_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--d_funit_32", "results/train/2019-05-31T15:52_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--d_funit_32_--d_weights_dict_path_models-2019-05-31T10:56_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--d_funit_32-discri", "results/train/2019-05-31T23:48_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--d_funit_32_--d_weights_dict_path_models-2019-05-31T15:52_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--d_funit_32_--d_we"],
"Hulf112Disc": [
"results/train/2019-06-05T22:44_gan_train.py_--weight_SSIM_0.2_--weight_L1_0.05_--cuda_device_1_--batch_size_18_--d_funit_29_--g_funit_29_--d_network_Hulf112Disc"
],
"PatchGAN": ["results/train/2019-05-31T23:53_gan_train.py_--weight_SSIM_0.2_--weight_L1_0.05_--cuda_device_1_--batch_size_18_--d_network_PatchGAN"],
"Hul112Disc (0.1 advantage)": ["results/train/2019-06-03T07:53_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--discriminator_advantage_0.1"],
"Hulf112Disc (0.1 advantage)": ["results/train/2019-06-05T22:44_gan_train.py_--batch_size_18_--cuda_device_2_--weight_SSIM_0.2_--weight_L1_0.05_--discriminator_advantage_0.1_--d_funit_29_--g_funit_29_--d_network_Hulf112Disc"],




#"Not conditional Hul112Disc": ["results/train/2019-05-31T12:39_gan_train.py_--not_conditional_--weight_SSIM_0.2_--weight_L1_0.05_--cuda_device_1_--batch_size_18_--d_funit_32"],
}

if args.list_experiments:
    print(experiments.keys())
    exit(0)

experiment_raw = experiments[args.experiment]

data = dict()
markers = graph_utils.make_markers_dict(components=experiment_raw.keys())
smallest_log = None

for component, path in experiment_raw.items():
    # if path is list
    if isinstance(path, list):
        data[component] = []
        for logfile_path in path:
            data[component]+=graph_utils.parse_log_file(logfile_path, smoothing_factor = args.smoothing_factor, pre=args.pre, post=args.post)
    else:
        data[component] = graph_utils.parse_log_file(path, smoothing_factor = args.smoothing_factor, pre=args.pre, post=args.post)
    if smallest_log is None or smallest_log > len(data[component]):
        smallest_log = len(data[component])
for component in data:
    if not args.uneven_graphs:
        data[component] = data[component][0:smallest_log]
    plt.plot(data[component], label=component, marker = markers[component])#, marker=markers[component])
plt.title(args.experiment)
plt.ylabel(args.yaxis)
plt.xlabel(xaxis)
plt.grid()
plt.legend()
plt.show()

# images = list(data[components[0]]['results'].keys())
# for image in images:
#     _, isos = sortISOs(list(data[components[0]]['results'][image].keys()))
    #isos = baseisos + isos
#     for component in components:
#         try:
#             ssimscore = [data[component]['results'][image][iso][args.metric] for iso in isos]
#             plt.plot(isos, ssimscore, label=component, marker=markers[component])
#             plt.title(image)
#             if args.metric == 'ssim':
#                 plt.ylabel('SSIM score')
#             else:
#                 plt.ylabel('MSE loss')
#             plt.xlabel('ISO value')
#         except KeyError as err:
#             print(err)
#             continue
#     plt.grid()
#     plt.legend()
#     if not args.noshow:
#         plt.show()
# TODO use json to handle nested dicts
# if not args.nojson:
#     with open('data.json', 'w') as f:
#         json.dump(data, f)
