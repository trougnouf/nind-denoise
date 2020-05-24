# train a (c)GAN or standard CNN network for denoising (ie with the NIND)
# should replace run_nn.py (TODO: no discriminator initialization if its weight is 0, add dataset options s.a. compression)
from __future__ import print_function
import argparse
import os
from dataset_torch_3 import DenoisingDataset
import time
import datetime
import sys

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import statistics
from nn_common import default_values, Generator, Discriminator, Printer, get_crop_boundaries, get_weights

# Training settings

parser = argparse.ArgumentParser(description='(c)GAN trainer for mthesis-denoise')
parser.add_argument('--batch_size', type=int, default=18, help='Training batch size')
parser.add_argument('--time_limit', type=int, default=172800, help='Time limit (ends training)')
parser.add_argument('--g_activation', type=str, default='PReLU', help='Final activation function for generator')
parser.add_argument('--g_funit', type=int, default=32, help='Filter unit size for generator')
parser.add_argument('--d_activation', type=str, default='PReLU', help='Final activation function for discriminator')
parser.add_argument('--d2_activation', type=str, default='PReLU', help='Final activation function for discriminator')
parser.add_argument('--d_funit', type=int, default=32, help='Filter unit size for discriminator')
parser.add_argument('--d2_funit', type=int, default=32, help='Filter unit size for discriminator')
parser.add_argument('--d_model_path', help='Discriminator pretrained model path (.pth for model, .pt for dictionary)')
parser.add_argument('--d2_model_path', help='Discriminator pretrained model path (.pth for model, .pt for dictionary)')
parser.add_argument('--g_model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
parser.add_argument('--beta1', type=float, default=default_values['beta1'], help='beta1 for adam. default=%f'%default_values['beta1'])
parser.add_argument('--d_loss_function', type=str, default='MSE', help='Discriminator loss function')
parser.add_argument('--d2_loss_function', type=str, default='MSE', help='Discriminator loss function')
parser.add_argument('--d_lr', type=float, default=default_values['lr'], help='Initial learning rate for adam (discriminator)')
parser.add_argument('--d2_lr', type=float, default=default_values['lr'], help='Initial learning rate for adam (discriminator)')
parser.add_argument('--g_lr', type=float, default=default_values['lr'], help='Initial learning rate for adam (generator)')
parser.add_argument('--weight_SSIM', type=float, help='Weight on SSIM term in objective')
parser.add_argument('--weight_L1', type=float, help='Weight on L1 term in objective')
parser.add_argument('--weight_D1', type=float, help='Weight on Discriminator 1 term in objective')
parser.add_argument('--weight_D2', type=float, help='Weight on Discriminator 2 term in objective')
parser.add_argument('--test_reserve', nargs='*', help='Space separated list of image sets to be reserved for testing')
parser.add_argument('--train_data', nargs='*', help="(space-separated) Path(s) to the pre-cropped training data (default: %s)"%(" ".join(default_values['train_data'])))
parser.add_argument('--debug_options', nargs='*', help="(space-separated) Debug options (available: discriminator_input)")
parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3, -1 for CPU)')
parser.add_argument('--d_network', type=str, default=default_values['d_network'], help='Discriminator network (default: %s)'%default_values['d_network'])
parser.add_argument('--d2_network', type=str, default=default_values['d2_network'], help='Discriminator2 network (default: %s)'%default_values['d2_network'])
parser.add_argument('--g_network', type=str, default=default_values['g_network'], help='Generator network (default: %s)'%default_values['g_network'])
parser.add_argument('--threads', type=int, default=6, help='Number of threads for data loader to use')
parser.add_argument('--min_lr', type=float, default=0.00000005, help='Minimum learning rate (ends training)')
parser.add_argument('--not_conditional', action='store_true', help='Regular GAN instead of cGAN')
parser.add_argument('--not_conditional_2', action='store_true', help='Regular GAN instead of cGAN')
parser.add_argument('--epochs', type=int, default=9001, help='Number of epochs (ends training)')
parser.add_argument('--compute_SSIM_anyway', action='store_true', help='Compute and display SSIM loss even if not used')
parser.add_argument('--freeze_generator', action='store_true', help='Freeze generator until discriminator is useful')
parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch (cosmetics)')
parser.add_argument('--discriminator_advantage', type=float, default=0.0, help='Desired discriminator correct prediction ratio is 0.5+advantage')
parser.add_argument('--discriminator2_advantage', type=float, default=0.0, help='Desired discriminator correct prediction ratio is 0.5+advantage')
parser.add_argument('--patience', type=int, default=default_values['patience'], help='Number of epochs without improvements before scheduler updates learning rate')

args = parser.parse_args()

# process some arguments

if args.test_reserve is None or args.test_reserve == []:
    test_reserve = default_values['test_reserve']
else:
    test_reserve = args.test_reserve
if args.train_data is None or args.train_data == []:
    train_data = default_values['train_data']
else:
    train_data = args.train_data
if args.cuda_device >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda:"+str(args.cuda_device))
else:
    device = torch.device('cpu')
if args.debug_options is None or args.debug_options == []:
    debug_options = []
else:
    debug_options = args.debug_options

weights = get_weights(args)
use_D = weights['D1'] > 0
use_D2 = weights['D2'] > 0


def crop_batch(batch, boundaries):
    return batch[:, :, boundaries[0]:boundaries[1], boundaries[0]:boundaries[1]]


cudnn.benchmark = True

torch.manual_seed(123)
torch.cuda.manual_seed(123)

expname = (datetime.datetime.now().isoformat()[:-10]+'_'+'_'.join(sys.argv).replace('/','-'))[0:255]
model_dir = os.path.join('models', expname)
txt_path = os.path.join('results', 'train', expname)
os.makedirs(model_dir, exist_ok=True)

frozen_generator = args.freeze_generator

p = Printer(file_path=os.path.join(txt_path))

p.print(args)
p.print("cmd: python3 "+" ".join(sys.argv))

DDataset = DenoisingDataset(train_data, test_reserve=test_reserve)
data_loader = DataLoader(dataset=DDataset, num_workers=args.threads, drop_last=True,
                         batch_size=args.batch_size, shuffle=True)

if use_D:
    discriminator = Discriminator(network=args.d_network, model_path=args.d_model_path,
                                  device=device, loss_function=args.d_loss_function,
                                  activation=args.d_activation, funit=args.d_funit,
                                  beta1=args.beta1, lr=args.d_lr,
                                  not_conditional=args.not_conditional, printer=p,
                                  patience=args.patience, debug_options=debug_options)
if use_D2:
    discriminator2 = Discriminator(network=args.d2_network, model_path=args.d2_model_path,
                                  device=device, loss_function=args.d2_loss_function,
                                  activation=args.d2_activation, funit=args.d2_funit,
                                  beta1=args.beta1, lr=args.d2_lr,
                                  not_conditional=args.not_conditional_2, printer=p,
                                  patience=args.patience, debug_options=debug_options)
generator = Generator(network=args.g_network, model_path=args.g_model_path, device=device,weights=weights,
                      activation=args.g_activation, funit=args.g_funit, beta1=args.beta1,
                      lr=args.g_lr, printer=p, compute_SSIM_anyway=args.compute_SSIM_anyway,
                      patience=args.patience, debug_options=debug_options)

crop_boundaries = get_crop_boundaries(DDataset.cs, DDataset.ucs, network=args.g_network, discriminator=args.d_network)


discriminator_predictions = None
generator_learning_rate = args.g_lr
discriminator_learning_rate = args.d_lr

start_time = time.time()
for epoch in range(args.start_epoch, args.epochs):
    loss_D_list = []
    loss_D2_list = []
    loss_G_list = []
    loss_G_SSIM_list = []
    epoch_start_time = time.time()
    for iteration, batch in enumerate(data_loader, 1):
        iteration_summary = 'Epoch %u batch %u/%u: ' % (epoch, iteration, len(data_loader))
        clean_batch_cropped = crop_batch(batch[0].to(device), crop_boundaries)
        noisy_batch = batch[1].to(device)
        noisy_batch_cropped = crop_batch(noisy_batch, crop_boundaries)
        generated_batch = generator.denoise_batch(noisy_batch)
        generated_batch_cropped = crop_batch(generated_batch, crop_boundaries)
        # train discriminator based on its previous performance
        discriminator_learns = (use_D and (discriminator.get_loss()+args.discriminator_advantage) > random.random()) or frozen_generator
        if discriminator_learns:
            discriminator.learn(noisy_batch_cropped=noisy_batch_cropped,
                                generated_batch_cropped=generated_batch_cropped,
                                clean_batch_cropped=clean_batch_cropped)
            loss_D_list.append(discriminator.get_loss())
            iteration_summary += 'loss D: %f (%s)' % (discriminator.get_loss(), discriminator.get_predictions_range())
        # train discriminator2 based on its previous performance
        discriminator2_learns = (use_D2 and (discriminator2.get_loss()+args.discriminator2_advantage) > random.random()) or (use_D2 and frozen_generator)
        if discriminator2_learns:
            discriminator2.learn(noisy_batch_cropped=noisy_batch_cropped,
                                generated_batch_cropped=generated_batch_cropped,
                                clean_batch_cropped=clean_batch_cropped)
            loss_D2_list.append(discriminator2.get_loss())
            if discriminator_learns:
                iteration_summary += ', '
            while len(iteration_summary) < 90:
                iteration_summary += ' '
            iteration_summary += 'loss D2: %f (%s)' % (discriminator2.get_loss(), discriminator2.get_predictions_range())
        # train generator if discriminator didn't learn or discriminator is somewhat useful
        generator_learns = not frozen_generator and (
            (not discriminator_learns and not discriminator2_learns)
            or (discriminator_learns and discriminator2_learns and (discriminator2.get_loss()+args.discriminator2_advantage+discriminator.get_loss()+args.discriminator_advantage)/2 < random.random())
            or (discriminator_learns and (not discriminator2_learns) and discriminator.get_loss()+args.discriminator_advantage < random.random())
            or (discriminator2_learns and (not discriminator_learns) and discriminator2.get_loss()+args.discriminator2_advantage < random.random())
            )
        if generator_learns:
            if discriminator_learns or discriminator2_learns:
                iteration_summary += ', '
            pregenres_space = 1 if not use_D else 125
            pregenres_space = 160 if use_D2 else pregenres_space
            while len(iteration_summary) < pregenres_space:
                iteration_summary += ' '
            if use_D:
                discriminator_predictions = discriminator.discriminate_batch(
                    generated_batch_cropped=generated_batch_cropped,
                    noisy_batch_cropped=noisy_batch_cropped)
            if use_D2:
                discriminator2_predictions = discriminator2.discriminate_batch(
                    generated_batch_cropped=generated_batch_cropped,
                    noisy_batch_cropped=noisy_batch_cropped)
            else:
                discriminator2_predictions = None
            generator.learn(generated_batch_cropped=generated_batch_cropped,
                            clean_batch_cropped=clean_batch_cropped,
                            discriminator_predictions=discriminator_predictions,
                            discriminator2_predictions=discriminator2_predictions)
            loss_G_list.append(generator.get_loss()['weighted'])
            loss_G_SSIM_list.append(generator.get_loss()['SSIM'])
            iteration_summary += 'loss G: %s' % generator.get_loss(pretty_printed=True)
        else:
            generator.zero_grad()
            if frozen_generator:
                frozen_generator = discriminator.get_loss() > 0.33 and ((not use_D2) or discriminator2.get_loss() > 0.33)
        p.print(iteration_summary)

    p.print("Epoch %u summary:" % epoch)
    p.print("Time elapsed (s): %u (epoch), %u (total)" % (time.time()-epoch_start_time,
                                                          time.time()-start_time))
    p.print("Generator:")
    if len(loss_G_SSIM_list) > 0:
        p.print("Average SSIM loss: %f" % statistics.mean(loss_G_SSIM_list))
    if len(loss_G_list) > 0:
        average_g_weighted_loss = statistics.mean(loss_G_list)
        p.print("Average weighted loss: %f" % average_g_weighted_loss)
        generator_learning_rate = generator.update_learning_rate(average_g_weighted_loss)
    else:
        p.print("Generator learned nothing")
    if use_D:
        p.print("Discriminator:")
        if len(loss_D_list) > 0:
            average_d_loss = statistics.mean(loss_D_list)
            p.print("Average normalized loss: %f" % (average_d_loss))
            discriminator_learning_rate = discriminator.update_learning_rate(average_d_loss)
            discriminator.save_model(model_dir, epoch, 'discriminator')
    if use_D2:
        p.print("Discriminator2:")
        if len(loss_D2_list) > 0:
            average_d2_loss = statistics.mean(loss_D2_list)
            p.print("Average normalized loss: %f" % (average_d2_loss))
            discriminator2_learning_rate = discriminator2.update_learning_rate(average_d2_loss)
            discriminator2.save_model(model_dir, epoch, 'discriminator2')
    if not frozen_generator:
        generator.save_model(model_dir, epoch, 'generator')
    if args.time_limit < time.time() - start_time:
        p.print("Time is up")
        exit(0)
    if (discriminator_learning_rate < args.min_lr or not use_D) and generator_learning_rate < args.min_lr:
        p.print("Minimum learning rate reached")
        exit(0)

