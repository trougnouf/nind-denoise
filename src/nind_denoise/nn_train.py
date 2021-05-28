'''
Train a denoising neural network with the NIND (or any dataset of clean-noisy images)
Optionally uses up to 2 (c)GAN discriminators. These can be turned on with --weight_D{1,2} > 0

egrun:
first check that everything works with a dummy run:
python3 nn_train.py --config configs/train_conf_unet.yaml --debug_options output_val_images output_test_images keep_all_output_images short_run --test_interval 0 --epochs 6
then launch the training with (eg):
python3 nn_train.py --config configs/train_conf_unet.yaml --debug_options output_val_images --test_interval 0 --epochs 600

Note that the discriminators are experimental and currently unmaintained; it is unknown whether they
will function with the current state of the source code.


'''
# This replaces run_nn.py
# TODO functions

# TODO json saver, keep best models, add data

# TODO full test every full_test_interval (done but too memory hungry at 32GB, could use cropping)

# TODO check on lr scheduler

import configargparse
import os
import time
import datetime
import sys
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import statistics
import collections
#from nn_common import default_values as DEFAULT
import yaml
import torchvision
import shutil
import sys
sys.path.append('..')
from nind_denoise import dataset_torch_3
from nind_denoise import nn_common
from common.libs import pt_ops
from common.libs import json_saver

DEFAULT_CONFIG_FPATH = os.path.join('configs', 'train_conf_defaults.yaml')

# Create validation set and test it now rather than waiting until the end of the first epoch
def validate_generator(model, validation_set, output_to_dir=None):
    '''
    currently doing one image at a time, limited to the same crop size as training.
    TODO check if too slow. possibly use other sizes. (If used once per epoch then shouldn't matter.)
    '''
    model.eval()
    losses = []
    for i, (clean, noisy) in enumerate(validation_set):
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        denoised = model.denoise_batch(noisy)
        denoised_fs = denoised.clone().detach().cpu()
        model.compute_loss(pt_ops.pt_crop_batch(denoised, args.loss_cs),
                           pt_ops.pt_crop_batch(clean, args.loss_cs))
        if output_to_dir is not None:
            os.makedirs(output_to_dir, exist_ok=True)
            # This saves as 8-bit tiff (we don't really care for the preview) and includes borders
            torchvision.utils.save_image(denoised_fs, os.path.join(output_to_dir, str(i)+'.tif'))
        losses.append(model.get_loss(component='weighted'))
    avgloss = statistics.mean(losses)
    model.train()
    return avgloss

def test_generator(model, test_set, output_to_dir=None):
    '''
    This test moves the model to CPU and tests on whole images. It is meant to run extremely slowly
    and should not be used frequently.
    FIXME: add padding to the lossf
    '''
    model.eval()
    model.tocpu()
    losses = []
    for i, (clean, noisy) in enumerate(test_set):
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        denoised = model.denoise_batch(noisy)
        model.compute_loss(denoised, clean)
        if output_to_dir is not None:
            os.makedirs(output_to_dir, exist_ok=True)
            torchvision.utils.save_image(denoised, os.path.join(output_to_dir, str(i)+'.tif'))
        losses.append(model.get_loss(component='weighted'))
    avgloss = statistics.mean(losses)
    model.todevice()
    model.train()
    return avgloss

def delete_outperformed_models(dpath: str,  keepers: set, model_t: str = 'generator', keep_all_output_images=False):
    '''
    remove models whose epoch is not in the keepers set
    '''
    removed = list()
    for fn in os.listdir(dpath):
        fpath = os.path.join(dpath, fn)
        if (fn == 'val' or fn == 'testimages') and not keep_all_output_images:
            for subdir in os.listdir(fpath):
                if int(subdir) not in keepers:
                    val_dpath = os.path.join(fpath, subdir)
                    shutil.rmtree(val_dpath)
                    removed.append(val_dpath)
            continue
        if not fn.startswith(f'{model_t}_'):
            continue
        epoch = int(fn.split('_')[1].split('.')[0])
        if epoch not in keepers:
            
            os.remove(fpath)
            removed.append(fpath)
    return removed
            
if __name__ == '__main__':
    
    # Training settings
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
        nn_common.COMMON_CONFIG_FPATH, DEFAULT_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='(yaml) config file path')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--time_limit', type=int, help='Time limit in seconds (ends training)')
    parser.add_argument('--g_activation', type=str, default='PReLU', help='Final activation function for generator')
    parser.add_argument('--g_funit', type=int, default=32, help='Filter unit size for generator')
    parser.add_argument('--g_model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
    parser.add_argument('--models_dpath', help='Directory where all models are saved')
    parser.add_argument('--beta1', type=float, help='beta1 for adam')
    parser.add_argument('--g_lr', type=float, help='Initial learning rate for adam (generator)')
    parser.add_argument('--weight_SSIM', type=float, help='Weight on SSIM term in objective')
    parser.add_argument('--weight_MSSSIM', type=float, help='Weight on MSSSIM term in objective')
    parser.add_argument('--weight_L1', type=float, help='Weight on L1 term in objective')
    parser.add_argument('--weight_MSE', type=float, help='Weight on L1 term in objective')
    parser.add_argument('--test_reserve', nargs='*', required=True, help='Space separated list of image sets to be reserved for testing, or yaml file path containing a list. Set to "0" to use all available data.')
    parser.add_argument('--train_data', nargs='*', help="(space-separated) Path(s) to the pre-cropped training data")
    parser.add_argument('--cs', '--crop_size', type=int, help='Crop size fed to NN. default: no additional cropping')
    parser.add_argument('--min_crop_size', type=int, help='Minimum crop size. Dataset will be checked if this value is set.')
    parser.add_argument('--loss_cs', '--loss_crop_size', type=int, help='Center crop size used in loss function. default: use stride size from dataset directory name')
    parser.add_argument('--debug_options', '--debug', nargs='*', default=[], help=f"(space-separated) Debug options (available: {nn_common.DebugOptions})")
    parser.add_argument('--cuda_device', '--device', default=0, type=int, help='Device number (default: 0, typically 0-3, -1 for CPU)')
    parser.add_argument('--g_network', type=str, help='Generator network')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for data loader to use')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate (ends training)')
    parser.add_argument('--epochs', type=int, default=9001, help='Number of epochs (ends training)')
    parser.add_argument('--compute_SSIM_anyway', action='store_true', help='Compute and display SSIM loss even if not used')
    parser.add_argument('--freeze_generator', action='store_true', help='Freeze generator until discriminator is useful')
    parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch (cosmetics)')
    parser.add_argument('--patience', type=int, help='Number of epochs without improvements before scheduler updates learning rate')
    parser.add_argument('--reduce_lr_factor', type=float, help='LR is multiplied by this factor when model performs poorly for <patience> epochs')
    parser.add_argument('--validation_interval', help='Validation interval in # of epochs. Affects learning rate update and helps to keep the best model in the end. 0 = no validation, default=1', default=1, type=int)
    parser.add_argument('--test_interval', default=0, help='Test interval in # of epochs. Performed on CPU with whole images (long and WARNING: uses enormous amount of RAM; keep off without >= 64 GB). 0 = no such tests', type=int)
    parser.add_argument('--orig_data', help='Location of the originally downloaded train data (before cropping); used when test_interval is set')
    parser.add_argument('--validation_set_yaml', help=f'Yaml file containing a list of clean/noisy images used for validation.')
    parser.add_argument('--exp_mult_min', type=float, help='Minimum exposure multiplicator (data augmentation)')
    parser.add_argument('--exp_mult_max', type=float, help='Maximum exposure multiplicator (data augmentation)')
    # discriminator stuff
    parser.add_argument('--d_activation', type=str, default='PReLU', help='Final activation function for discriminator')
    parser.add_argument('--d2_activation', type=str, default='PReLU', help='Final activation function for discriminator')
    parser.add_argument('--d_funit', type=int, default=32, help='Filter unit size for discriminator')
    parser.add_argument('--d2_funit', type=int, default=32, help='Filter unit size for discriminator')
    parser.add_argument('--d_model_path', help='Discriminator pretrained model path (.pth for model, .pt for dictionary)')
    parser.add_argument('--d2_model_path', help='Discriminator pretrained model path (.pth for model, .pt for dictionary)')
    parser.add_argument('--d_loss_function', type=str, default='MSE', help='Discriminator loss function')
    parser.add_argument('--d2_loss_function', type=str, default='MSE', help='Discriminator loss function')
    parser.add_argument('--d_lr', type=float, help='Initial learning rate for adam (discriminator)')
    parser.add_argument('--d2_lr', type=float, help='Initial learning rate for adam (discriminator)')
    parser.add_argument('--weight_D1', type=float, help='Weight on Discriminator 1 term in objective')
    parser.add_argument('--weight_D2', type=float, help='Weight on Discriminator 2 term in objective')
    parser.add_argument('--d_network', type=str, help='Discriminator network')
    parser.add_argument('--d2_network', type=str, help='Discriminator2 network')
    parser.add_argument('--not_conditional', action='store_true', help='Regular GAN instead of cGAN')
    parser.add_argument('--not_conditional_2', action='store_true', help='Regular GAN instead of cGAN')
    parser.add_argument('--discriminator_advantage', type=float, default=0.0, help='Desired discriminator correct prediction ratio is 0.5+advantage')
    parser.add_argument('--discriminator2_advantage', type=float, default=0.0, help='Desired discriminator correct prediction ratio is 0.5+advantage')
    
    args = parser.parse_args()
        
    # process some arguments

    if args.cuda_device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = torch.device("cuda:"+str(args.cuda_device))
        cudnn.benchmark = True
        torch.cuda.manual_seed(123)
    else:
        device = torch.device('cpu')
    torch.manual_seed(123)
    
    debug_options = [nn_common.DebugOptions(opt) for opt in args.debug_options]

    weights = nn_common.get_weights(args)
    use_D = weights['D1'] > 0
    use_D2 = weights['D2'] > 0
    
    expname = (datetime.datetime.now().isoformat()[:-10]+'_'+'_'.join(sys.argv).replace('/','-'))[0:255]
    model_dir = os.path.join(args.models_dpath, expname)
    os.makedirs(model_dir, exist_ok=True)
    txt_path = os.path.join(model_dir, 'train.log')
    jsonsaver = json_saver.JSONSaver(os.path.join(model_dir, 'trainres.json'), step_type='epoch')
    
    
    frozen_generator = args.freeze_generator
    
    p = nn_common.Printer(file_path=os.path.join(txt_path))
    
    p.print(args)
    p.print("cmd: python3 "+" ".join(sys.argv))
    
    args.test_reserve = nn_common.get_test_reserve_list(args.test_reserve)
    p.print(f'test_reserve: {args.test_reserve}')
    
    # Train data
    if (args.min_crop_size is None or args.min_crop_size == 0) and nn_common.DebugOptions.CHECK_DATASET in debug_options:
        args.min_crop_size = args.cs
    DDataset = dataset_torch_3.DenoisingDataset(args.train_data, test_reserve=args.test_reserve,
                                                cs=args.cs, min_crop_size=args.min_crop_size,
                                                exp_mult_min=args.exp_mult_min,
                                                exp_mult_max=args.exp_mult_max)
    if args.do_quality_check:
        exit('Quality check done')
    if args.loss_cs is None:
        args.loss_cs = DDataset.min_crop_size
        assert args.loss_cs is not None
    if args.cs is None:
        args.cs = DDataset.cs
    if nn_common.DebugOptions.SHORT_RUN in debug_options:
        DDataset.dataset = DDataset.dataset[:3*args.batch_size]
    data_loader = DataLoader(dataset=DDataset, num_workers=args.threads, drop_last=True,
                             batch_size=args.batch_size, shuffle=True)
    
    # init models
    
    if use_D:
        discriminator = nn_common.Discriminator(network=args.d_network, model_path=args.d_model_path,
                                      device=device, loss_function=args.d_loss_function,
                                      activation=args.d_activation, funit=args.d_funit,
                                      beta1=args.beta1, lr=args.d_lr,
                                      not_conditional=args.not_conditional, printer=p,
                                      patience=args.patience, debug_options=debug_options,
                                      models_dpath=args.models_dpath,
                                      reduce_lr_factor=args.reduce_lr_factor)
    if use_D2:
        discriminator2 = nn_common.Discriminator(network=args.d2_network, model_path=args.d2_model_path,
                                      device=device, loss_function=args.d2_loss_function,
                                      activation=args.d2_activation, funit=args.d2_funit,
                                      beta1=args.beta1, lr=args.d2_lr,
                                      not_conditional=args.not_conditional_2, printer=p,
                                      patience=args.patience, debug_options=debug_options,
                                      models_dpath=args.models_dpath,
                                      reduce_lr_factor=args.reduce_lr_factor)
    generator = nn_common.Generator(network=args.g_network, model_path=args.g_model_path,
                                    device=device,weights=weights, activation=args.g_activation,
                                    funit=args.g_funit, beta1=args.beta1, lr=args.g_lr,
                                    printer=p, compute_SSIM_anyway=args.compute_SSIM_anyway,
                                    patience=args.patience, debug_options=debug_options,
                                    models_dpath=args.models_dpath,
                                    reduce_lr_factor=args.reduce_lr_factor)
    
    discriminator_predictions, discriminator2_predictions = None, None
    generator_learning_rate = args.g_lr
    discriminator_learning_rate = args.d_lr

    # Validation data
    if args.validation_interval > 0:
        validation_set = dataset_torch_3.ValidationDataset(args.validation_set_yaml, device=device, cs=args.cs)
        if nn_common.DebugOptions.OUTPUT_VAL_IMAGES in debug_options:
            get_validation_dpath = lambda epoch: os.path.join(model_dir, 'val', str(epoch))
        else:
            get_validation_dpath = lambda epoch: None
        validation_loss = validate_generator(generator, validation_set, output_to_dir=get_validation_dpath(0))
        jsonsaver.add_res(0, {'validation_loss': validation_loss}, write=True)
        p.print(f'Validation loss: {validation_loss}')
    # Test data
    if args.test_interval > 0:
        test_set = dataset_torch_3.TestDenoiseDataset(data_dpath=args.orig_data,
                                                      sets=args.test_reserve)
        if nn_common.DebugOptions.OUTPUT_TEST_IMAGES in debug_options:
            get_test_dpath = lambda epoch: os.path.join(model_dir, 'testimages', str(epoch))
        else:
            get_test_dpath = lambda epoch: None

    with open(os.path.join(model_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(vars(args), fp)

    start_time = time.time()
    generator_loss_hist = collections.deque(maxlen=args.patience)
    
    # Train        
    for epoch in range(args.start_epoch, args.epochs):
        loss_D_list = []
        loss_D2_list = []
        loss_G_list = []
        loss_G_SSIM_list = []
        epoch_start_time = time.time()
        for iteration, batch in enumerate(data_loader, 1):
            iteration_summary = 'Epoch %u batch %u/%u: ' % (epoch, iteration, len(data_loader))
            clean_batch_cropped = pt_ops.pt_crop_batch(batch[0].to(device), args.loss_cs)
            noisy_batch = batch[1].to(device)
            noisy_batch_cropped = pt_ops.pt_crop_batch(noisy_batch, args.loss_cs)
            generated_batch = generator.denoise_batch(noisy_batch)
            generated_batch_cropped = pt_ops.pt_crop_batch(generated_batch, args.loss_cs)
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
                                discriminators_predictions=[discriminator_predictions,
                                                            discriminator2_predictions])
                loss_G_list.append(generator.get_loss(component='weighted'))
                if generator.weights['SSIM'] > 0 or generator.compute_SSIM_anyway:
                    loss_G_SSIM_list.append(generator.get_loss(component='SSIM'))
                iteration_summary += 'loss G: %s' % generator.get_loss(pretty_printed=True)
            else:
                generator.zero_grad()
                if frozen_generator:
                    frozen_generator = discriminator.get_loss() > 0.33 and ((not use_D2) or discriminator2.get_loss() > 0.33)
            p.print(iteration_summary)
        
        # cleanup previous epochs
        removed = delete_outperformed_models(dpath=model_dir, keepers=jsonsaver.get_best_steps(),
                                   model_t='generator',
                                   keep_all_output_images= nn_common.DebugOptions.KEEP_ALL_OUTPUT_IMAGES in debug_options)
        p.print(f'delete_outperformed_models removed {removed}')
        
        # Do validation
        if args.validation_interval > 0 and epoch%args.validation_interval == 0:
            validation_loss = validate_generator(generator, validation_set,
                                                 output_to_dir=get_validation_dpath(epoch))
            jsonsaver.add_res(epoch, {'validation_loss': validation_loss}, write=False)
            p.print(f'Validation loss: {validation_loss}')
        if args.test_interval > 0 and epoch%args.test_interval == 0:
            test_loss = test_generator(generator, test_set, output_to_dir=get_test_dpath(epoch))
            jsonsaver.add_res(epoch, {'test_loss': test_loss}, write=False)
            
        p.print("Epoch %u summary:" % epoch)
        p.print("Time elapsed (s): %u (epoch), %u (total)" % (time.time()-epoch_start_time,
                                                              time.time()-start_time))
        p.print("Generator:")
        if len(loss_G_SSIM_list) > 0:
            p.print("Average SSIM loss: %f" % statistics.mean(loss_G_SSIM_list))
            jsonsaver.add_res(epoch, {'train_SSIM_loss': statistics.mean(loss_G_SSIM_list)},
                              write=False)
        if len(loss_G_list) > 0:
            average_g_weighted_loss = statistics.mean(loss_G_list)
            p.print("Average weighted loss: %f" % average_g_weighted_loss)
            jsonsaver.add_res(
                epoch, {'train_weighted_loss': average_g_weighted_loss},
                write=False)
            lr_loss = validation_loss if validation_loss is not None else average_g_weighted_loss
            
            if len(generator_loss_hist) > 0 and max(generator_loss_hist) < lr_loss:
                generator_learning_rate = generator.update_learning_rate(args.reduce_lr_factor)
                p.print(f'Generator learning rate updated to {generator_learning_rate} because',
                        f'generator_loss_hist={generator_loss_hist} < lr_loss={lr_loss}')
            generator_loss_hist.append(lr_loss)
            
            jsonsaver.add_res(
                epoch, {'gen_lr': generator_learning_rate},
                write=True)
        else:
            p.print("Generator learned nothing")
        if use_D:
            # TODO add discriminator(s) to jsonsaver and use for cleanup if plan to use those
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
        if args.time_limit and args.time_limit < time.time() - start_time:
            p.print("Time is up")
            exit(0)
        if ((not use_D) or discriminator_learning_rate < args.min_lr) and generator_learning_rate < args.min_lr:
            p.print("Minimum learning rate reached")
            exit(0)
        
        