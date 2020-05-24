import torch
import torch.nn as nn
import torch.optim as optim
from lib import pytorch_ssim
from torch.optim import lr_scheduler
import math
import torchvision
#from networks.p2p_networks import define_D
import os
import time
from networks.Hul import Hulb128Net, Hul112Disc, Hulf112Disc
from networks.ThirdPartyNets import PatchGAN, UNet
from networks.UtNet import UtNet, UtdNet

default_values = {
    'g_network': 'Hulb128Net',
    'd_network': 'Hul112Disc',
    'd2_network': 'PatchGAN',
    'train_data': ['datasets/train/NIND_128_112'],
    'beta1': 0.5,
    'weight_SSIM': 0.2,
    'weight_L1': 0.05,
    'lr': 0.0003,
    'test_reserve': ['ursulines-red stefantiek', 'ursulines-building', 'MuseeL-Bobo', 'CourtineDeVillersDebris', 'C500D', 'Pen-pile'],
    'patience': 3,
    'weights': {
        'SSIM': 0.2,
        'L1': 0.05,
        'D1': 0.75,
        'D2': 0
    }
}


class Model:
    def __init__(self, save_dict=True, device='cuda:0', printer=None, debug_options=[]):
        if printer is None:
            self.print = print
        else:
            self.print = printer.print
        self.loss = 1
        self.save_dict = save_dict
        self.device = device
        self.debug_options=debug_options

    def save_model(self, model_dir, epoch, name):
        save_path = os.path.join(model_dir, '%s_%u.pt' % (name, epoch))
        if self.save_dict:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model, save_path+'h')

    @staticmethod
    def complete_path(path, keyword=''):
        def find_highest(paths, keyword):
            best = [None, 0]
            for path in paths:
                curval = int(path.split('_')[-1].split('.')[0])
                if curval > best[1] and keyword in path:
                    best = [path, curval]
            return best[0]
        if os.path.isfile(path):
            return path
        elif os.path.isdir(path):
            return os.path.join(path, find_highest(os.listdir(path), keyword))
        elif os.path.isdir(os.path.join('models', path)):
            return Model.complete_path(os.path.join('models', path), keyword)
        else:
            print("Model path not found: %s"%path)
            exit(0)

    @staticmethod
    def instantiate_model(model_path=None, network=None, device='cuda:0', strparameters=None, pfun=print, keyword='', **parameters):
        model = None
        if strparameters is not None and strparameters != "":
            parameters.update(dict([parameter.split('=') for parameter in strparameters.split(',')]))
        if model_path is not None:
            path = Model.complete_path(model_path, keyword)
            if path.endswith('.pth'):
                model = torch.load(path, map_location=device)
            elif path.endswith('pt'):
                assert network is not None
                model = globals()[network](**parameters)
                model.load_state_dict(torch.load(path, map_location=device))
            else:
                pfun('Error: unable to load invalid model path: %s'%path)
                exit(1)
        else:
            model = globals()[network](**parameters)
        return model.to(device)

    def update_learning_rate(self, avg_loss):
        self.scheduler.step(metrics=avg_loss)
        lr = self.optimizer.param_groups[0]['lr']
        self.print('Learning rate: %f' % lr)
        return lr


class Generator(Model):
    def __init__(self, network = default_values['g_network'], model_path = None,
                 device = 'cuda:0', weights=default_values['weights'], activation='PReLU', funit=32,
                 beta1=default_values['beta1'], lr=default_values['lr'], printer=None, compute_SSIM_anyway=False,
                 save_dict=True, patience=default_values['patience'], debug_options=[]):
        Model.__init__(self, save_dict, device, printer, debug_options=[])
        self.weights = weights
        if weights['SSIM'] > 0 or compute_SSIM_anyway:
            self.criterion_SSIM = pytorch_ssim.SSIM().to(device)
        if weights['L1'] > 0:
            self.criterion_L1 = nn.L1Loss().to(device)
        if weights['D1'] > 0:
            self.criterion_D = nn.MSELoss().to(device)
        if weights['D2'] > 0:
            self.criterion_D2 = nn.MSELoss().to(device)
        self.model = self.instantiate_model(model_path=model_path, network=network, pfun=self.print, device=device, funit=funit, keyword='generator')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=patience)
        self.device = device
        self.loss = {'SSIM': 1, 'L1': 1, 'D': 1, 'weighted': 1}
        self.compute_SSIM_anyway = compute_SSIM_anyway

    def get_loss(self, pretty_printed=False):
        if pretty_printed:
            return ", ".join(["%s: %.3f"%(key, val) if val != 1 else 'NA' for key,val in self.loss.items()])
        return self.loss

    def denoise_batch(self, noisy_batch):
        return self.model(noisy_batch)

    def learn(self, generated_batch_cropped, clean_batch_cropped, discriminator_predictions=None, discriminator2_predictions=None):
        if self.weights['SSIM'] > 0 or self.compute_SSIM_anyway:
            loss_SSIM = self.criterion_SSIM(generated_batch_cropped, clean_batch_cropped)
            loss_SSIM = 1-loss_SSIM
            self.loss['SSIM'] = loss_SSIM.item()
        if self.weights['SSIM'] == 0:
            loss_SSIM = torch.zeros(1).to(self.device)
        if self.weights['L1'] > 0:
            loss_L1 = self.criterion_L1(generated_batch_cropped, clean_batch_cropped)
            self.loss['L1'] = loss_L1.item()
        else:
            loss_L1 = torch.zeros(1).to(self.device)
        if self.weights['D1'] > 0:
            loss_D = self.criterion_D(discriminator_predictions,
                                      gen_target_probabilities(True, discriminator_predictions.shape,
                                                               device=self.device, noisy=False))
            self.loss['D'] = math.sqrt(loss_D.item())
        else:
            loss_D = torch.zeros(1).to(self.device)
        if self.weights['D2'] > 0:
            loss_D2 = self.criterion_D2(discriminator2_predictions,
                                      gen_target_probabilities(True, discriminator2_predictions.shape,
                                                               device=self.device, noisy=False))
            self.loss['D2'] = math.sqrt(loss_D2.item())
        else:
            loss_D2 = torch.zeros(1).to(self.device)
        loss = loss_SSIM * self.weights['SSIM'] + loss_L1 * self.weights['L1'] + loss_D * self.weights['D1'] + loss_D2 * self.weights['D2']
        self.loss['weighted'] = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()


class Discriminator(Model):
    def __init__(self, network='Hul112Disc', weights_dict_path=None,
                 model_path=None, device='cuda:0', loss_function='MSE',
                 activation='PReLU', funit=32, beta1=default_values['beta1'],
                 lr = default_values['lr'], not_conditional = False, printer=None, save_dict=True,
                 patience=default_values['patience'], debug_options=[]):
        Model.__init__(self, save_dict, device, printer, debug_options)
        self.device = device
        self.loss = 1
        self.loss_function = loss_function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss().to(device)
        if not_conditional:
            input_channels = 3
        else:
            input_channels = 6
        self.model = self.instantiate_model(model_path=model_path, network=network, pfun=self.print, device=device, funit=funit, input_channels = input_channels)
            #elif network == 'PatchGAN':
            #    self.model = net_d = define_D(input_channels, 2*funit, 'basic', gpu_id=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
        self.conditional = not not_conditional
        self.predictions_range = None
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=patience)

    def get_loss(self):
        return self.loss

    def get_predictions_range(self):
        return 'range (r-r+f-f+): '+str(self.predictions_range)

    def update_loss(self, loss_fake, loss_real):
        if self.loss_function == 'MSE':
            self.loss = (math.sqrt(loss_fake)+math.sqrt(loss_real))/2
        else:
            self.print('Error: loss function not implemented: %s'%(self.loss_function))

    def discriminate_batch(self, generated_batch_cropped, noisy_batch_cropped=None):
        if self.conditional:
            fake_batch = torch.cat([noisy_batch_cropped, generated_batch_cropped], 1)
        else:
            fake_batch = generated_batch_cropped
        return self.model(fake_batch)

    def learn(self, generated_batch_cropped, clean_batch_cropped, noisy_batch_cropped=None):
        self.optimizer.zero_grad()
        if self.conditional:
            real_batch = torch.cat([noisy_batch_cropped, clean_batch_cropped], 1)
            fake_batch = torch.cat([noisy_batch_cropped, generated_batch_cropped.detach()], 1)
        else:
            real_batch = clean_batch_cropped
            fake_batch = generated_batch_cropped.detach()
        if 'discriminator_input' in self.debug_options:
            os.makedirs('dbg', exist_ok=True)
            batch_savename = os.path.join('dbg',str(time.time()))
            if self.conditional:
                real_batch_detached = torch.cat([real_batch[:,:3,:,:], real_batch[:,3:,:,:]],0).detach().cpu()
                fake_batch_detached = torch.cat([fake_batch[:,:3,:,:], fake_batch[:,3:,:,:]],0).detach().cpu()
            else:
                real_batch_detached = real_batch.detach().cpu()
                fake_batch_detached = fake_batch.detach().cpu()
            torchvision.utils.save_image(real_batch_detached, batch_savename+'_real.png')
            torchvision.utils.save_image(fake_batch_detached, batch_savename+'_fake.png')

        pred_real = self.model(real_batch)
        loss_real = self.criterion(pred_real,
                                   gen_target_probabilities(True, pred_real.shape,
                                                            device=self.device, noisy=True))
        loss_real_detached = loss_real.item()
        loss_real.backward()
        pred_fake = self.model(fake_batch)
        loss_fake = self.criterion(pred_fake,
                                   gen_target_probabilities(False, pred_fake.shape,
                                                            device=self.device,
                                                            noisy=self.loss < 0.25))
        loss_fake_detached = loss_fake.item()
        loss_fake.backward()
        try:
            self.predictions_range = ", ".join(["{:.2}".format(float(i)) for i in (pred_real.min(), pred_real.max(), pred_fake.min(), pred_fake.max())])
        except:
            self.predictions_range = '(not implemented)'
        self.update_loss(loss_fake_detached, loss_real_detached)
        self.optimizer.step()


class Printer:
    def __init__(self, tostdout=True, tofile=True, file_path='log'):
        self.tostdout = tostdout
        self.tofile = tofile
        self.file_path = file_path

    def print(self, msg):
        if self.tostdout:
            print(msg)
        if self.tofile:
            try:
                with open(self.file_path, 'a') as f:
                    f.write(str(msg)+'\n')
            except Exception as e:
                print('Warning: could not write to log: %s'%e)

def get_crop_boundaries(cs, ucs, network=None, discriminator=None):
    # if '112' in discriminator:
    #     loss_crop_lb = int((cs-112)/2)
    #     loss_crop_up = cs - loss_crop_lb
    #     assert (loss_crop_up - loss_crop_lb) == 112
    if network == 'UNet':    # UNet requires huge borders
        loss_crop_lb = int(cs/8)
        loss_crop_up = cs-loss_crop_lb
    elif network == 'UtNet' or network == 'UtdNet': # mirrored in-net
        loss_crop_lb = max(1,int(cs/32))
        loss_crop_up = cs-loss_crop_lb
    else:
        loss_crop_lb = int(cs/16)
        loss_crop_up = cs-loss_crop_lb
    print('Using %s as bounds'%(str((loss_crop_lb, loss_crop_up))))
    assert (loss_crop_up - loss_crop_lb) <= ucs
    return loss_crop_lb, loss_crop_up

def gen_target_probabilities(target_real, target_probabilities_shape, device=None, invert_probabilities=False, noisy = True):
    if (target_real and not invert_probabilities) or (not target_real and invert_probabilities):
        if noisy:
            res = 19/20+torch.rand(target_probabilities_shape)/20
        else:
            res = torch.ones(target_probabilities_shape)
    else:
        if noisy:
            res = torch.rand(target_probabilities_shape)/20
        else:
            res = torch.zeros(target_probabilities_shape)
    if device is None:
        return res
    else:
        return res.to(device)


def get_weights(args):
    total = 0
    weights = {'L1': None, 'SSIM': None, 'D1': None, 'D2': None}
    if args.weight_SSIM:
        weights['SSIM'] = args.weight_SSIM
        total += args.weight_SSIM
    if args.weight_L1:
        weights['L1'] = args.weight_L1
        total += args.weight_L1
    if args.weight_D1:
        weights['D1'] = args.weight_D1
        total += args.weight_D1
    if args.weight_D2:
        weights['D2'] = args.weight_D2
        total += args.weight_D2
    if not weights['SSIM']:
        weights['SSIM'] = default_values['weights']['SSIM']
        total += weights['SSIM']
    if not weights['L1']:
        weights['L1'] = min(default_values['weights']['L1'], 1-total)
        total += weights['L1']
    if not weights['D1']:
        weights['D1'] = min(default_values['weights']['D1'], 1-total)
        total += weights['D1']
    if not weights['D2']:
        weights['D2'] = min(default_values['weights']['D2'], 1-total)
        total += weights['D2']
    assert sum(weights.values()) == 1
    print('Loss weights: '+str(weights))
    return weights

