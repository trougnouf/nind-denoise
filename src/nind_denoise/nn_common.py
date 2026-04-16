import torch
import math
import torchvision
#from networks.p2p_networks import define_D
import os
import time
import yaml
from enum import Enum
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from nind_denoise.lib import pytorch_ssim
from nind_denoise.networks.Hul import Hulb128Net, Hul112Disc, Hulf112Disc
from nind_denoise.networks.ThirdPartyNets import PatchGAN, UNet, MobileNetV3, deeplabv3_resnet101
from nind_denoise.networks.UtNet import UtNet
#from nind_denoise.networks.MIRNet import MIRNet (huge VRAM, bs=2, nope?)

try:
    from common.libs import pt_losses
except ModuleNotFoundError as e:
    print(f'nn_common: import error, (MS-)SSIM loss is not available without the piqa library ({e})')
from common.libs import pt_helpers

COMMON_CONFIG_FPATH = os.path.join('configs', 'common_conf_default.yaml')
# FIXME: losses

# default_values = {
#     'g_network': 'UNet',
#     'd_network': None,
#     'd2_network': None,
#     'train_data': [os.path.join('..', '..', 'datasets', 'cropped', 'NIND_256_192')],
#     'models_dpath': os.path.join('..', '..', 'models', 'nind_denoise'),
#     'beta1': 0.5,
#     #'weight_SSIM': 0.2,
#     #'weight_L1': 0.05,
#     'lr': 0.0003,
#     'test_reserve_yaml': os.path.join('configs', 'test_set_nind.yaml'),
#     'validation_set_yaml': os.path.join('configs', 'validation_set_30_NIND_256_192_test_set_nind.yaml'),
#     'patience': 3,
#     'weights': {
#         'SSIM': 0.,
#         'L1': 0.,
#         'MSE': 0.,
#         'MSSSIM': 1.,
#         'D1': 0.,
#         'D2': 0.
#     }
# }
# with open(default_values['test_reserve_yaml'], 'r') as fp:
#     default_values['test_reserve'] = yaml.safe_load(fp)

class DebugOptions(Enum):
    SHORT_RUN = 'short_run'
    CHECK_DATASET = 'check_dataset'
    OUTPUT_VAL_IMAGES = 'output_val_images'
    OUTPUT_TEST_IMAGES = 'output_test_images'
    KEEP_ALL_OUTPUT_IMAGES = 'keep_all_output_images'

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
    def complete_path(path, models_dpath, keyword=''):
        '''
        Return a filepath for instantiate_model when the provided model_path is a directory (path, 
        or name of a directory located in models_dpath)
        models_dpath is the root directory where all models would be located
        '''
        def find_highest(paths, model_t):
            best = [None, 0]
            for path in paths:
                curval = int(path.split('_')[-1].split('.')[0])
                if curval > best[1] and model_t in path:
                    best = [path, curval]
            return best[0]
        def find_best(dpath, model_t):
            if model_t != 'generator':
                return None   # hardcoded rules and not implemented for discriminators
            resdpath = os.path.join(dpath, 'trainres.json')
            if not os.path.isfile(resdpath):
                print(f'find_best did not find {resdpath}')
                return None
            with open(resdpath, 'r') as fp:
                res = json.load(fp)
                best_epoch = res['best_epoch']['validation_loss']
            return os.path.join(dpath, f'generator_{best_epoch}.pt')
        if os.path.isfile(path):
            # path exists; nothing to do
            return path
        elif os.path.isdir(path):
            # path is a directory; try to find best model from json, or return latest
            best_model_path = find_best(path, model_t=keyword)
            if best_model_path is not None:
                return best_model_path
            return os.path.join(path, find_highest(os.listdir(path), keyword))
        elif os.path.isdir(os.path.join(models_dpath, path)):
            # if models_dpath/path is a directory, recurse
            return Model.complete_path(os.path.join(models_dpath, path), keyword)
        else:
            print("Model path not found: %s"%path)
            exit(0)

    @staticmethod
    def instantiate_model(models_dpath, model_path=None, network=None, device='cuda:0', strparameters=None, pfun=print, keyword='', **parameters):
        '''
        instantiate the internal model used by a Model object
        '''
        model = None
        if strparameters is not None and strparameters != "":
            parameters.update(dict([parameter.split('=') for parameter in strparameters.split(',')]))
        if model_path is not None:
            path = Model.complete_path(path=model_path, keyword=keyword, models_dpath=models_dpath)
            if path.endswith('.pth'):
                model = torch.load(path, map_location=device, weights_only=False)
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


    def eval(self):
        self.model = self.model.eval()
        return self
    
    def train(self):
        self.model = self.model.train()
        return self
    
def get_test_reserve_list(test_reserve):
    '''
    input: test_reserve argument (list or yaml path)
    output: test_reserve list
    '''
    if len(test_reserve) == 1:
        if test_reserve[0].endswith('.yaml'):
            with open(test_reserve[0], 'r') as fp:
                return yaml.safe_load(fp)
        elif test_reserve[0] == '0':
            return []
    return test_reserve


class Generator(Model):
    def __init__(self, network, device, weights, beta1, lr, patience, models_dpath, model_path = None,
                 activation='PReLU', funit=32, printer=None, compute_SSIM_anyway=False,
                 save_dict=True, debug_options=[], reduce_lr_factor=.75):
        Model.__init__(self, save_dict, device, printer, debug_options=[])
        self.weights = weights
        self.criterions = dict()
        if weights['SSIM'] > 0 or compute_SSIM_anyway:
            self.criterions['SSIM'] = pt_losses.SSIM_loss().to(device)
        if weights['L1'] > 0:
            self.criterions['L1'] = torch.nn.L1Loss(reduction=None).to(device)
        if weights['MSE'] > 0:
            self.criterions['MSE'] = torch.nn.MSELoss(reduction=None).to(device)
        if weights['MSSSIM'] > 0:
            self.criterions['MSSSIM'] = pt_losses.MS_SSIM_loss().to(device)
        if weights['D1'] > 0:
            self.print('DBG/FIXME may need to mess with the losses reduction for discriminator compatibility')
            self.criterions['D1'] = torch.nn.MSELoss().to(device)
        if weights['D2'] > 0:
            self.criterions['D2'] = torch.nn.MSELoss().to(device)
        self.model = self.instantiate_model(model_path=model_path, network=network, pfun=self.print, device=device, funit=funit, keyword='generator', models_dpath=models_dpath, activation=activation)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999), amsgrad=True)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=reduce_lr_factor, verbose=True, threshold=1e-8, patience=patience)
        self.device = device
        self.loss = {'SSIM': 1, 'L1': 1, 'D1': 1, 'D2': 1, 'MSE': 1, 'MSSSIM': 1, 'weighted': 1}
        self.compute_SSIM_anyway = compute_SSIM_anyway

    def get_loss(self, pretty_printed=False, component='weighted'):
        '''
        Return a component of the last computed loss (print-only; no compute)
        '''
        if pretty_printed:
            return ", ".join(["%s: %.3f"%(key, val) if val != 1 else 'NA' for key,val in self.loss.items()])
        return self.loss[component]

    def denoise_batch(self, noisy_batch):
        return self.model(noisy_batch).clip(0,1)

    def learn(self, generated_batch_cropped, clean_batch_cropped, discriminators_predictions=[None, None]):
        '''
        input: self-generated clean batch, noisy batch
        compute loss and optimize self
        '''
        loss = self.compute_loss(generated_batch_cropped, clean_batch_cropped, discriminators_predictions)
#         # too late, must have gotten a bad crop from previous batch
#         if loss > 0.4:
#             if self.stable:
#                 os.makedirs('dbg', exist_ok=True)
#                 torchvision.utils.save_image(generated_batch_cropped, os.path.join('dbg', str(self)+'_gen.png'))
#                 torchvision.utils.save_image(clean_batch_cropped, os.path.join('dbg', str(self)+'_gt.png'))
#                 breakpoint()
#         elif loss < 0.2 and not self.stable:
#             self.stable = True
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def compute_loss(self, generated_batch_cropped, clean_batch_cropped, discriminators_predictions=[None, None]):
        '''
        Compute the loss between a denoised and ground-truth batch. Update internal loss dict and
        return weighted loss tensor.
        '''
        loss_weighted = 0
        for loss_name, weight in self.weights.items():
            if weight == 0:
                continue
            elif loss_name[0] == 'D':
                discriminator_predictions = discriminators_predictions[int(loss_name[1])]
                if discriminator_predictions is None:
                    continue
                self.loss[loss_name] = self.criterions[loss_name](
                    discriminator_predictions, gen_target_probabilities(
                        True, discriminator_predictions.shape, device=self.device, noisy=False))
            else:
                self.loss[loss_name] = self.criterions[loss_name](generated_batch_cropped,
                                                                  clean_batch_cropped)
            loss_weighted += self.loss[loss_name] * weight
            self.loss[loss_name] = self.loss[loss_name].mean().item()
        self.loss['weighted'] = loss_weighted.mean().item()
        # Debug. can be removed to increase performance slightly
        if self.loss['weighted'] < 0.25 and loss_weighted.min() > 0.4:
            self.p.print('problematic crop saved to dbg')
            worstval, worstindex = loss_weighted.max()
            os.makedirs('dbg', exist_ok=True)
            torchvision.utils.save_image(generated_batch_cropped, os.path.join('dbg', f'{self}_{float(self.loss["weighted"])}_{float(worstval)}_{int(worstindex)}_gen.png'))
            torchvision.utils.save_image(clean_batch_cropped, os.path.join('dbg', f'{self}_{float(self.loss["weighted"])}_{float(worstval)}_{int(worstindex)}_gt.png'))
            breakpoint()
        return loss_weighted
    
    def update_learning_rate(self, lr_decay):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
        return param_group['lr'] * lr_decay
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def tocpu(self):
        self.model = self.model.cpu()
        for lossname, lossf in self.criterions.items():
            if lossf is None:
                continue
            self.criterions[lossname] = lossf.cpu()
    
    def todevice(self):
        self.model = self.model.to(self.device)
        for lossname, lossf in self.criterions.items():
            if lossf is None:
                continue
            self.criterions[lossname] = lossf.to(self.device)


class Discriminator(Model):
    def __init__(self, models_dpath, beta1, lr, patience, network='Hul112Disc', weights_dict_path=None,
                 model_path=None, device='cuda:0', loss_function='MSE',
                 activation='PReLU', funit=32, not_conditional = False, printer=None, save_dict=True,
                 debug_options=[], reduce_lr_factor=.75):
        Model.__init__(self, save_dict, device, printer, debug_options)
        self.device = device
        self.loss = 1
        self.loss_function = loss_function
        if loss_function == 'MSE':
            self.criterion = torch.nn.MSELoss().to(device)
        if not_conditional:
            input_channels = 3
        else:
            input_channels = 6
        self.model = self.instantiate_model(model_path=model_path, models_dpath=models_dpath, network=network, pfun=self.print, device=device, funit=funit, input_channels = input_channels)
            #elif network == 'PatchGAN':
            #    self.model = net_d = define_D(input_channels, 2*funit, 'basic', gpu_id=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
        self.conditional = not not_conditional
        self.predictions_range = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=patience)

    def update_learning_rate(self, avg_loss):
        self.scheduler.step(metrics=avg_loss)
        lr = self.optimizer.param_groups[0]['lr']
        self.print('Learning rate: %f' % lr)
        return lr
    
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

# should have been replaced with pt_ops.crop_batch_pair and mandatory loss_crop_size argument
# def get_crop_boundaries(cs, ucs, network=None, discriminator=None):
#     '''
#     Determine the center crop used by the loss function; how much the network is allowed to throw away
#     (typically 1/8 per side with U-Net, 1/16 with more efficient networks)
#     '''
#     # if '112' in discriminator:
#     #     loss_crop_lb = int((cs-112)/2)
#     #     loss_crop_up = cs - loss_crop_lb
#     #     assert (loss_crop_up - loss_crop_lb) == 112
#     if network == 'UNet':    # UNet requires huge borders
#         loss_crop_lb = int(cs/8)
#         loss_crop_up = cs-loss_crop_lb
#     elif network == 'UtNet' or network == 'UtdNet': # mirrored in-net
#         loss_crop_lb = max(1,int(cs/32))
#         loss_crop_up = cs-loss_crop_lb
#     else:
#         loss_crop_lb = int(cs/16)
#         loss_crop_up = cs-loss_crop_lb
#     print('Using %s as bounds'%(str((loss_crop_lb, loss_crop_up))))
#     #assert (loss_crop_up - loss_crop_lb) <= ucs, f'({loss_crop_up} - {loss_crop_lb}) <= {ucs}'
#     return loss_crop_lb, loss_crop_up

def gen_target_probabilities(target_real, target_probabilities_shape, device=None, invert_probabilities=False, noisy = True):
    '''
    fuzziness for the discriminator's targets, because blind confidence is not right.
    '''
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
    weights = {'MSSSIM': 0, 'L1': 0, 'MSE': 0, 'SSIM': 0, 'D1': 0, 'D2': 0}
    if args.weight_SSIM:
        weights['SSIM'] = args.weight_SSIM
        total += args.weight_SSIM
    if args.weight_MSSSIM:
        weights['MSSSIM'] = args.weight_MSSSIM
        total += args.weight_MSSSIM
    if args.weight_L1:
        weights['L1'] = args.weight_L1
        total += args.weight_L1
    if args.weight_D1:
        weights['D1'] = args.weight_D1
        total += args.weight_D1
    if args.weight_D2:
        weights['D2'] = args.weight_D2
        total += args.weight_D2
    if args.weight_MSE:
        weights['MSE'] = args.weight_MSE
        total += args.weight_MSE
    if total == 0:
        weights = default_values['weights']
        print('Using default weights')
    elif total != 1:
        for akey in weights.keys():
            weights['akey'] /= total
    assert sum(weights.values()) == 1, weights
    print(f'Loss weights: {weights}')
    return weights

if __name__ == '__main__':
    print(f'default_values: {default_values}')
