import matplotlib as mpl

import os
import torch

from utils.model_normalization import MNISTWrapper
import utils.datasets as dl
import utils.models.model_factory_32 as factory
import utils.run_file_helpers as rh
from distutils.util import strtobool

import argparse

# change backend
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Define hyperparameters.',
                                 prefix_chars='-')
parser.add_argument('--net', type=str, default='ResNet18',
                    help='Resnet18, 34 or 50, WideResNet28')
parser.add_argument('--model_params', nargs='+', default=[])
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or semi-cifar10')
parser.add_argument('--od_dataset', type=str, default='tinyImages',
                    help=('tinyImages or cifar100'))
parser.add_argument('--exclude_cifar', dest='exclude_cifar',
                    type=lambda x: bool(strtobool(x)), default=True,
                    help='whether to exclude cifar10 from tiny images')

rh.parser_add_commons(parser)
rh.parser_add_adversarial_commons(parser)
rh.parser_add_adversarial_norms(parser, 'cifar10')

hps = parser.parse_args()

# device
device_ids = None
if len(hps.gpu) == 0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu) == 1:
    device = torch.device('cuda:' + str(hps.gpu[0]))
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))

# load model
num_classes = 10

model, model_name, model_config, _ = factory.build_model(
    hps.net, num_classes, model_params=hps.model_params, is_rgb=False)

# image size
img_size = 28

model_root_dir = 'MNIST_models'
logs_root_dir = 'MNIST_logs'
model_dir = os.path.join(model_root_dir, model_name)
log_dir = os.path.join(logs_root_dir, model_name)

start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir,
                                                         device, hps)
model = MNISTWrapper(model).to(device)

msda_config = rh.create_msda_config(hps)

# load dataset
od_bs = int(hps.od_bs_factor * hps.bs)

id_config = {}
train_loader = dl.MNIST(train=True, batch_size=hps.bs, augm_flag=True)

od_config = {}
loader_config = {'ID config': id_config, 'OD config': od_config}

if hps.train_type.lower() in ['ceda', 'acet', 'advacet',
                              'tradesacet', 'tradesceda']:
    # FIXME: for debug only
    # download datasets and test these cases
    # for now, only `plain` train type is tested
    tiny_train = train_loader

    if hps.od_dataset == 'tinyImages':
        tiny_train = dl.get_80MTinyImages(batch_size=od_bs, augm_type=hps.augm,
                                          num_workers=1, size=img_size,
                                          exclude_cifar=hps.exclude_cifar,
                                          exclude_cifar10_1=hps.exclude_cifar,
                                          config_dict=od_config)
    elif hps.od_dataset == 'cifar100':
        tiny_train = dl.get_CIFAR100(train=True, batch_size=od_bs,
                                     shuffle=True, augm_type=hps.augm,
                                     size=img_size, config_dict=od_config)
    elif hps.od_dataset == 'openImages':
        tiny_train = dl.get_openImages('train', batch_size=od_bs, shuffle=True,
                                       augm_type=hps.augm, size=img_size,
                                       exclude_dataset=None,
                                       config_dict=od_config)
else:
    loader_config = {'ID config': id_config}

# test dataset
test_loader = dl.MNIST(train=False, batch_size=hps.bs, augm_flag=True)


scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)
id_attack_config, od_attack_config = rh.create_attack_config(hps, 'cifar10')
trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config,
                            device, num_classes, model_dir, log_dir,
                            msda_config=msda_config, model_config=model_config,
                            id_attack_config=id_attack_config,
                            od_attack_config=od_attack_config)
# DEBUG:
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# run training
if trainer.requires_out_distribution():
    train_loaders, test_loaders = trainer.create_loaders_dict(
        train_loader, test_loader=test_loader,
        out_distribution_loader=tiny_train)

    trainer.train(train_loaders, test_loaders, loader_config=loader_config,
                  start_epoch=start_epoch, optim_state_dict=optim_state_dict,
                  device_ids=device_ids)
else:
    train_loaders, test_loaders = trainer.create_loaders_dict(
        train_loader, test_loader=test_loader)

    trainer.train(train_loaders, test_loaders, loader_config=loader_config,
                  start_epoch=start_epoch, optim_state_dict=optim_state_dict,
                  device_ids=device_ids)
