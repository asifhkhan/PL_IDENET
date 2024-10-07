import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch
import os


def save_network(path_model, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(path_model, save_filename)
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


def load_network(load_path, network, strict:bool=True):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path)
    # if 'params_ema' in load_net:
    #     keyname = 'params_ema'

    #        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    #        for k, v in load_net.items():
    #            if k.startswith('module.'):
    #                load_net_clean[k[7:]] = v
    #            else:
    #                load_net_clean[k] = v
    network.load_state_dict(load_net, strict=True)
    return network


def custom_load(netG, netD, load_path_G, load_path_D, logger):
    if load_path_G is not None:
        logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
        netG = load_network(load_path_G, netG)

    if load_path_D is not None:
        logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
        netD = load_network(load_path_D, netD)
    return netG, netD


def save_checkpoint_best(model_path, epoch, iter_step, optimizers, schedulers, metrics, label, logger):
    state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
    state = {**state, **metrics}
    for s in schedulers:
        state['schedulers'].append(s.state_dict())
    for o in optimizers:
        state['optimizers'].append(o.state_dict())
    save_filename = '{}_tr_states_best_'.format(epoch) + label + '.pth'
    save_path = os.path.join(model_path, save_filename)
    torch.save(state, save_path)
    logger.info("===> Checkpoint saved to {:s}".format(save_path))


def save_training_state(epoch, iter_step, optimizers, schedulers, metrics,training_states_save_path):
    """Save training state during training, which will be used for resuming"""
    state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
    state = {**state, **metrics}
    for s in schedulers:
        state['schedulers'].append(s.state_dict())
    for o in optimizers:
        state['optimizers'].append(o.state_dict())
    save_filename = '{}_tr_states.pth'.format(epoch)
    save_path = os.path.join(training_states_save_path, save_filename)
    torch.save(state, save_path)


def resume_training(resume_path, optimizers, schedulers, opt):
    """Resume the optimizers and schedulers for training"""
    if opt.cuda:
        resume_state = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
    else:
        resume_state = torch.load(resume_path, map_location=lambda storage, loc: storage)

    start_epoch = resume_state['epoch']
    iter_step = resume_state['iter']
    epoch_psnr_old = resume_state['epoch_psnr_old']
    epoch_lpips_old = resume_state['epoch_lpips_old']
    resume_optimizers = resume_state['optimizers']
    resume_schedulers = resume_state['schedulers']
    assert len(resume_optimizers) == len(optimizers), 'Wrong lengths of optimizers'
    assert len(resume_schedulers) == len(schedulers), 'Wrong lengths of schedulers'
    for i, o in enumerate(resume_optimizers):
        optimizers[i].load_state_dict(o)
    for i, s in enumerate(resume_schedulers):
        schedulers[i].load_state_dict(s)

    return start_epoch, iter_step, epoch_psnr_old, epoch_lpips_old, optimizers, schedulers
