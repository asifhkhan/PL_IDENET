import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_network_description(network):
    """Get the string and total parameters of the network"""
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    return str(network), sum(map(lambda x: x.numel(), network.parameters()))


def print_network(netG, netD, netF, netK, logger):
    # Generator
    if netG is not None:
        s, n = get_network_description(netG)
        if isinstance(netG, nn.DataParallel) or isinstance(netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(netG.__class__.__name__,
                                             netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Kernel Estimator
    if netK is not None:
        s, n = get_network_description(netK)
        if isinstance(netK, nn.DataParallel) or isinstance(netK, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(netK.__class__.__name__,
                                             netK.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(netK.__class__.__name__)
        logger.info('Network K structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Discriminator
    if netD is not None:
        s, n = get_network_description(netD)
        if isinstance(netD, nn.DataParallel) or isinstance(netD, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(netD.__class__.__name__,
                                             netD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(netD.__class__.__name__)
        logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # F, Perceptual Network
    if netF is not None:
        s, n = get_network_description(netF)
        if isinstance(netF, nn.DataParallel) or isinstance(netF, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(netF.__class__.__name__,
                                             netF.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(netF.__class__.__name__)
        logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)


### Utils for managing network parameters ###
def get_module_name_dict(root, rootname="/"):
    def _rec(module, d, name):
        for key, child in module.__dict__["_modules"].items():
            d[child] = name + key + "/"
            _rec(child, d, d[child])

    d = {root: rootname}
    _rec(root, d, d[root])
    return d


def parameters_by_module(net, name=""):
    modulenames = get_module_name_dict(net, name + "/")
    params = [{"params": p, "name": n, "module": modulenames[m]} for m in net.modules() for n, p in
              m._parameters.items() if p is not None]
    return params


def parameter_count(net):
    parameters = parameters_by_module(net)
    nparams = 0
    for pg in parameters:
        for p in pg["params"]:
            nparams += p.data.numel()
    return nparams
