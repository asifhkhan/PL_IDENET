import torch
from modules import lr_scheduler as lr_scheduler


def build(netG, netD, opt):

    # Adam
    trainable_parameters = []
    trainable_parameters += list(netG.parameters())
    # trainable_parameters += list(netK.parameters())
    optimizer_G = torch.optim.Adam(trainable_parameters,
                                   lr=opt.lr_G,
                                   betas=(opt.beta1_G, opt.beta2_G),
                                   eps=opt.eps_G,
                                   weight_decay=opt.weightdecay_G,
                                   amsgrad=opt.amsgrad)

    optimizer_D = torch.optim.Adam(netD.parameters(),
                                   lr=opt.lr_D,
                                   betas=(opt.beta1_D, opt.beta2_D),
                                   eps=opt.eps_D,
                                   weight_decay=opt.weightdecay_D,
                                   amsgrad=opt.amsgrad)
    optimizers = [optimizer_G, optimizer_D]

    # schedulers
    scheduler_G = lr_scheduler.MultiStepLR_Restart(optimizer_G,
                                                   milestones=opt.lr_milestones,
                                                   restarts=opt.lr_restart,
                                                   weights=opt.lr_restart_weights,
                                                   gamma=opt.lr_gamma)
    scheduler_D = lr_scheduler.MultiStepLR_Restart(optimizer_D,
                                                   milestones=opt.lr_milestones,
                                                   restarts=opt.lr_restart,
                                                   weights=opt.lr_restart_weights,
                                                   gamma=opt.lr_gamma)
    schedulers = [scheduler_G, scheduler_D]

    return optimizers, schedulers