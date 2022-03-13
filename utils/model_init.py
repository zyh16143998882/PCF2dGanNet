# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import os
import torch
import logging
import numpy as np
import configs.model_names as name
from utils.p2i_utils import ComputeDepthMaps, MviewComputeDepthMaps
from models.sparenet_discriminator import ProjectionD, PatchDiscriminator

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def generator_init(cfg):
    """
    input:
        cfg: EasyDict

    outputs:
        net_G: torch.nn.module
        optimizerG: torch.optim.Adam
        lr_schedulerG: torch.optim.lr_scheduler.MultiStepLR
    """
    # Create the networks
    net_G = define_G(cfg)
    net_G.apply(init_weights)               # 从初始值开始初始化权重
    logger.debug("Parameters in net_G: %d." % count_parameters(net_G))

    # Optimizer and learning scheduler
    optimizerG = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net_G.parameters()),              # 已经提前设置好了不进行反向传播的filter
        lr=cfg.TRAIN.learning_rate,
        weight_decay=cfg.TRAIN.weight_decay,
        betas=cfg.TRAIN.betas,
    )
    lr_schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=cfg.TRAIN.lr_milestones, gamma=cfg.TRAIN.gamma)
    return net_G, optimizerG, lr_schedulerG

# 初始化渲染器
def renderer_init(cfg):
    """
    input:
        cfg: EasyDict

    output:
        renderer: torch.nn.module
    """
    # Create the differentiable renderer
    renderer_gen = MviewComputeDepthMaps(
        projection=cfg.RENDER.projection,
        eyepos_scale=cfg.RENDER.eyepos,
        image_size=cfg.RENDER.img_size,
    ).float()
    renderer_dis = ComputeDepthMaps(
        projection=cfg.RENDER.projection,
        eyepos_scale=cfg.RENDER.eyepos,
        image_size=cfg.RENDER.img_size,
    ).float()

    return renderer_gen, renderer_dis


# netD初始化
def discriminator_init(cfg):

    # Create discriminator: projection discriminator or patchgan
    if cfg.GAN.use_cgan:
        net_D = ProjectionD(
            num_classes=cfg.DATASET.num_classes,
            img_shape=(cfg.RENDER.n_views * 2, cfg.RENDER.img_size, cfg.RENDER.img_size),
        )
    else:
        net_D = PatchDiscriminator(
            img_shape=(cfg.RENDER.n_views * 2, cfg.RENDER.img_size, cfg.RENDER.img_size),
        )

    net_D.apply(init_weights_D)
    logger.debug("Parameters in net_D: %d." % count_parameters(net_D))

    # Optimizer and learning scheduler
    optimizerD = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net_D.parameters()),
        lr=cfg.TRAIN.learning_rate,
        weight_decay=cfg.TRAIN.weight_decay,
        betas=cfg.TRAIN.betas,
    )
    lr_schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=cfg.TRAIN.lr_milestones, gamma=cfg.TRAIN.gamma)
    return net_D, optimizerD, lr_schedulerD



def define_G(cfg):
    if cfg.NETWORK.model_type == name.MODEL_SPARENET:
        from models.sparenet_generator import SpareNetGenerator

        # from models.mvnet import MVNet
        network = SpareNetGenerator(
            num_points=cfg.DATASET.n_outpoints,
            bottleneck_size=4096,
            n_primitives=cfg.NETWORK.n_primitives,
            use_SElayer=cfg.NETWORK.use_selayer,
            use_AdaIn=cfg.NETWORK.use_adain,
            use_RecuRefine=cfg.NETWORK.use_recurefine,
            encode=cfg.NETWORK.encode,
            decode=cfg.NETWORK.decode,
            hide_size=4096,
        )
    elif cfg.NETWORK.model_type == name.MODEL_PCF2DNET:
        from models.sparenet_generator import PCF2dNetGenerator

        # from models.mvnet import MVNet
        network = PCF2dNetGenerator(
            num_points=cfg.DATASET.n_outpoints,
            bottleneck_size=4096,
            n_primitives=cfg.NETWORK.n_primitives,
            use_SElayer=cfg.NETWORK.use_selayer,
            use_AdaIn=cfg.NETWORK.use_adain,
            use_RecuRefine=cfg.NETWORK.use_recurefine,
            encode=cfg.NETWORK.encode,
            decode=cfg.NETWORK.decode,
            hide_size=4096,
        )
    else:
        raise Exception("Unknown model type")
    return network


def init_weights(m):
    if type(m) in [
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose3d,
    ] and hasattr(m, "weight"):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) in [torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == torch.nn.Linear:
        if hasattr(m, "weight"):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def init_weights_D(m):
    classname = m.__class__.__name__
    if (
        classname.find("Conv2d") != -1
        and hasattr(m, "weight")
        or classname.find("Conv2d") == -1
        and classname.find("Conv1d") != -1
        and hasattr(m, "weight")
    ):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if (
        classname.find("BatchNorm2d") != -1
        or classname.find("BatchNorm2d") == -1
        and classname.find("BatchNorm1d") != -1
    ):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())

