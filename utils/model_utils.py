import torch
import argparse
from models.psp import pSp


def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    is_cars = 'car' in opts['dataset_type']
    is_faces = 'ffhq' in opts['dataset_type']
    if is_faces:
        opts['stylegan_size'] = 1024
    elif is_cars:
        opts['stylegan_size'] = 512
    else:
        opts['stylegan_size'] = 256

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts
