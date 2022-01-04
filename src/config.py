import torch.nn as nn
from ray import tune


large_netG_normalized = {
    'normalize': True,
    'blocks': [
        {'in_features': 100, 'out_features': 256, 'activation_func': nn.ReLU()},
        {'in_features': 256, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


large_netG = {
    'normalize': False,
    'blocks': [
        {'in_features': 100, 'out_features': 256, 'activation_func': nn.ReLU()},
        {'in_features': 256, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


medium_netG_normalized = {
    'normalize': True,
    'blocks': [
        {'in_features': 100, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


medium_netG = {
    'normalize': False,
    'blocks': [
        {'in_features': 100, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


small_netG_normalized = {
    'normalize': True,
    'blocks': [
        {'in_features': 100, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


small_netG = {
    'normalize': False,
    'blocks': [
        {'in_features': 100, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()}
    ]
}


large_netD_normalized = {
    'normalize': True,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 64, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 64, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ]
}


large_netD = {
    'normalize': False,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 64, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 64, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ] 
}


medium_netD_normalized = {
    'normalize': True,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ]
}


medium_netD = {
    'normalize': False,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ]
}


small_netD_normalized = {
    'normalize': True,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ]
}


small_netD = {
    'normalize': False,
    'p': 0.5,
    'blocks': [
        {'in_features': 784, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 1, 'activation_func': nn.Sigmoid()}
    ]
}  


nets = {
    'large_netG_normalized': large_netG_normalized,
    'large_netG': large_netG,
    'medium_netG_normalized': medium_netG_normalized,
    'medium_netG': medium_netG,
    'small_netG_normalized': small_netG_normalized,
    'small_netG': small_netG,
    'large_netD_normalized': large_netD_normalized,
    'large_netD': large_netD,
    'medium_netD_normalized': medium_netD_normalized,
    'medium_netD': medium_netD,
    'small_netD_normalized': small_netD_normalized,
    'small_netD': small_netD
}


config = {
    'netG': tune.grid_search([
        'large_netG_normalized',
        'large_netG',
        'medium_netG_normalized',
        'medium_netG',
        'small_netG_normalized',
        'small_netG'
    ]),
    'netD': tune.grid_search([
        'large_netD_normalized',
        'large_netD',
        'medium_netD_normalized',
        'medium_netD',
        'small_netD_normalized',
        'small_netD'
    ]),
    'batch_size': 128,
    'n_epochs': 200,
    'lrG': tune.grid_search([4e-3, 4e-4]),
    'lrD': tune.grid_search([4e-3, 4e-4]),
}
