from datetime import datetime
import argparse
from tqdm import tqdm
from ray import tune
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator
from config import nets, config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print("Device", device)


def train(netG, netD, lrG, lrD, default_batch_size, n_epochs, name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    dataset = MNIST('/data', download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=default_batch_size, shuffle=True)

    # netG = Generator().to(device)
    # netD = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizerG = Adam(netG.parameters(), lr=lrG)
    optimizerD = Adam(netD.parameters(), lr=lrD)

    fixed_noises = torch.rand(64, 100, device=device)

    writer = SummaryWriter('/logs/' + name)

    n_iters = 1
    for e in tqdm(range(1, n_epochs + 1)):
        netG.train()
        netD.train()
        for data in data_loader:
            #
            # Train Discriminator
            #
            netD.zero_grad()

            real_images = data[0].view(-1, 784).to(device)
            batch_size = real_images.size(dim=0)
            labels = torch.full((batch_size, ), 1., dtype=torch.float, device=device)
            outputs = netD(real_images).view(-1)

            errD_real = criterion(outputs, labels)
            errD_real.backward()

            noises = torch.rand(batch_size, 100, device=device)
            fake_images = netG(noises)
            labels.fill_(0.)
            outputs = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(outputs, labels)
            errD_fake.backward()

            errD = (errD_real + errD_fake) / 2.

            optimizerD.step()

            writer.add_scalar('Loss/Discriminator', errD.item(), n_iters)

            #
            # Train Generator
            # 
            netG.zero_grad()

            labels.fill_(1.)
            outputs = netD(fake_images).view(-1)
            errG = criterion(outputs, labels)
            errG.backward()

            optimizerG.step()

            writer.add_scalar('Loss/Generator', errG.item(), n_iters)

            n_iters += 1

        netG.eval()
        fake_images = netG(fixed_noises).view(-1, 1, 28, 28)

        writer.add_images('Generated Images', fake_images, e)

        torch.save({
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, f'/models/{name}-{e}.pth')


def trainable(config):
    netG = Generator(**nets[config['netG']]).to(device)
    netD = Discriminator(**nets[config['netD']]).to(device)
    name = tune.get_trial_name() + '--' + '--'.join([f'{key}-{value}' for key, value in config.items()])
    train(netG, netD, config['lrG'], config['lrD'], config['batch_size'], config['n_epochs'], name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if args.train:
        tune.run(trainable, config=config, resources_per_trial={'gpu': 0.2, 'cpu': 1})
    else:
        print('Start gen.')
