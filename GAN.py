import numpy as np
import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        return
    def forward(self, x):
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        return
    def forward(self, x):
        return x

class GAN(torch.nn.Module):
    def __init__(self, input_size=32**2, latent_dimensions=100):
        super(GAN,self).__init__()
        self.generator = Generator(latent_size = latent_dimensions,output_size = input_size)
        self.critic = Discriminator(input_size = input_size, n_classes = 10)
        return
    def forward(self, x):
        return