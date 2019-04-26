import torch
import numpy as np



##################################################################
######################## HELPER SECTION ##########################
##################################################################

class Upsample(torch.nn.Module):
    def __init__(self):
        super(Upsample,self).__init__()
        self.upsample = torch.nn.functional.interpolate
    def forward(self, x):
        return self.upsample(x,scale_factor=2,mode='bilinear',align_corners=True)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, x):
        return x.view(x.size()[0],-1)

##################################################################
########################## VAE SECTION ###########################
##################################################################

class q2_VAE(torch.nn.Module):
    def __init__(self):
        super(q2_VAE,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,(3,3)),
            torch.nn.ELU(),
            torch.nn.AvgPool2d(2,2),
            torch.nn.Conv2d(32,64,(3,3)),
            torch.nn.ELU(),
            torch.nn.AvgPool2d(2,2),
            torch.nn.Conv2d(64,256,(5,5)),
            torch.nn.ELU()
        )
        self.latent_mean = torch.nn.Linear(256,100)
        self.latent_logvar = torch.nn.Linear(256,100)
        self.decoder = self.decoder_linear = torch.nn.Sequential(
            torch.nn.Linear(100,256),
            torch.nn.ReLU()
        )
        self.decoder_conv = torch.nn.Sequential(      
            torch.nn.Conv2d(256,128,(5,5),padding=(4,4)),
            torch.nn.ELU(),
            Upsample(),
            torch.nn.Conv2d(128,32,(3,3),padding=(2,2)),
            torch.nn.ELU(),
            Upsample(),
            torch.nn.Conv2d(32,16,(3,3),padding=(2,2)),
            torch.nn.ELU(),
            torch.nn.Conv2d(16,1,(3,3),padding=(2,2)),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        encoded_data = self.encoder(x)
        encoded_data = torch.squeeze(encoded_data) # Get rid of the singleton dims

        latent_means = self.latent_mean(encoded_data)
        latent_logvars = self.latent_logvar(encoded_data)
        self.prev_means = latent_means
        self.prev_vars = latent_logvars

        sampled_data = self.sample_given_estimate(latent_means,latent_logvars)
        decoded_data = self.decoder_linear(sampled_data).view(x.size()[0],256,1,1)
        decoded_data = self.decoder_conv(decoded_data)
        return decoded_data

    def sample_given_estimate(self, est_means, est_logvar):
        """
        Reformats our estimated means and logvariances as samples taken with their values
        """
        est_var = torch.exp(est_logvar/2)
        sample_stdevs = torch.randn_like(est_var)
        samples = est_means + est_var*sample_stdevs
        return samples

    def decode(self, sample):
        fc_decode = self.decoder_linear(sample)
        return self.decoder_conv(fc_decode.view(sample.size()[0],256,1,1))


class q3_VAE(torch.nn.Module):
    def __init__(self):
        super(q3_VAE,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3,64,(5,5)),
            torch.nn.LeakyReLU(0.05),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64,128,(5,5)),
            torch.nn.LeakyReLU(0.05),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128,256,(5,5)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
        )
        self.latent_mean = torch.nn.Linear(256,100)
        self.latent_logvar = torch.nn.Linear(256,100)

        self.decoder = Generator()


    def forward(self,x):
        encoded_data = self.encoder(x)
        encoded_data = torch.squeeze(encoded_data) # Get rid of the singleton dims

        latent_means = self.latent_mean(encoded_data)
        latent_logvars = self.latent_logvar(encoded_data)
        self.prev_means = latent_means
        self.prev_vars = latent_logvars

        sampled_data = self.sample_given_estimate(latent_means,latent_logvars)
        decoded_data = self.decoder(sampled_data)
        return decoded_data

    def sample_given_estimate(self, est_means, est_logvar):
        """
        Reformats our estimated means and logvariances as samples taken with their values
        """
        est_var = torch.exp(est_logvar/2)
        sample_stdevs = torch.randn_like(est_var)
        samples = est_means + est_var*sample_stdevs
        return samples

    def decode(self, sample):
        return self.decoder(sample)
    
    def generate(self, sample):
        return self.decoder(sample)
##################################################################
########################## GAN SECTION ###########################
##################################################################

class Generator(torch.nn.Module):
    def __init__(self, latent_size=100):
        super(Generator,self).__init__()
        self.fc_section = torch.nn.Sequential(
            torch.nn.Linear(latent_size,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
        )
        self.conv_section = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256,128,(5,5),padding=(4,4)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            Upsample(),
            torch.nn.Conv2d(128,64,(5,5),padding=(4,4)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Conv2d(64,32,(3,3),padding=(2,2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            Upsample(),
            torch.nn.Conv2d(32,16,(3,3),padding=(1,1)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ELU(),
            torch.nn.Conv2d(16,3,(3,3),padding=(1,1)),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid()
        )
        self.latent_size = latent_size

    def forward(self, x):
        x = self.fc_section(x)
        x = x.view(x.size()[0],256,1,1)
        x = self.conv_section(x)
        return x
    
    def generate(self, sample=False, shape=None):
        if(torch.is_tensor(sample)):
            return self.forward(sample)
        else:
            if(shape is None):
                raise(AssertionError("No sample or shape given to generate function"))
            generated = None
            with torch.no_grad():  
                samples = torch.randn(size=shape)
                generated = self.forward(samples)
            return generated
    

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(3*32*32,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        # Had to simplify since it was outperforming the generator by too much
        # self.critic = torch.nn.Sequential(
        #     torch.nn.Conv2d(3,64,(5,5)),
        #     torch.nn.LeakyReLU(0.05),
        #     torch.nn.AvgPool2d((2,2)),
        #     torch.nn.Conv2d(64,128,(5,5)),
        #     torch.nn.LeakyReLU(0.05),
        #     torch.nn.AvgPool2d((2,2)),
        #     torch.nn.Conv2d(128,256,(5,5)),
        #     torch.nn.LeakyReLU(0.05),
        #     Flatten(),
        #     torch.nn.Linear(256,128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128,64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64,1)
        # )
        self.act_fn = torch.nn.Sigmoid()
        return

    def forward(self, x):
        x = x.view(-1,3*32*32)
        x = self.critic(x)
        return self.act_fn(x)

class GAN(torch.nn.Module):
    def __init__(self, latent_size = 100):
        super(GAN,self).__init__()
        self.generator = Generator(latent_size = latent_size)
        self.critic = Discriminator()
        self.latent_size = latent_size
        return

    def critique(self, x):
        return self.critic(x)

    def generate(self, sample=None, shape=None):
        return self.generator.generate(sample,shape)
        