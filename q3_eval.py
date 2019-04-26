import numpy as np
import torch
import utils
import models
import matplotlib.pyplot as plt
import torchvision



def compare4by4(vae,gan):
    with torch.no_grad():
        noise_inputs = torch.randn(size=(16,100)).cuda()
        vae_out = vae.generate(noise_inputs).cpu().numpy()
        gan_out = gan.generate(noise_inputs).cpu().numpy()
        f, ax_arr = plt.subplots(4,8)
        plt.tight_layout()
        for i in range(16):
            ax_arr[i//4,i%4].imshow(np.transpose(vae_out[i],(1,2,0)))
            ax_arr[i//4,4+i%4].imshow(np.transpose(gan_out[i],(1,2,0)))
            ax_arr[i//4,4+i%4].axis('off')
            ax_arr[i//4,i%4].axis('off')
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0)
        plt.title("Comparison of VAE and GAN")
        plt.show()

def interpolate_latent(vae,gan,z0,z1):
    f, ax_arr = plt.subplots(2,11, figsize = (13,3))
    with torch.no_grad():
        for a_val in range(11):
            point = (a_val/10)*z0 + (1-(a_val/10))*z1
            vae_out = vae.generate(point).cpu()
            gan_out = gan.generate(point).cpu()
            ax_arr[0,a_val].imshow(np.transpose(vae_out[0],(1,2,0)))
            ax_arr[1,a_val].imshow(np.transpose(gan_out[0],(1,2,0)),)
            ax_arr[0,a_val].axis('off')
            ax_arr[1,a_val].axis('off')
    plt.tight_layout(pad=0.1, h_pad=0.05, w_pad=0)
    plt.title("Interpolation in Latent Space")
    plt.show()
    return

def interpolate_data(vae,gan,z0,z1):
    with torch.no_grad():
        vae_0 = vae.generate(z0).cpu()
        vae_1 = vae.generate(z1).cpu()
        gan_0 = gan.generate(z0).cpu()
        gan_1 = gan.generate(z1).cpu()
        f, ax_arr = plt.subplots(2,11, figsize = (13,3))
        plt.tight_layout()
        for a_val in range(11):
            vae_interp = (a_val/10)*vae_0 + (1-(a_val/10))*vae_1
            gan_interp = (a_val/10)*gan_0 + (1-(a_val/10))*gan_1
            ax_arr[0,a_val].imshow(np.transpose(vae_interp[0],(1,2,0)))
            ax_arr[1,a_val].imshow(np.transpose(gan_interp[0],(1,2,0)))
            ax_arr[0,a_val].axis('off')
            ax_arr[1,a_val].axis('off')
        plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0)
        plt.title("Interpolation in Output Space")
        plt.show()
    return

def wiggle_latent(vae,gan):
    base_noise = torch.randn((1,100)).cuda()
    print(base_noise.device)
    base_vae = vae.generate(base_noise)
    base_gan = gan.generate(base_noise)
    with torch.no_grad():
        for i in range(100):
            epsilon_vec = torch.zeros((1,100)).cuda()
            epsilon_vec[i] = 0.2
            wiggled_noise = base_noise + epsilon_vec
            wiggled_vae = vae.generate(wiggled_noise)
            wiggled_gan = gan.generate(wiggled_noise)

            f, ax_arr = plt.subplots(1,11, figsize = (13,1.5))
            for a_val in range(11):
                vae_interp = (a_val/10)*base_vae + (1-(a_val/10))*wiggled_vae
                ax_arr[0,a_val].imshow(np.transpose(vae_interp[0],(1,2,0)))
                ax_arr[0,a_val].axis('off')
            plt.title("Dimension {} VAE".format(i))
            plt.savefig("figures/VAE_wiggle/Latent_{}".format(i))
            plt.close()

            f, ax_arr = plt.subplots(1,11, figsize = (13,1.5))
            for a_val in range(11):
                gan_interp = (a_val/10)*base_gan + (1-(a_val/10))*wiggled_gan
                ax_arr[0,a_val].imshow(np.transpose(gan_interp[0],(1,2,0)))
                ax_arr[0,a_val].axis('off')
            plt.title("Dimension {} GAN".format(i))
            plt.savefig("figures/GAN_wiggle/Latent_{}".format(i))
            plt.close()
    return

def populate_FID_folder(vae,gan):
    noise_inputs = torch.randn(500,100).cuda()
    vae_outs = vae.generate(noise_inputs)
    gan_outs = gan.generate(noise_inputs)
    for i in range(500):
        torchvision.utils.save_image(vae_outs[i].cpu(),"FID_folder/img_{}".format(i*2))
        torchvision.utils.save_image(gan_outs[i].cpu(),"FID_folder/img_{}".format(i*2+1))