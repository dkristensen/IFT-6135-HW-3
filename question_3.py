import torch
import numpy as np
from models import GAN, q3_VAE
import utils
import classify_svhn
import models
import q3_eval

%pylab inline

def vae_loss(model_in, model_out, mus, logvars):
    MSE = torch.sum(torch.nn.functional.mse_loss(model_out, model_in, reduction="none"))
    KLD = 0.5 * torch.sum(mus**2 + torch.exp(logvars) -1 - logvars)
    return (MSE + KLD) / len(mus)
  
def train_GAN(model, optimizers, train_dataloader, tradeoff_update=8, tradeoff_length=1, lambda_val=10):
    model.train()
    model.cuda()
    lambda_val = 15

    gen_optimizer = optimizers["generator"]
    dis_optimizer = optimizers["discriminator"]

    g_loss, d_loss = 0,0
    for i, (batch,labels) in enumerate(train_dataloader):
        dis_optimizer.zero_grad()
        batch = batch.view(-1,3,32,32).cuda()
        critic_real = model.critique(batch)

        generator_out = model.generate(torch.randn(size=(len(batch),model.latent_size)).cuda())
        critic_fake = model.critique(generator_out)
        loss = torch.mean(critic_real) - torch.mean(critic_fake)
        
        # Update the Critic/Discriminator  
        interpolation_values = torch.randn(batch.shape).cuda()

        interpolation_points = interpolation_values*batch + (1-interpolation_values) * generator_out
        critic_interp = model.critique(interpolation_points)
        # Lifted from my code in density_estimation.py
        model_gradient = torch.autograd.grad(outputs=critic_interp,inputs = interpolation_points, grad_outputs=torch.ones(critic_interp.size()).cuda(), create_graph=True, only_inputs=True, retain_graph=True)[0]
        loss = -(loss - lambda_val * torch.mean((torch.norm(model_gradient) - 1)**2))
        loss.backward()
        dis_optimizer.step()

        d_loss += -loss.item()

        # Update the Generator
        if(i%tradeoff_update == 0):
            gen_optimizer.zero_grad()
            generator_out = model.generate(torch.randn(size=(len(batch),model.latent_size)).cuda())
            critic_fake = model.critique(generator_out)
            
            loss = -torch.mean(critic_fake)
            loss.backward()
            gen_optimizer.step()

            g_loss += -loss.item()

    return d_loss/len(train_loader),g_loss
  
def evaluate_GAN(model, valid_loader):
    model.eval()
    fake = None
    valid_loss = 0
    n_examples = 0
    with torch.no_grad():
        for i, (batch, label) in enumerate(valid_loader):
            n_examples += len(batch)
            batch = batch.view(-1,3,32,32).cuda()
            real_prob = model.critique(batch)
            fake = model.generate(torch.randn(batch.size()[0], model.latent_size).cuda())
            with torch.enable_grad():
                batch_loss = torch.mean(model.critique(fake))
                valid_loss += -batch_loss.item()
    fake_img = np.transpose(fake[0].cpu().numpy(), (1, 2, 0))
    valid_loss = valid_loss/len(valid_loader)
    return valid_loss
            
            
            
def train_VAE(model, optimizer, train_dataloader):
    model.train()

    loss_fn = vae_loss

    # Bulk of the code lifted from question_2.py
    total_training_loss = 0

    n_examples = 0
    for batch_index, (batch,labels) in enumerate(train_dataloader):
        batch = batch.view(-1,3,32,32).float().cuda()
        output_batch = model(batch)
        loss = loss_fn(batch, output_batch, model.prev_means, model.prev_vars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_training_loss += loss.item()
        n_examples+=len(batch)
        
    return total_training_loss/n_examples
    
def evaluate_VAE(model, valid_loader):
    """Computes the ELBO for a model.

    Args:
        model (torch.nn.Module): Module to use for computing the ELBO
        val_dataloader (torch.utils.data.DataLoader): Dataset which we use to compute the ELBO
        loss_fn (any loss function): The loss function, in our case, the ELBO, we use to check our model

    Returns:
        float: The ELBO of our model for the given dataset
    """
    model.eval()
    total_loss = 0
    n_examples = 0
    output_batch = None
    with torch.no_grad():
        for batch_index, (batch,label) in enumerate(valid_loader):
            batch = batch.view(-1,3,32,32).float().cuda()
            output_batch = model(batch)
            total_loss += vae_loss(batch, output_batch, model.prev_means, model.prev_vars)
            n_examples += len(batch)
    fake_img = np.transpose(output_batch[0].cpu().numpy(), (1, 2, 0))
    return total_loss/n_examples

def show_generated(vae, gan):
    with torch.no_grad():
        noise = torch.randn(1,100).cuda()
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(vae.generate(noise).cpu().numpy()[0], (1, 2, 0)))
        plt.title("VAE")
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(gan.generate(noise).cpu().numpy()[0], (1, 2, 0)))
        plt.title("GAN")
        plt.show()
  
  
GAN, VAE = None, None
if __name__ == "__main__":
    train_loader, val_loader, test_loader = classify_svhn.get_data_loader("data/",256)
    GAN = models.GAN(100).cuda()
    gen_optimizer = torch.optim.Adam(GAN.generator.parameters(),lr=1e-3)
    dis_optimizer = torch.optim.Adam(GAN.critic.parameters(),lr=3e-4)
    gan_optimizers = {"generator":gen_optimizer,
                        "discriminator":dis_optimizer}
    
    VAE = models.q3_VAE().cuda()
    vae_optimizer = torch.optim.Adam(VAE.parameters(),lr=3e-4)
    gan_val_loss, gan_train_loss = 0,0
    print("Epoch\tVAE T Loss\tVAE V Loss\tGAN V Loss\t GAN T D Loss\tGAN T G Loss")
    with open("results.csv","w") as f:
      f.write("Epoch,VAE T Loss,VAE V Loss,GAN V Loss,GAN T D Loss,GAN T G Loss")
      for i in range(50):
          d_loss,g_loss = train_GAN(GAN,gan_optimizers,train_loader,tradeoff_update=30)
          gan_val_loss = evaluate_GAN(GAN, val_loader)
          vae_train_loss = train_VAE(VAE,vae_optimizer,train_loader)
          vae_val_loss = evaluate_VAE(VAE,val_loader)
          if(i%5 == 0):
              show_generated(VAE,GAN)
          print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(i,vae_train_loss,vae_val_loss,gan_val_loss,d_loss,g_loss))
          f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(i,vae_train_loss,vae_val_loss,gan_val_loss,d_loss,g_loss))
    q3_eval.interpolate_latent(VAE, GAN, torch.randn(1,100).cuda(), torch.randn(1,100).cuda())
    q3_eval.interpolate_data(VAE, GAN, torch.randn(1,100).cuda(), torch.randn(1,100).cuda())
    q3_eval.compare4by4(VAE,GAN)
    q3_eval.wiggle_latent(VAE,GAN)
    q3_eval.populate_FID_folder(VAE,GAN)
#!python score_fid.py "./FID_folder/"