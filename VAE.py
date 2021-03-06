import torch
import numpy as np
import utils
import matplotlib.pyplot as plt
from models import q2VAE as VAE
import time

def train_epoch(model, train_dataloader, optimizer, loss_fn):
    """Train the given model for one epoch.

    Longer Summary

    Args:
        model (torch.nn.Module): Module to use for the training process
        train_dataloader (torch.utils.data.DataLoader): Dataset which we use to train the model
        optimizer (torch.optim.Optimizer): Optimizer for the model's parameters utilizing backprop
        loss_fn (any loss function): The loss function we use to optimize the model
    """
    model.train()
    total_training_loss = 0
    for batch_index, batch in enumerate(train_dataloader):
        batch = batch[0].view(-1,1,28,28).float()
        output_batch = model(batch)
        loss = loss_fn(batch, output_batch, model.prev_means, model.prev_vars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_training_loss += loss

def validate(model,val_dataloader,loss_fn):
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
    
    for batch_index, batch in enumerate(val_dataloader):
        batch = batch[0].view(-1,1,28,28).float()
        output_batch = model(batch)
        total_loss += loss_fn(batch, output_batch, model.prev_means, model.prev_vars)

    total_loss *= float(val_dataloader.batch_size) / len(val_dataloader.dataset)
    return total_loss

def importance_sampling_function(model, xs, zs):
    """Computes the log probabilities of data given samples and a model

    Args:
        model (torch.nn.Module): Model to use for our importance sampling estimations
        xs (torch.Tensor): Tensor holding real inputs from our dataset
        zs (torch.Tensor): Samples of noise to use with our generator

    Returns:
        list: List of log probabilities of the input data given our model
    """
    M = xs.shape[0]
    D = xs.shape[1]
    K = zs.shape[1]
    L = zs.shape[2]

    assert(xs.shape[0] is zs.shape[0])

    importance_probabilities = []
    BCE_error = torch.nn.BCELoss(reduction='none')
    isotropic_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros(L), covariance_matrix = torch.eye(L))
    
    with torch.no_grad():
        for datum,latent_data in zip(xs,zs):
            datum = datum.float()
            # Run the datum through the network to set the values
            model_output = model(datum.view(1,1,28,28))

            # Set up the multivarite normal we will compare with our estimates
            est_mean, est_logvar = model.prev_means, model.prev_vars
            est_var = torch.exp(est_logvar/2)
            est_cov_matrix = torch.diag(est_var)
            normal_for_comparison = torch.distributions.multivariate_normal.MultivariateNormal(est_mean,est_cov_matrix)
            
            # Compute the probability for the datum
            p_datum = 0
            for latent_sample in latent_data:
                # Get the result of our sample
                model_output = model.decode(latent_sample.view(1,L)).view(28**2)
                # Compare the latent sample with the gaussian our model predicts
                model_prob = normal_for_comparison.log_prob(latent_sample)
                # Compare with an isotropic gaussian for importance sample
                iso_prob = isotropic_gaussian.log_prob(latent_sample)
                # Get the error of our sample with the datum
                prob_given_latent = -BCE_error(model_output,datum)
                # Since all of these are log probs, we add them up and then exponentiate
                p_datum += np.exp(prob_given_latent + model_prob - iso_prob)
            # Take the average prob and get the log (as per instructions_)
            importance_probabilities.append(np.log(p_datum/K))
    return importance_probabilities

def log_likelihood(model, dataloader, K=200):
    """Estimates the log likelihood of our model given a dataloader
    using importance sampling.

    Args:
        model (torch.nn.Module): Our model to find the log-likhood of
        dataloader (torch.utils.data.DataLoader): Dataset to test with

    Returns:
        float: the log likelihood estimate of our model over the dataset
    """
    total_sum = 0
    zs_batch = torch.randn((dataloader.batch_size, K, 100))
    for i, minibatch in enumerate(dataloader):
        minibatch = minibatch[0]
        total_sum += sum(importance_sampling_function(model, minibatch, zs_batch))
    return total_sum/len(dataloader)

if __name__ == "__main__":
    start_time = time.time()
    print("Creating Model ... ")
    my_VAE = VAE()
    optimizer = torch.optim.Adam(my_VAE.parameters(), lr=3e-4)
    loss_fn = utils.ELBO
    print("Loading Data ... ")
    loaders = utils.get_BMNIST_dataloaders()
    data_points = [[],[]]
    print("Starting Training ... ")
    for i in range(20):
        train_epoch(my_VAE,loaders['train'],optimizer,loss_fn)
        val_loss = validate(my_VAE,loaders['valid'],loss_fn)
        print("Epoch: {}\t ELBO: {}".format(i+1,-val_loss))
        data_points[0].append(i+1)
        data_points[1].append(-val_loss)
    plt.figure()
    plt.plot(data_points[0],data_points[1],label='Validation ELBO')
    plt.plot(data_points[0],[-96]*len(data_points[0]),label='Q2 Min Elbo')
    plt.show()
    print("Testing Model ...")
    print("Validation ELBO".format(-validate(my_VAE,loaders['valid'],loss_fn)))
    print("Test ELBO".format(-validate(my_VAE,loaders['test'],loss_fn)))
    print("Validation LL: {}".format(log_likelihood(my_VAE,loaders['valid'])))
    print("Test LL: {}".format(log_likelihood(my_VAE,loaders['test'])))
    print("Running Time of Script: {}".format(time.time()-start_time))
    