import torch
import torchvision
import torch.utils.data
import numpy as np
import math


base_file_name = "binarized_mnist"

def get_BMNIST_dataloaders(root_dir = "data/", batch_size = 64, shuffle=True):
    """Get the dataloaders for the BMNIST datasets.

    Args:
        root_dir (str): The directory the .amat files are stored in.
        batch_size (int): The size of the minibatches that the dataloaders will draw.
        shuffle (bool): Wether or not the dataloaders should shuffle the datasets.

    Returns:
        dict: A dictionary containing the dataloaders for the training set ('train'),
                 validation set ('valid'), and testing set ('test').
    """
    dataloaders = {}
    for set_name in ['train','valid','test']:
        raw_inputs = np.loadtxt("{}{}_{}.amat".format(root_dir, base_file_name, set_name),delimiter=' ')
        
        tensor_inputs = torch.from_numpy(raw_inputs)
        dataset = torch.utils.data.TensorDataset(tensor_inputs)
        dataloaders[set_name] = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle=shuffle)
    return dataloaders


def KL_estimate(p_dist,q_dist):
    """Returns an estimate of the KL between samples from p and q respectively.

    Args:
        p_dist (torch.Tensor): The first distribution for the KL computation
        q_dist (torch.Tensor): The second distribution for the KL computation

    Returns:
        torch.Tensor : Tensor holding the value of the KL divergence between the two distributions
    """
    sample_vals = p_dist*torch.log(torch.div(p_dist,q_dist))
    return torch.sum(sample_vals)

def JS_estimate(p_dist,q_dist):
    r_dist = torch.div(p_dist+q_dist, torch.Tensor([2]))
    return 1/2 * (KL_estimate(p_dist,r_dist) + KL_estimate(q_dist,r_dist))

def JS_goal(x_logits,y_logits):
    first_expectation = torch.mean(torch.log(x_logits))
    second_expectation = torch.mean(torch.log(1-y_logits))
    return  - (first_expectation + second_expectation)

def WAS_goal(x_logits,y_logits, lambda_val, gradient):
    x_expectation = torch.mean(x_logits)
    y_expectation = torch.mean(y_logits)
    gradient_pen = lambda_val * ((torch.norm(gradient,p=2)-1)**2)
    return - (x_expectation - y_expectation - gradient_pen)

def compute_normal_pdf(x, mu, var):
    denominator = torch.sqrt(2 * math.pi * torch.sum(var))
    numerator = torch.exp( -(x-mu)**2 / (2*var))
    return torch.div(numerator, denominator)

def compute_log_normal_density(x, mu, var):
    return torch.log(compute_normal_pdf(x,mu,var))

BCE = torch.nn.BCELoss(reduction='none')
MSE = torch.nn.MSELoss(reduction='none')
def ELBO(x, model_output, est_mus, est_logvar):
    """Computes the ELBO of the given inputs and outputs.

    Returns the average ELBO of our model by taking the mean
    of the sum of the loss and the KL of our model

    Args:
        x (torch.Tensor): The input tensor which produced the model output
        model_output (torch.Tensor): The output from the model when given x
        est_mus (torch.Tensor): The estimated means in the latent space
        est_logvar (torch.Tensor): The estimated log variances in the latent space
    """
    # Take the loss for each input by summing the BCE for the individual inputs and respective outputs
    BCE_loss = -torch.sum(BCE(model_output.view(-1,model_output.shape[1]*model_output.shape[2]*model_output.shape[3]),x.view(-1,model_output.shape[1]*model_output.shape[2]*model_output.shape[3])),dim=1)

    # Take the KL divergence for each input as well
    KL_div = (0.5) * torch.sum( torch.exp(est_logvar) + est_mus**2 - 1 - est_logvar, dim=1)
    
    # Return the PER INSTANCE average ELBO by taking the average of the difference
    return -torch.mean(BCE_loss-KL_div)

def q3Loss(x, model_output, est_mus, est_logvar):
    # Take the loss for each input by summing the BCE for the individual inputs and respective outputs
    MSE_loss = -torch.sum(MSE(model_output.view(-1,model_output.shape[1]*model_output.shape[2]*model_output.shape[3]),x.view(-1,model_output.shape[1]*model_output.shape[2]*model_output.shape[3])),dim=1)

    # Take the KL divergence for each input as well
    KL_div = (0.5) * torch.sum( torch.exp(est_logvar) + est_mus**2 - 1 - est_logvar, dim=1)
    
    # Return the PER INSTANCE average ELBO by taking the average of the difference
    return -torch.mean(MSE_loss-KL_div)