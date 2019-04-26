#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt
from samplers import *
import utils


 
########### Question 1.1 ###########

class Discriminator(torch.nn.Module):
    def __init__(self,input_size):
        super(Discriminator,self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1)
        )
        self.act_fn = torch.nn.Sigmoid()

    def forward(self,x):
        return self.act_fn(self.features(x))

def train_JS(model, optimizer, p_dist, q_dist, n_epochs=5000, verbose=False):
    """Short Summary.

    Longer Summary

    Args:
        param1 (type): description
        param2 (type): description

    Returns:
        type: description
    """
    loss_fn = utils.JS_goal
    for i in range(n_epochs):
        p_samples = torch.Tensor(next(p_dist))
        q_samples = torch.Tensor(next(q_dist))
        p_logits = model(p_samples)
        q_logits = model(q_samples)
        loss = loss_fn(p_logits,q_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(verbose):
            if(i%(n_epochs/10) == 0):
                print(i,loss.item(), utils.JS_estimate(p_logits,q_logits))

def test_JS(model,p_dist,q_dist):
    """Short Summary.

    Longer Summary

    Args:
        param1 (type): description
        param2 (type): description

    Returns:
        type: description
    """
    p_samples = torch.Tensor(next(p_dist))
    q_samples = torch.Tensor(next(q_dist))
    p_logits = model(p_samples)
    q_logits = model(q_samples)
    return np.log(2) + (0.5*torch.mean(torch.log(p_logits)) + 0.5*torch.mean(torch.log(1 - q_logits)))

# discriminator = Discriminator(1)
# optimizer = torch.optim.SGD(discriminator.parameters(), lr=1e-3)
# batch_size = 512
# x_gen_1 = distribution4(batch_size=batch_size)
# x_gen_2 = distribution3(batch_size = batch_size)
# train_JS(discriminator,optimizer,x_gen_1,x_gen_2, 5000, True)


########### Question 1.2 ###########

class Critic(torch.nn.Module):
    def __init__(self, input_size):
        super(Critic,self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,2)
        )

    def forward(self,x):
        return self.features(x)

def train_Was(model, optimizer, p_dist, q_dist, n_epochs=5000, verbose=False):
    """Short Summary.

    Longer Summary

    Args:
        param1 (type): description
        param2 (type): description

    Returns:
        type: description
    """
    loss_fn = utils.WAS_goal
    was_lambda = 10
    a_samples = torch.distributions.uniform.Uniform(0,1)
    for i in range(n_epochs):
        p_samples = torch.Tensor(next(p_dist))
        q_samples = torch.Tensor(next(q_dist))
        p_logits = model(p_samples)
        q_logits = model(q_samples)

        a_values = a_samples.sample(torch.Size(p_samples.shape))
        augmented_inputs = torch.mul(a_values,p_samples) + torch.mul((1-a_values),q_samples)
        augmented_inputs.requires_grad = True
        augmented_logits = model(augmented_inputs)

        gradient_for_penalty = torch.autograd.grad(outputs=augmented_logits,inputs = augmented_inputs, grad_outputs=torch.ones(augmented_logits.size()), create_graph=True, only_inputs=True, retain_graph=True)[0]
        loss = loss_fn(p_logits, q_logits, was_lambda, gradient_for_penalty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(verbose):
            if(i%(n_epochs/10) == 0):
                print(i,loss.item())

def test_Was(model,p_dist,q_dist):
    """Short Summary.

    Longer Summary

    Args:
        param1 (type): description
        param2 (type): description

    Returns:
        type: description
    """
    p_samples = torch.Tensor(next(p_dist))
    q_samples = torch.Tensor(next(q_dist))
    p_logits = model(p_samples)
    q_logits = model(q_samples)   
    return torch.mean(p_logits) - torch.mean(q_logits)

# critic = Critic(1)
# optimizer = torch.optim.SGD(critic.parameters(), lr=1e-3)
# train_Was(critic,optimizer,x_gen_1,x_gen_2, 2500, True)


if __name__ == "__main__":

    print("Starting Question 1.3...")
    ########### Question 1.3 ###########

    phi_values = [np.round(phi, 1) for phi in np.arange(-1,1.1,0.1)]

    data_points = [[],[],[]]
    print("Experiment Results\nPhi, JS, WAS")
    for phi in phi_values:
        p_dist = distribution1(0,512)
        q_dist = distribution1(phi,512)
        
        discriminator = Discriminator(2)
        dis_optim = torch.optim.SGD(discriminator.parameters(),lr=1e-3)
        train_JS(discriminator,dis_optim,p_dist,q_dist,10000)
        js_val = test_JS(discriminator,p_dist,q_dist)
        critic = Critic(2)
        cri_optim = torch.optim.SGD(critic.parameters(),lr=1e-3)
        # train_Was(critic,dis_optim,p_dist,q_dist,5000)
        # was_val = test_Was(critic,p_dist,q_dist)
        data_points[0].append(phi)
        data_points[1].append(js_val)
        # data_points[2].append(was_val)
        
        print("{},{}".format(phi,js_val))#,was_val))
    plt.plot(data_points[0], data_points[1], '.')
    # plt.plot(data_points[0], data_points[2], '-')
    plt.xlim(-1, 1)
    plt.show()

    ########### Question 1.4 ###########
    print("Starting Question 1.4...")
    # plot p0 and p1
    plt.figure()

    # empirical
    xx = torch.randn(10000)
    f = lambda x: torch.tanh(x*2+1) + x*0.75
    d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
    plt.hist(f(xx), 100, alpha=0.5, density=1)
    plt.hist(xx, 100, alpha=0.5, density=1)
    plt.xlim(-5,5)
    # exact
    xx = np.linspace(-5,5,1000)
    N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
    plt.plot(xx, N(xx))

    ############### import the sampler ``samplers.distribution4'' 
    ############### train a discriminator on distribution4 and standard gaussian
    ############### estimate the density of distribution4

    #######--- INSERT YOUR CODE BELOW ---#######

    batch_size = 512

    f0 = distribution3(batch_size=batch_size)
    f1 = distribution4(batch_size=batch_size)

    n_training_epochs = 15000
    print("Training Disciminator for {} epochs...".format(n_training_epochs))
    density_estimator = Discriminator(1)
    optimizer = torch.optim.SGD(density_estimator.parameters(),lr=1e-3)
    train_JS(density_estimator, optimizer, f1, f0, n_training_epochs)

    tensor_xx = torch.Tensor(xx).view(len(xx),1)

    xx_estimates = torch.squeeze(density_estimator(tensor_xx),1).detach().numpy()
    approximated_dist = N(xx) * (xx_estimates / (1-xx_estimates))

    print("Plotting....")
    ############### plotting things
    ############### (1) plot the output of your trained discriminator 
    ############### (2) plot the estimated density contrasted with the true density

    r = xx_estimates # evaluate xx using your discriminator; replace xx with the output
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(xx,r)
    plt.title(r'$D(x)$')

    estimate = approximated_dist # estimate the density of distribution4 (on xx) using the discriminator; 
                                    # replace "np.ones_like(xx)*0." with your estimate
    plt.subplot(1,2,2)
    plt.plot(xx,estimate)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
    plt.legend(['Estimated','True'])
    plt.title('Estimated vs True')
    plt.show()











