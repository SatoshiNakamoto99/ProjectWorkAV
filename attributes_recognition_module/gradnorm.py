import torch
import numpy as np


def gradnorm(iters,  loss, layer, alpha, lr2, optimizer1, log = False):
    """
    Args:
        net (nn.Module): a multitask network with task loss
        data: 
        layer (nn.Module): a layers of the full network where appling GradNorm on the weights
        alpha (float): hyperparameter of restoring force
        lr2（float): learning rate of weights
        optimizer1 (): 
        log (bool): flag of result log
    """
    #loss.requires_grad = True
    # init log
    if log:
        log_weights = []
        log_loss = []
    
    
    # forward pass
    #loss = net(*data)
    # initialization
    if iters == 0:
        # init weights
        weights = torch.ones_like(loss)
        weights = torch.nn.Parameter(weights)
        T = weights.sum().detach() # sum of weights
        # set optimizer for weights
        optimizer2 = torch.optim.Adam([weights], lr=lr2)
        # set L(0)
        l0 = loss.detach()
    # compute the weighted loss
    weighted_loss = weights @ loss
    # clear gradients of network
    optimizer1.zero_grad()
    # backward pass for weigthted task loss
    weighted_loss.backward(retain_graph=True)
    # compute the L2 norm of the gradients for each task
    gw = []
    for i in range(len(loss)):
        #print("layer: " + str(layer.parameters()) + "\n")
        dl = torch.autograd.grad(weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True, allow_unused=True)[0]
        print(dl)
        #gw.append(torch.norm(dl))
    gw = torch.stack(gw)
    # compute loss ratio per task
    loss_ratio = loss.detach() / l0
    # compute the relative inverse training rate per task
    rt = loss_ratio / loss_ratio.mean()
    # compute the average gradient norm
    gw_avg = gw.mean().detach()
    # compute the GradNorm loss
    constant = (gw_avg * rt ** alpha).detach()
    gradnorm_loss = torch.abs(gw - constant).sum()
    # clear gradients of weights
    optimizer2.zero_grad()
    # backward pass for GradNorm
    gradnorm_loss.backward()
    # log weights and loss
    if log:
        # weight for each task
        log_weights.append(weights.detach().cpu().numpy().copy())
        # task normalized loss
        log_loss.append(loss_ratio.detach().cpu().numpy().copy())
    # update model weights
    optimizer1.step()
    # update loss weights
    optimizer2.step()
    # renormalize weights
    weights = (weights / weights.sum() * T).detach()
    weights = torch.nn.Parameter(weights)
    optimizer2 = torch.optim.Adam([weights], lr=lr2)
    # update iters
    iters += 1
    # get logs
    if log:
        return np.stack(log_weights), np.stack(log_loss)