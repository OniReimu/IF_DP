import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import grad
import argparse
import os
from torch.utils.data import DataLoader

def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)
    
    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)
        
    return return_grads


def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU/MPS

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    
    # Handle None gpu parameter and device selection
    if gpu is None:
        gpu = -1
    
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif gpu == -1 and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    z, t = z.to(device), t.to(device)
    y = model(z)
    loss = calc_loss(y, t)

    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.parameters() if p.requires_grad]
    
    return list(grad(loss, params, create_graph=True))


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    
    # Check for empty tensors which cause MPS issues
    if y.numel() == 0 or t.numel() == 0:
        return torch.tensor(0.0, device=y.device, requires_grad=True)
    
    # Handle MPS compatibility issues
    if y.device.type == 'mps':
        try:
            loss = torch.nn.functional.cross_entropy(y, t, reduction='mean')
        except RuntimeError as e:
            if "empty" in str(e).lower() or "placeholder" in str(e).lower():
                # Fallback to CPU for problematic MPS operations
                y_cpu = y.cpu()
                t_cpu = t.cpu()
                loss_cpu = torch.nn.functional.cross_entropy(y_cpu, t_cpu, reduction='mean')
                loss = loss_cpu.to(y.device)
            else:
                raise e
    else:
        loss = torch.nn.functional.cross_entropy(y, t, reduction='mean')
    
    return loss

def compute_hessian_inverse(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0, -1 for CPU/MPS
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, gpu)
    h_estimate = v.copy()
    
    # Handle None gpu parameter and device selection
    if gpu is None:
        gpu = -1
    
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif gpu == -1 and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        for batch_data in z_loader:
            x, t, _ = unpack_batch(batch_data)
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = calc_loss(y, t)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
       
    return h_estimate

def unpack_batch(batch_data):
    """Helper function to handle both (x, y) and (x, y, user_id) formats"""
    if len(batch_data) == 3:
        return batch_data[0], batch_data[1], batch_data[2]  # x, y, user_id
    else:
        return batch_data[0], batch_data[1], None  # x, y, None


