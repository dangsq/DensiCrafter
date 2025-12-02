import torch
import numpy as np
global config


config ={
    "device": "cuda",
    "epoch":2000,
    "lr":1e-2,
    "enable_smoothing":True,
    "epsilon":2,
    "lambda":{
        "z":1,
        "area":1,
        "center":1,
        "mass":100,
    }
}


    
def get_grid(x_dim, y_dim, z_dim, dtype=torch.float32, device='cpu'):
    """ Generate a 3D grid of points with the given dimensions.
    Args:
        x_dim (int): The number of points in the x dimension.
        y_dim (int): The number of points in the y dimension.
        z_dim (int): The number of points in the z dimension.
        dtype (torch.dtype, optional): The data type of the grid points. Defaults to torch.float32.
        device (str, optional): The device on which to create the grid. Defaults to 'cpu'.
    Returns:
        torch.Tensor: A 3D tensor of shape (x_dim, y_dim, z_dim, 3) representing the grid points."""
    X, Y, Z = np.mgrid[0:x_dim, 0:y_dim, 0:z_dim]
    X = torch.FloatTensor(X).to(dtype).to(device)
    Y = torch.FloatTensor(Y).to(dtype).to(device)
    Z = torch.FloatTensor(Z).to(dtype).to(device)
    merged_grid = torch.stack([X, Y, Z], dim=-1)
    return merged_grid
global GRID
GRID = get_grid(64,64,64, device=config['device'])
