import torch 
import numpy as np
from config import GRID,config
def compute_mass_center(rho_field, GRID):
    mass = rho_field.sum(dim=[-3, -2, -1])
    X = GRID[..., 0]
    Y = GRID[..., 1]
    Z = GRID[..., 2]
    
    mass_center = torch.stack([
        (X * rho_field).sum(dim=[-3, -2, -1]),
        (Y * rho_field).sum(dim=[-3, -2, -1]),
        (Z * rho_field).sum(dim=[-3, -2, -1])
    ]) / mass
    
    return mass_center

