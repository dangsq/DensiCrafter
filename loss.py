import torch
import numpy as np
from physical import compute_mass_center
from config import GRID, config
import torch.nn.functional as F
import math

def mass_penalty(rho):
    return rho.mean()

def physical_loss(smooth_ground,rho):
    plane = GRID[:,:,0,:2]
    smooth_ground = smooth_ground.reshape(-1)
    plane = plane.reshape(-1,2)
    mass_center = compute_mass_center(rho, GRID)
    mass_center_plane_proj = mass_center[:2].reshape(1,2)
    dis = mass_center_plane_proj - plane
    dis = dis.norm(dim=-1)
    dis = dis.reshape(64,64)*smooth_ground.reshape(64,64)
    dis = dis.sum()/smooth_ground.sum()
    return dis, -smooth_ground.sum(), mass_center.reshape(3)[-1]
    
    
    
    