import torch 
import numpy as np
from scipy.spatial import ConvexHull
from config import GRID
from typing import *
from collections import deque
from scipy.ndimage import binary_erosion

def to_numpy64(input_):
    try:
        return input_.clone().detach().cpu().numpy().reshape(64,64,64)
    except:
        # if numpy
        return input_.reshape(64,64,64)
def get_smooth_bottom_contact_region(rho, ground_level,exp = 4):
    rho_used = rho[...,ground_level:]
    weight = torch.arange(0,rho_used.shape[-1],device=rho.device)
    weight = exp**(-weight)
    weight = weight.detach().clone()
    rho_used = weight*rho_used
    z_plane = rho_used.max(dim=-1).values
    return z_plane  

def is_neighboor_filled(rho, k = 1 ,directions = ["x+","x-","y+","y-","z+","z-"]):
    """
    Check if the given point is a neighbor of a filled point in the density field.
    Args:
        rho (torch.Tensor): A tensor of shape (N,N,N) representing the density field.
        k (int, optional): The radius of the neighborhood to check. Defaults to 1.
    Returns:
        torch.Tensor: A tensor of shape (N,N,N) indicating whether each point is a neighbor of a filled point.
    """
    assert k>=1
    with torch.no_grad():
        filled = torch.zeros_like(rho)
        filled[rho > 0.5] = 1
        filled[rho <= 0.5] = 0
        x_shift_neg = torch.roll(filled, shifts=1, dims=0)
        x_shift_pos = torch.roll(filled, shifts=-1, dims=0)
        y_shift_neg = torch.roll(filled, shifts=1, dims=1)
        y_shift_pos = torch.roll(filled, shifts=-1, dims=1)
        z_shift_neg = torch.roll(filled, shifts=1, dims=2)
        z_shift_pos = torch.roll(filled, shifts=-1, dims=2)
        neighbor_filled = torch.zeros_like(rho)
        if "x+" in directions:
            neighbor_filled += x_shift_pos
        if "x-" in directions:
            neighbor_filled += x_shift_neg
        if "y+" in directions:
            neighbor_filled += y_shift_pos
        if "y-" in directions:
            neighbor_filled += y_shift_neg
        if "z+" in directions:
            neighbor_filled += z_shift_pos
        if "z-" in directions:
            neighbor_filled += z_shift_neg
        neighbor_filled[neighbor_filled > 0.99]=1
        neighbor_filled[rho>0.5]=1
        if k <= 1:
            return neighbor_filled 
        else:
            return is_neighboor_filled(neighbor_filled,k-1,directions)


def find_the_fill_and_shell(rho,return_type:["grid","mask"],device):
    """
    Find the solid object and its shell.
    """
    rho_ = to_numpy64(rho)
    surface_coords = np.argwhere(rho_ > 0.5)
    # print(surface_coords)
    fill = full_voxel(surface_coords)
    shell = shell_voxel(fill)
    if return_type == "grid":
        return fill,shell
    elif return_type == "mask":
        zeros = np.zeros((64,64,64),dtype=bool)
        fill_mask = zeros.copy()
        fill_mask[fill[:,0],fill[:,1],fill[:,2]] = True
        shell_mask = zeros.copy()
        shell_mask[shell[:,0],shell[:,1],shell[:,2]] = True
        fill_mask = fill_mask.reshape(1,1,64,64,64)
        shell_mask = shell_mask.reshape(1,1,64,64,64)
        fill_mask = torch.from_numpy(fill_mask).to(device)
        shell_mask = torch.from_numpy(shell_mask).to(device)
        return fill_mask,shell_mask
    return fill, shell

def bds_for_64_64_field(plane):
    # expand_to_66_66
    enen, ex_mask = np.zeros((66,66)), np.zeros((66,66))
    enen[1:65,1:65]=plane
    graph = {}
    dirs = [(-1,0),(1,0),(0,1),(0,-1)]
    q = deque()
    all_q = set()
    for i in range(66):
        for j in range(66):
            if i ==0 or i==65 or j==0 or j==65:
                q.append((i, j))
                all_q.add((i,j))
            graph[(i,j)] = []
            for di in dirs:
                ii,jj = i+di[0],j+di[1]
                if ii>=0 and ii<=65 and jj>=0 and jj<=65:
                    if enen[i,j] == enen[ii,jj]:
                        graph[(i,j)].append((ii,jj))
    while q:
        coor = q.popleft()
        for zz in graph[coor]:
            if zz in all_q:
                pass
            else:
                q.append(zz)
                all_q.add(zz)
    for i,j in all_q:
        ex_mask[i,j] = 1
    return ex_mask[1:65,1:65] == 0
        
def full_voxel(surface_coords, grid_size=64, padding=1):
    new_size = grid_size + 2 * padding
    grid = np.zeros((new_size, new_size, new_size), dtype=np.uint8)
    surface_coords = np.array(surface_coords)
    shifted_coords = surface_coords + padding
    filled_voxels_z = np.zeros((64,64,64)).astype(bool)
    filled_voxels_x = np.zeros((64,64,64)).astype(bool)
    filled_voxels_y = np.zeros((64,64,64)).astype(bool)
    for coord in shifted_coords:
        x, y, z = coord
        grid[x, y, z] = 1
    for index in range(64):
        filled_voxels_z[:,:,index] = bds_for_64_64_field(grid[1:65,1:65,index+1])
        filled_voxels_x[index,:,:] = bds_for_64_64_field(grid[index+1,1:65,1:65])
        filled_voxels_y[:,index,:] = bds_for_64_64_field(grid[1:65,index+1,1:65])
    filled_voxels = ((1.0*filled_voxels_x) *(1.0* filled_voxels_y) *(1.0*filled_voxels_z))>0.5
    filled_voxels = np.argwhere(filled_voxels)
    return filled_voxels


def shell_voxel(full_voxels:np.array, grid_size=64):
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    full_voxels = np.array(full_voxels)
    grid[full_voxels[:,0], full_voxels[:,1], full_voxels[:,2]] = True
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True  
    structure[2, 1, 1] = True  
    structure[1, 0, 1] = True  
    structure[1, 2, 1] = True  
    structure[1, 1, 0] = True  
    structure[1, 1, 2] = True  
    eroded = binary_erosion(grid, structure=structure)
    shell = grid & ~eroded
    shell_voxels = np.argwhere(shell)
    return shell_voxels