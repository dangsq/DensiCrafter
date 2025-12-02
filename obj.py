import torch
import numpy as np
from config import config as global_config
from config import GRID
import matplotlib.pyplot as plt
import torchvision
import os
import time
from typing import *
from geometry import *
from loss import *
import tqdm
import torch
import numpy as np
from skimage.measure import marching_cubes
from physical import *
import trimesh

def to_numpy64(input_):
    try:
        return input_.clone().detach().cpu().numpy().reshape(64,64,64)
    except:
        # if numpy
        return input_.reshape(64,64,64)
    
class Rho:
    kernel = torch.ones(3,3,3)
    kernel = kernel/kernel.sum()
    kernel = kernel.to(global_config["device"])
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    max_,min_ = 10, -10
    soft_ground = global_config["epsilon"]
    
    
    # init_method = "surface" # or "fill"
    
    def __init__(self,fill_mask,surface_mask,init_method = "surface"):
        self.init_method = init_method
        self.rho_score = torch.randn((1,1,64,64,64),device=global_config["device"])
        self.rho_score = torch.where( fill_mask.bool() if self.init_method == "fill" else surface_mask.bool(), self.rho_score+self.max_, self.rho_score+self.min_)
        self.fill_mask = fill_mask
        self.surface_mask = surface_mask
        self.ori_rho_score = self.rho_score.clone().detach()
        coords = torch.argwhere(self.rho_score > 0)[:, [0, 2, 3, 4]].int()
        grid_obj = GRID[coords[:, 1], coords[:, 2], coords[:, 3]]
        self.ground_level = int(grid_obj[:,2].min())
        self.optimization_field = self.build_the_optimize_field().clone().detach()
        
    def build_the_optimize_field(self):
        
        ## bottum
        bb = torch.zeros(64,64,64).to(global_config["device"])
        bb[:,:,:self.ground_level+self.soft_ground+1] = 1
        buttom_field = (bb*self.fill_mask).reshape(64,64,64)
        k_order_buttom = is_neighboor_filled(buttom_field*1.0,k=self.soft_ground,
                                             directions = ["z+","x+","x-","y+","y-","z-"],
                                             ).bool()
        k_order_buttom[:,:,:self.ground_level] = False
        k_order_buttom[:,:,self.ground_level+self.soft_ground:]=False
        
        ## inner 
        inner_rho = (1*self.fill_mask)*(1-1*self.surface_mask)
        inner_field = inner_rho.bool()
        
        k_order_buttom = (k_order_buttom*1.0-self.surface_mask*1.0)>0.5
        optimization_field = (1*inner_field+1*k_order_buttom.reshape(self.ori_rho_score.shape)).bool()
        self.buttom_field = k_order_buttom
        return optimization_field
    
    @staticmethod
    def load_from_rho(rho):
        fill_mask,surface_mask = find_the_fill_and_shell(rho,return_type="mask",device = global_config["device"])
        return Rho(fill_mask,surface_mask)
    
    @staticmethod
    def load_from_rho_score(rho_score):
        rho = torch.sigmoid(rho_score)
        return Rho.load_from_rho(rho)
        
    @staticmethod
    def build_inner_mesh(inner_rho, apply_trans = True):
        # inner_rho should be float
        inner_rho = to_numpy64(inner_rho)    
        inner_rho = inner_rho/inner_rho.max()
        verts, faces, normals, values = marching_cubes(inner_rho, level= 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if apply_trans:
            mesh.vertices = (0.5+mesh.vertices)/64-0.5
            mesh.vertices = mesh.vertices[:,[0,2,1]]
            mesh.vertices[:,2] = -mesh.vertices[:,2]
        return mesh

    def opt(self,config = global_config):
        rho_score = self.rho_score.clone().detach()
        rho_score.requires_grad_(True)
        rho_init = torch.sigmoid(rho_score).clone().detach()
        optimizer = torch.optim.Adam([rho_score], lr=0.05)
        loss_recorder = []
        device = config["device"]
        results = {
            "loss":None, "rho":{},"final_rho":None,"best_rho":None
        }
        best_rho = None 
        best_total_loss = float('inf')
        t0 = time.time()
        with tqdm.tqdm(total=config["epoch"], desc="Training Progress") as pbar:
            for epoch in range(config["epoch"]):
                optimizer.zero_grad()
                rho_base = torch.sigmoid(rho_score)
                ks = torch.nn.functional.conv3d(rho_base, self.kernel, padding=1)
                rho = torch.where(self.surface_mask,rho_base,ks)
                # ks+=1
                if not config["enable_smoothing"]:
                    if epoch == 0:
                        print("[***] smoothing is disabled")
                    rho = rho_base
                else:
                    if epoch == 0:
                        print("[***] apply smoothing")
                rho = torch.where( 
                                  (self.optimization_field.reshape(rho.shape)
                                + self.fill_mask.reshape(rho.shape)).bool(),
                                rho, 
                                torch.zeros_like(rho).to(device))
                smooth_ground = get_smooth_bottom_contact_region(rho,self.ground_level).reshape(64,64)

                loss_step = {}
                total_loss = 0.0
                
                if config['lambda']['area'] > 0 or config['lambda']['z'] > 0 or config['lambda']['center'] > 0:
                    dis_loss,  area_loss, z_loss = physical_loss(smooth_ground,rho)
                    z_loss = z_loss -float(self.ground_level)
                if config['lambda']['mass'] > 0:
                    volume_loss = mass_penalty(rho)
                    total_loss += config['lambda']['mass'] * volume_loss.reshape(1)
                    loss_step['mass'] = volume_loss.item()*64*64*64
                if config['lambda']['area'] > 0:
                    total_loss += config['lambda']['area'] * area_loss.reshape(1)
                    loss_step['area'] = area_loss.item()
                if config['lambda']['z'] > 0:
                    total_loss += config['lambda']['z'] * z_loss.reshape(1)
                    loss_step['z'] = z_loss.item()
                if config['lambda']['center'] > 0:
                    total_loss += config['lambda']['center'] * dis_loss.reshape(1)
                    loss_step['center'] = dis_loss.item()
                loss_step['total'] = total_loss.item()
                loss_step['time'] = time.time() - t0
                
                loss_recorder.append(loss_step)
                if total_loss < best_total_loss:
                    best_total_loss = total_loss
                    best_rho = rho.clone().detach()
                    results["best_rho"] = best_rho
                total_loss.backward()
                rho_score.grad = torch.where(self.optimization_field,
                                            rho_score.grad,
                                            torch.zeros_like(rho_score.grad))
                optimizer.step()
                formatted_loss_step = {key: f'{value:.3f}' for key, value in loss_step.items()}
                pbar.set_postfix(**formatted_loss_step)
                pbar.update(1)
                if (epoch+1) % 500 == 0:
                    results["rho"][epoch] = rho.clone().detach()
        results["loss"] = loss_recorder
        results["final_rho"] = rho.clone().detach()
        return results         
        
    def reset(self,):
        self.rho_score = self.ori_rho_score.clone().detach()

    def to_rho(self,soft = False):
        if soft:
            rho_base = torch.sigmoid(rho_score) 
            ks = torch.nn.functional.conv3d(rho_base, kernel, padding=1)
            rho = torch.where( (self.full_field).bool(),rho_base,ks,)    
        return torch.sigmoid(rho_score) 



