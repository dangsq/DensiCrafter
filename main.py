import os
import random
import numpy as np
from config import config as global_config
from config import GRID
import torch
from obj import Rho
from geometry import find_the_fill_and_shell
from PIL import Image
import imageio
import trimesh
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from typing import *

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def mkdir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.mkdir(dir_name)
        except:
            os.makedirs(dir_name)
            
            
def trellis_stage1_multi_image(pipeline,seed,images,num_samples: int = 1,sampler_params: dict = {},):
    with torch.no_grad():
        images = [pipeline.preprocess_image(image) for image in images]
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
        sparse_structure_sampler_params = {}
        slat_sampler_params= {}
        cond = pipeline.get_cond(images)
        torch.manual_seed(seed)
        mode = 'stochastic'
        cond['neg_cond'] = cond['neg_cond'][:1]
        ss_steps = {**pipeline.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with pipeline.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = pipeline.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        return coords,None,cond
    
def trellis_stage2_mutil_image(pipeline,seed,images, coords, cond,slat_sampler_params: dict = {},formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],):
    with torch.no_grad():
        torch.manual_seed(seed)
        images = [pipeline.preprocess_image(image) for image in images]
        mode = 'stochastic'
        slat_steps = {**pipeline.slat_sampler_params, **slat_sampler_params}.get('steps')
        with pipeline.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = pipeline.sample_slat(cond, coords, slat_sampler_params)    
            return pipeline.decode_slat(slat, formats)

def trellis_stage1_single_image(pipeline,seed,image,num_samples,sampler_params: dict = {},):
    with torch.no_grad():
        image = pipeline.preprocess_image(image)
        cond = pipeline.get_cond([image])
        torch.manual_seed(seed)
        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
        sampler_params = {**pipeline.sparse_structure_sampler_params, **sampler_params}
        z_s = pipeline.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            ).samples
        decoder = pipeline.models['sparse_structure_decoder']
        rho_score = decoder(z_s)
        coords = torch.argwhere(rho_score>0)[:, [0, 2, 3, 4]].int()
        return coords, rho_score, cond
def trellis_stage2_single_image(pipeline,seed,coords, cond,slat_sampler_params: dict = {},formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],):
    with torch.no_grad():
        torch.manual_seed(seed)
        slat = pipeline.sample_slat(cond, coords, slat_sampler_params)
        return pipeline.decode_slat(slat, formats)
    
def trellis_stage1_text(pipeline,seed,text,num_samples,sampler_params: dict = {},):
    with torch.no_grad():
        cond = pipeline.get_cond([text])
        torch.manual_seed(seed)
        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
        sampler_params = {**pipeline.sparse_structure_sampler_params, **sampler_params}
        z_s = pipeline.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            ).samples

        # Decode occupancy latent
        decoder = pipeline.models['sparse_structure_decoder']
        rho_score = decoder(z_s)
        coords = torch.argwhere(rho_score>0)[:, [0, 2, 3, 4]].int()
        return coords, rho_score, cond
    
def trellis_stage2_text(pipeline,seed,coords, cond,slat_sampler_params: dict = {},formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],):
    with torch.no_grad():
        torch.manual_seed(seed)
        slat = pipeline.sample_slat(cond, coords, slat_sampler_params)
        return pipeline.decode_slat(slat, formats)   
    

def run_multi_image_task(pipeline, img_paths, result_dir):
    mkdir( result_dir )
    imgs = [Image.open(i) for i in img_paths]

    coords, rho_score, cond = trellis_stage1_multi_image(pipeline, 42, imgs,1)
    rho = torch.zeros(64,64,64).cuda()
    rho[coords[:,1],coords[:,2],coords[:,3]] = 1
        
    fill,surface = find_the_fill_and_shell(rho,"mask",global_config["device"])
    surface = rho > 0.5
    rho_obj = Rho(fill,surface)
    torch.save(fill,'fill.pt')
    torch.save(surface,'surface.pt')
        
    results = rho_obj.opt(global_config)
    final_filled = ((results["final_rho"]>0.5)*1+fill*1).bool()
    final_inner = (final_filled*1 - 1*(results["final_rho"]>0.5)).bool()
    inner_coords = torch.argwhere(final_inner)[:, [0, 2, 3, 4]].int()
    torch.save(inner_coords, os.path.join(result_dir,"inner_coords.pt"))
    final_coords = torch.argwhere(   ((1.0*final_filled-final_inner*1.0)>0.5).reshape(1,1,64,64,64))[:, [0, 2, 3, 4]].int()
    final_output = trellis_stage2_mutil_image(pipeline,42,imgs,final_coords,cond)
    final_glb = postprocessing_utils.to_glb(
                final_output['gaussian'][0],
                final_output['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
    final_glb.export(os.path.join(result_dir,"final.glb"))
    surface_mesh = trimesh.load(os.path.join(result_dir,"final.glb"), force='mesh')
    inner_mesh = Rho.build_inner_mesh(final_inner*1)
    combined_mesh = trimesh.util.concatenate([ surface_mesh, inner_mesh])
    combined_mesh.export(os.path.join(result_dir,"mesh_for_3d_print.stl"))
    surface_mesh.export(os.path.join(result_dir,"final_surface.stl"))
    inner_mesh.export(os.path.join(result_dir,"final_inner.stl"))



def run_single_image_task(pipeline, img_path, result_dir):
    mkdir( result_dir )
    img = Image.open(img_path)
    coords, rho_score, cond = trellis_stage1_single_image(pipeline, 42, img,1)
    rho = torch.zeros(64,64,64).cuda()
    rho[coords[:,1],coords[:,2],coords[:,3]] = 1
        
    fill,surface = find_the_fill_and_shell(rho,"mask",global_config["device"])
    surface = rho > 0.5
    rho_obj = Rho(fill,surface)
    torch.save(fill,'fill.pt')
    torch.save(surface,'surface.pt')
        
    results = rho_obj.opt(global_config)
    final_filled = ((results["final_rho"]>0.5)*1+fill*1).bool()
    final_inner = (final_filled*1 - 1*(results["final_rho"]>0.5)).bool()
    inner_coords = torch.argwhere(final_inner)[:, [0, 2, 3, 4]].int()
    torch.save(inner_coords, os.path.join(result_dir,"inner_coords.pt"))
    final_coords = torch.argwhere(   ((1.0*final_filled-final_inner*1.0)>0.5).reshape(1,1,64,64,64))[:, [0, 2, 3, 4]].int()
    final_output =  trellis_stage2_single_image(pipeline,42,final_coords,cond)
    final_glb = postprocessing_utils.to_glb(
                final_output['gaussian'][0],
                final_output['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
    final_glb.export(os.path.join(result_dir,"final.glb"))
    surface_mesh = trimesh.load(os.path.join(result_dir,"final.glb"), force='mesh')
    inner_mesh = Rho.build_inner_mesh(final_inner*1)
    combined_mesh = trimesh.util.concatenate([ surface_mesh, inner_mesh])
    combined_mesh.export(os.path.join(result_dir,"mesh_for_3d_print.stl"))
    surface_mesh.export(os.path.join(result_dir,"final_surface.stl"))
    inner_mesh.export(os.path.join(result_dir,"final_inner.stl"))
    
def run_text_task(pipeline, text_cond, result_dir):
    mkdir( result_dir )
    coords, rho_score,cond = trellis_stage1_text(pipeline, 42, text_cond,1)
    rho = torch.zeros(64,64,64).cuda()
    rho[coords[:,1],coords[:,2],coords[:,3]] = 1
        
    fill,surface = find_the_fill_and_shell(rho,"mask",global_config["device"])
    surface = rho > 0.5
    rho_obj = Rho(fill,surface)
    torch.save(fill,'fill.pt')
    torch.save(surface,'surface.pt')
        
    results = rho_obj.opt(global_config)
    final_filled = ((results["final_rho"]>0.5)*1+fill*1).bool()
    final_inner = (final_filled*1 - 1*(results["final_rho"]>0.5)).bool()
    inner_coords = torch.argwhere(final_inner)[:, [0, 2, 3, 4]].int()
    torch.save(inner_coords, os.path.join(result_dir,"inner_coords.pt"))
    final_coords = torch.argwhere(   ((1.0*final_filled-final_inner*1.0)>0.5).reshape(1,1,64,64,64))[:, [0, 2, 3, 4]].int()
    final_output =  trellis_stage2_text(pipeline,42,final_coords,cond)
    final_glb = postprocessing_utils.to_glb(
                final_output['gaussian'][0],
                final_output['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
    final_glb.export(os.path.join(result_dir,"final.glb"))
    surface_mesh = trimesh.load(os.path.join(result_dir,"final.glb"), force='mesh')
    inner_mesh = Rho.build_inner_mesh(final_inner*1)
    combined_mesh = trimesh.util.concatenate([ surface_mesh, inner_mesh])
    combined_mesh.export(os.path.join(result_dir,"mesh_for_3d_print.stl"))
    surface_mesh.export(os.path.join(result_dir,"final_surface.stl"))
    inner_mesh.export(os.path.join(result_dir,"final_inner.stl"))
    
    
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['SPCONV_DEBUG_SAVE_PATH'] = './debug/data'
os.environ["LD_LIBRARY_PATH"] = ""
os.environ['TORCH_HOME'] = './ckpts'


if __name__ == "__main__":
    root_path = "./pretrained-models/"
    
    pipeline_image = TrellisImageTo3DPipeline.from_pretrained(
           root_path+"TRELLIS-image-large"
        )
    pipeline_image.cuda()
    image_paths = ["./test_200_new/1/1_1.png", "./test_200_new/1/1_2.png", "./test_200_new/1/1_3.png"]
    run_multi_image_task(pipeline_image, image_paths, "results/multi-image/1")
    run_single_image_task(pipeline_image, image_paths[2], "results/single-image/1")

    
    pipeline_text = TrellisTextTo3DPipeline.from_pretrained(
            root_path+"TRELLIS-text-xlarge"
        )
    
    pipeline_text.cuda()    
    run_text_task(pipeline_text, "A stylized cartoon cat character with a spherical head, pointed ears, prominent eyes, a curved mouth, a cylindrical torso with a white patch, elongated arms with large hands, tapered legs, and rounded feet. Predominantly black with white details.", "results/text/1")
    
    
